"""
Real end-to-end Aegis tests.

These tests intentionally depend on LM Studio being available at the OpenAI
compatible endpoint configured in the app code. They exercise the actual:

- extraction LLM
- embedding model
- graph writes
- cyclic detection
- retriever
- query router
- knowledge base
- generator

Run:
    pytest -q tests/test_ingestion_suite.py -s
"""

from datetime import datetime, timedelta, timezone

from graph.schema import HealthGraph
from ingestion.cyclic_detector import run_cyclic_detection
from ingestion.pipeline import ingest_text
from reasoning.engine import reason_about_query


BASE_TIME = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
QUERY_TIME = datetime(2026, 4, 26, 18, 0, tzinfo=timezone.utc)


JOURNAL = [
    (0, "I was diagnosed with Type 2 diabetes about two years ago and have been taking metformin every single day since."),
    (0, "I have a severe peanut allergy that I've had since childhood. Diagnosed and confirmed."),
    (2, "Did my usual Sunday morning run today, about 5km. Felt decent, nothing special."),
    (3, "Twisted my left ankle badly during football practice this evening. It's quite swollen and really painful to walk on."),
    (4, "Left ankle still very sore and swollen today. Took ibuprofen to manage the pain. Had to completely skip my workout."),
    (9, "Sunday run done. Ankle is still a bit tender so I took it easy and only did about 3km."),
    (10, "Felt nauseous after lunch today. Think it might have been the leftover chicken I had from two days ago."),
    (12, "Bad migraine hit around 3pm. Had strong light sensitivity and nausea. Had to lie down in a dark room for hours. This is the second one this month already."),
    (21, "Another migraine hit this afternoon, third one in three weeks. Same pattern as always: intense light sensitivity, nausea, had to stop working completely."),
    (24, "Blood sugar was elevated when I checked this morning. Need to be more careful with my diet. Took my metformin as usual."),
]


SCENARIOS = [
    {
        "name": "food_nausea_causality",
        "question": "Could the leftover chicken have caused my nausea?",
        "answer_any": ["chicken", "leftover", "nausea", "nauseous", "food"],
        "memory_any": ["chicken", "leftover", "nausea", "nauseous"],
    },
    {
        "name": "migraine_recurrence",
        "question": "What should I know about my migraines recently?",
        "answer_any": ["migraine", "recurring", "pattern", "light sensitivity", "nausea"],
        "memory_any": ["migraine", "light sensitivity", "nausea"],
        "min_memory_hits": 2,
    },
    {
        "name": "running_ankle_context",
        "question": "Is my Sunday running connected to the ankle issue?",
        "answer_any": ["ankle", "running", "run", "sprain", "football", "tender"],
        "memory_any": ["ankle", "sunday", "run", "football", "sprain", "tender"],
    },
    {
        "name": "chronic_background",
        "question": "What chronic health issues should I always keep in mind?",
        "answer_any": ["diabetes", "metformin", "peanut", "allergy", "blood sugar"],
        "background_any": ["diabetes", "metformin", "peanut", "allergy"],
    },
    {
        "name": "general_knowledge",
        "question": "What is metformin used for?",
        "answer_any": ["diabetes", "blood sugar", "glucose", "insulin", "type 2"],
        "expect_query_type": {"GENERAL", "HYBRID"},
    },
]


def _text_contains_any(text, terms):
    text = (text or "").lower()
    return any(term.lower() in text for term in terms)


def _joined_event_text(events):
    return " ".join(event.get("raw_text") or "" for event in events).lower()


def _joined_background_text(result):
    return " ".join(
        event.get("raw_text") or ""
        for event in result.get("evidence_bundle", {}).get("background_facts", [])
    ).lower()


def _print_result(scenario, result):
    print(f"\n\n=== {scenario['name']} ===")
    print(f"Q: {scenario['question']}")
    print(f"query_type: {result.get('query_type')}")
    print(f"A: {result.get('answer')}")
    print("Retrieved memories:")
    for event in result.get("events", [])[:5]:
        matched = ", ".join(event.get("matched_entities", [])) or "-"
        print(f"- matched=({matched}) text={event.get('raw_text')}")
    background = result.get("evidence_bundle", {}).get("background_facts", [])
    if background:
        print("Background facts:")
        for event in background[:5]:
            print(f"- {event.get('raw_text')}")
    if result.get("knowledge_hits"):
        print("Knowledge hits:")
        for fact in result["knowledge_hits"]:
            print(f"- {fact}")


def build_real_graph():
    graph = HealthGraph()
    failed = []

    for day_offset, text in JOURNAL:
        event_id = ingest_text(
            graph,
            text,
            reference_time=BASE_TIME + timedelta(days=day_offset),
        )
        if event_id is None:
            failed.append(text)
            continue
        run_cyclic_detection(graph)

    assert not failed, "Some journal entries failed ingestion:\n" + "\n".join(failed)
    assert len(graph.get_event_nodes()) >= len(JOURNAL) * 0.8
    return graph


def test_complex_journal_ingestion_and_questions_real_lmstudio():
    graph = build_real_graph()

    assert len(graph.nodes) > len(JOURNAL)
    assert len(graph.edges) > 0

    failures = []

    for scenario in SCENARIOS:
        result = reason_about_query(
            graph,
            scenario["question"],
            limit=15,
            now=QUERY_TIME,
        )
        _print_result(scenario, result)

        answer = result.get("answer", "")
        memories = result.get("events", [])
        memory_text = _joined_event_text(memories)
        background_text = _joined_background_text(result)

        expected_types = scenario.get("expect_query_type")
        if expected_types and result.get("query_type") not in expected_types:
            failures.append(
                f"{scenario['name']}: query_type={result.get('query_type')} expected one of {expected_types}"
            )

        if not _text_contains_any(answer, scenario["answer_any"]):
            failures.append(
                f"{scenario['name']}: answer did not contain any of {scenario['answer_any']!r}. "
                f"Answer: {answer!r}"
            )

        if scenario.get("memory_any") and not _text_contains_any(memory_text, scenario["memory_any"]):
            failures.append(
                f"{scenario['name']}: retrieved memories did not contain any of {scenario['memory_any']!r}. "
                f"Memories: {memory_text!r}"
            )

        if scenario.get("background_any") and not _text_contains_any(background_text, scenario["background_any"]):
            failures.append(
                f"{scenario['name']}: background facts did not contain any of {scenario['background_any']!r}. "
                f"Background: {background_text!r}"
            )

        min_memory_hits = scenario.get("min_memory_hits")
        if min_memory_hits and len(memories) < min_memory_hits:
            failures.append(
                f"{scenario['name']}: expected at least {min_memory_hits} retrieved memories, got {len(memories)}"
            )

    assert not failures, "\n\n".join(failures)


if __name__ == "__main__":
    try:
        print("Starting Aegis Ingestion Suite Test (Standalone)")
        test_complex_journal_ingestion_and_questions_real_lmstudio()
        print("\n✅ SUCCESS: All scenarios passed.")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
