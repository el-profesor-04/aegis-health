from datetime import datetime, timezone

from graph.schema import HealthGraph
from ingestion.pipeline import ingest_text
from reasoning.engine import reason_about_query
from reasoning.retrieval import retrieve_relevant_events


REFERENCE_TIME = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


def fake_embedding(text):
    return str(text)


def fake_similarity(left, right):
    return 1.0 if left == right else 0.0


def patch_pipeline(monkeypatch, extracted):
    import ingestion.canonicalizer as canonicalizer
    import ingestion.episodes as episodes
    import ingestion.pipeline as pipeline

    monkeypatch.setattr(canonicalizer, "get_embedding", fake_embedding)
    monkeypatch.setattr(canonicalizer, "cosine_similarity", fake_similarity)
    monkeypatch.setattr(episodes, "get_embedding", fake_embedding)
    monkeypatch.setattr(episodes, "cosine_similarity", fake_similarity)
    monkeypatch.setattr(pipeline, "extract_health_data", lambda _: extracted)


def state_names(graph):
    return set(graph.nodes[graph.nodes["type"] == "state"]["canonical_name"])


def test_ingestion_splits_multiple_body_parts(monkeypatch):
    patch_pipeline(monkeypatch, {
        "event_type": "symptom",
        "symptom": "pain",
        "body_part": "neck and upper back",
        "laterality": None,
        "trigger": "sitting",
        "time_reference": 0,
        "severity": None,
    })

    graph = HealthGraph()
    event_id = ingest_text(
        graph,
        "My neck and upper back hurts after sitting all day",
        reference_time=REFERENCE_TIME,
    )

    assert event_id is not None
    assert {"pain", "neck", "upper back", "sitting"} <= state_names(graph)

    targets = graph.edges[
        (graph.edges["source_id"] == event_id) &
        (graph.edges["relation"] == "TARGETS")
    ]
    assert len(targets) == 2


def test_missing_symptom_does_not_create_none_episode(monkeypatch):
    patch_pipeline(monkeypatch, {
        "event_type": "symptom",
        "symptom": None,
        "body_part": "knee",
        "laterality": None,
        "trigger": None,
        "time_reference": 0,
        "severity": None,
    })

    graph = HealthGraph()

    assert ingest_text(graph, "Something weird", reference_time=REFERENCE_TIME) is None
    assert graph.nodes.empty


def test_temporal_grounding_keeps_ingestion_time_separate(monkeypatch):
    patch_pipeline(monkeypatch, {
        "event_type": "symptom",
        "symptom": "pain",
        "body_part": "knee",
        "laterality": "left",
        "trigger": "running",
        "time_reference": -1,
        "severity": 4,
    })

    graph = HealthGraph()
    event_id = ingest_text(graph, "Knee pain yesterday", reference_time=REFERENCE_TIME)
    metadata = graph.get_node(event_id).iloc[0]["metadata"]

    assert metadata["ingestion_time"] == "2026-05-03T12:00:00+00:00"
    assert metadata["event_time"] == "2026-05-02T12:00:00+00:00"
    assert metadata["event_day_offset"] == -1


def test_retrieval_and_reasoning_return_grounded_events(monkeypatch):
    graph = HealthGraph()

    patch_pipeline(monkeypatch, {
        "event_type": "food",
        "symptom": None,
        "body_part": None,
        "laterality": None,
        "trigger": "sushi",
        "time_reference": -1,
        "severity": None,
    })
    ingest_text(graph, "I had sushi yesterday", reference_time=REFERENCE_TIME)

    patch_pipeline(monkeypatch, {
        "event_type": "symptom",
        "symptom": "nausea",
        "body_part": None,
        "laterality": None,
        "trigger": "sushi",
        "time_reference": 0,
        "severity": 5,
    })
    ingest_text(graph, "Now I feel nauseous", reference_time=REFERENCE_TIME)

    events = retrieve_relevant_events(
        graph,
        "I think I got food poisoning from what I ate yesterday",
        now=REFERENCE_TIME,
    )
    reasoning = reason_about_query(
        graph,
        "I think I got food poisoning from what I ate yesterday",
    )

    assert events
    assert "sushi" in reasoning["candidate_triggers"]
