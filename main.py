"""
main.py — Aegis full pipeline test

Ingests a realistic 25-event health journal, then fires 5 queries that
mirror how a real user would actually talk to a personal health assistant.
Each query has a graded coverage check — what MUST appear in the retrieved
memories for the answer to be grounded correctly.
"""

import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

DB_PATH = Path(os.getenv("AEGIS_DB_PATH", "data/aegis_demo.sqlite3"))
os.environ["AEGIS_DB_PATH"] = str(DB_PATH)

from fastapi.testclient import TestClient       # noqa: E402
from app.main import app                        # noqa: E402
from app.sqlite_graph import SQLiteHealthGraph  # noqa: E402

client = TestClient(app)


# ── Health journal ─────────────────────────────────────────────────────────
# 25 events over 25 days. Each (day_offset, text) where day_offset is
# relative to BASE_DATE = April 1 2026.

inputs = [
    (0,  "I was diagnosed with Type 2 diabetes about two years ago and have been taking metformin every single day since."),
    (0,  "I have a severe peanut allergy that I've had since childhood. Diagnosed and confirmed."),
    (1,  "Slept terribly last night, maybe 4 hours at most. Feel awful and groggy this morning, can barely function."),
    (1,  "Had my morning coffee but skipped breakfast entirely."),
    (2,  "Did my usual Sunday morning run today, about 5km. Felt decent, nothing special."),
    (2,  "Feeling really anxious and on edge this afternoon. Hard to concentrate on anything, not sure what triggered it."),
    (3,  "Twisted my left ankle badly during football practice this evening. It's quite swollen and really painful to walk on."),
    (4,  "Left ankle still very sore and swollen today. Took ibuprofen to manage the pain. Had to completely skip my workout."),
    (5,  "Headache for most of the afternoon. Realized I'd only had one small glass of water the entire day."),
    (6,  "This is the fourth night in a row with under 5 hours of sleep because of back-to-back work deadlines. Completely burnt out and can barely keep my eyes open."),
    (7,  "My mood has been really low all week. No energy, zero motivation, been skipping workouts and eating junk food."),
    (9,  "Sunday run done. Ankle is still a bit tender so I took it easy and only did about 3km."),
    (10, "Felt nauseous after lunch today. Think it might have been the leftover chicken I had from two days ago."),
    (11, "Had my morning coffee and actually ate a proper breakfast for once. Feeling slightly more human today."),
    (12, "Bad migraine hit around 3pm. Had strong light sensitivity and nausea. Had to lie down in a dark room for hours. This is the second one this month already."),
    (14, "Finally slept a full 8 hours last night. Woke up feeling genuinely rested for the first time in weeks."),
    (15, "Left ankle feels much better today, nearly back to normal. Just some mild stiffness when I get up in the morning."),
    (16, "Did my Sunday run today! Full 5km and the ankle held up completely fine. Great mood afterwards."),
    (17, "Accidentally ate something with hidden peanuts at a restaurant. My lips started swelling immediately and I had to use my EpiPen."),
    (18, "Still completely wiped out from yesterday's allergic reaction. Severe fatigue and feel drained, stayed home all day."),
    (20, "Went for a light evening walk, about 20 minutes. Mood has noticeably lifted compared to last week."),
    (21, "Another migraine hit this afternoon, third one in three weeks. Same pattern as always: intense light sensitivity, nausea, had to stop working completely."),
    (23, "Sunday run done. Easy 5km, felt strong. Ankle is completely fine now."),
    (24, "Blood sugar was elevated when I checked this morning. Need to be more careful with my diet. Took my metformin as usual."),
    (25, "Woke up with a stiff neck and a dull headache. Slept in a weird position. Also feeling a bit dehydrated."),
]


# ── Realistic test queries ─────────────────────────────────────────────────
# These are questions a real user would ask a personal health assistant.
# must_surface: list of strings. At least ONE must appear (as a word token)
#               in the retrieved memories for the test to pass.
# must_surface_all: if True, ALL strings must appear (default False).

queries = [
    {
        "question": "Why do I keep getting headaches? Is there a pattern?",
        "must_surface": ["headache", "water", "dehydrat"],
        "must_surface_all": False,
        "notes": (
            "Should find the afternoon headache linked to dehydration (1 glass "
            "of water) AND the recent stiff-neck headache. Dehydration is the "
            "common trigger — retriever needs the TRIGGERED_BY edge."
        ),
    },
    {
        "question": "I got sick after lunch — could it have been something I ate?",
        "must_surface": ["nauseous", "chicken", "lunch"],
        "must_surface_all": True,
        "notes": (
            "Should retrieve the nausea event AND the leftover-chicken trigger. "
            "Tests the TRIGGERED_BY edge between nausea event and state:leftover chicken."
        ),
    },
    {
        "question": "My ankle has been bothering me. Give me a full picture — what happened and am I back to normal?",
        "must_surface": ["ankle", "swollen", "football", "better"],
        "must_surface_all": False,
        "notes": (
            "Should surface the entire ankle arc: initial sprain (C3, football trigger), "
            "follow-up soreness, and the recovery note. Tests multi-hop episode traversal."
        ),
    },
    {
        "question": "I've been getting migraines pretty frequently. Should I be worried?",
        "must_surface": ["migraine", "light", "nausea"],
        "must_surface_all": False,
        "notes": (
            "Should surface both C4 migraine events (days 12 and 21). "
            "The pattern — third in three weeks, light sensitivity + nausea each time — "
            "is what makes this clinically notable. Tests C4 episode linking."
        ),
    },
    {
        "question": "I have a peanut allergy — what happened when I was exposed recently?",
        "must_surface": ["peanut", "swelling", "epipen"],
        "must_surface_all": False,
        "notes": (
            "Should surface the accidental peanut exposure (day 17, C5+High), "
            "the lip swelling symptom, the EpiPen use, AND the C5 allergy baseline. "
            "Tests C5 chronic context + acute triggered event linkage."
        ),
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────

def api(method, path, **kwargs):
    r = client.request(method, path, **kwargs)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}:\n{r.text}")
    return r.json()


def ingest_one(day, text, ref):
    """Ingest a single event, tolerating graceful skip (success=False)."""
    res = api("POST", "/ingest", json={
        "text":           text,
        "reference_time": ref.isoformat(),
    })
    return res


def _all_memory_text(memories):
    """Combine all retrieved memory texts into one lowercase blob."""
    return " ".join((m.get("text") or "").lower() for m in memories)


def _word_present(term, text):
    """
    True if `term` appears as a word (or word-prefix) in text.
    Handles:
      - exact word:    'headache' in 'had a headache'   → True
      - prefix match:  'dehydrat' matches 'dehydration' → True  (useful for stems)
      - substring:     'nausea'   matches 'nauseous'    → True  (handles inflections)
    """
    term = term.lower().strip()
    if not term:
        return False
    # Fast path: substring check (covers 'nausea' in 'nauseous', 'dehydrat' in 'dehydration')
    if term in text:
        return True
    return False


def _check_coverage(memories, must_surface, must_surface_all):
    text    = _all_memory_text(memories)
    results = {term: _word_present(term, text) for term in must_surface}

    if must_surface_all:
        passed = all(results.values())
    else:
        passed = any(results.values())   # at least one must hit

    return passed, results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    BASE_DATE  = datetime(2026, 4, 1, 9, 0, 0, tzinfo=timezone.utc)
    QUERY_TIME = datetime(2026, 4, 26, 18, 0, 0, tzinfo=timezone.utc)

    SEP  = "=" * 62
    DASH = "─" * 62

    print(f"\n{SEP}")
    print(f"  Aegis — Full Pipeline Test")
    print(f"  DB: {DB_PATH}")
    print(f"{SEP}\n")

    # ── Ingest ────────────────────────────────────────────────────────
    print("Clearing graph …")
    api("DELETE", "/graph")

    print(f"Ingesting {len(inputs)} events …\n")
    skipped = 0
    for idx, (day, text) in enumerate(inputs, 1):
        ref = BASE_DATE + timedelta(days=day)
        res = ingest_one(day, text, ref)

        if not res.get("success", True):
            skipped += 1
            reason = res.get("skip_reason", "unknown")
            print(f"  [{idx:02d}] day={day:02d}  ⚠️  SKIPPED — {reason}")
            print(f"         text: {text[:60]}…")
        else:
            eid = (res.get("event_id") or "")[:8]
            print(
                f"  [{idx:02d}] day={day:02d}  "
                f"nodes={res['node_count']}  edges={res['edge_count']}  "
                f"event={eid}…"
            )
    if skipped:
        print(f"\n  ⚠️  {skipped} event(s) skipped — check extractor / LLM output")

    # ── Graph summary ─────────────────────────────────────────────────
    print(f"\n{DASH}")
    summary = api("GET", "/graph/summary")
    print("Graph summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    reloaded = SQLiteHealthGraph(DB_PATH)
    print(f"\nPersistence check: nodes={len(reloaded.nodes)}  edges={len(reloaded.edges)}")

    # ── Retrieval test ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RETRIEVAL TEST")
    print(f"{SEP}")

    results_log = []

    for q in queries:
        print(f"\n{DASH}")
        print(f"Q: {q['question']}")
        print(f"   {q['notes']}")

        res      = api("POST", "/query", json={
            "question":   q["question"],
            "limit":      12,
            "query_time": QUERY_TIME.isoformat(),
        })
        memories = res.get("memories", [])
        qp       = res.get("query_plan", {})

        # Answer
        print(f"\n  Answer:\n    {res['answer']}")

        # Memories
        print(f"\n  Memories retrieved ({len(memories)}):")
        for m in memories:
            matched = ", ".join(m.get("matched") or []) or "—"
            text    = (m.get("text") or "")[:78]
            print(f"    [{m.get('event_type','?'):10}] [{m.get('impact_class','?'):2}] "
                  f"matched=({matched})")
            print(f"               \"{text}\"")

        # Coverage
        mode = "ALL" if q["must_surface_all"] else "ANY"
        passed, detail = _check_coverage(
            memories, q["must_surface"], q["must_surface_all"]
        )
        icon = "✅" if passed else "❌"
        print(f"\n  Coverage {icon} ({mode}): {detail}")
        print(f"  Query plan: intent={qp.get('intent')}  "
              f"entities={[e['text'] for e in qp.get('entities', [])]}")

        if res.get("missing_information"):
            print(f"  Missing info: {res['missing_information']}")

        results_log.append(passed)

    # ── Final verdict ─────────────────────────────────────────────────
    passed_count = sum(results_log)
    total        = len(results_log)
    all_pass     = passed_count == total

    print(f"\n{SEP}")
    print(f"  Results: {passed_count}/{total} passed  "
          f"{'✅  ALL PASSED' if all_pass else '❌  SOME FAILED'}")
    print(f"{SEP}\n")

    if not all_pass:
        print("Diagnosis hints:")
        for i, (q, passed) in enumerate(zip(queries, results_log)):
            if not passed:
                print(f"  Q{i+1}: '{q['question'][:55]}…'")
                print(f"        Must surface ({('ALL' if q['must_surface_all'] else 'ANY')}): "
                      f"{q['must_surface']}")
        print()

    print("Run API server:  uvicorn app.main:app --reload\n")


if __name__ == "__main__":
    main()