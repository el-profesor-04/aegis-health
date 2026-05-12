import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient


warnings.filterwarnings("ignore", category=FutureWarning)

DB_PATH = Path(os.getenv("AEGIS_DB_PATH", "data/aegis_demo.sqlite3"))
os.environ["AEGIS_DB_PATH"] = str(DB_PATH)

from app.main import app  # noqa: E402
from app.sqlite_graph import SQLiteHealthGraph  # noqa: E402


client = TestClient(app)


inputs = [
    (0, "I was diagnosed with Type 2 diabetes about two years ago and have been taking metformin every single day since."),
    (0, "I have a severe peanut allergy that I've had since childhood. Diagnosed and confirmed."),
    (1, "Slept terribly last night, maybe 4 hours at most. Feel awful and groggy this morning, can barely function."),
    (1, "Had my morning coffee but skipped breakfast entirely."),
    (2, "Did my usual Sunday morning run today, about 5km. Felt decent, nothing special."),
    (2, "Feeling really anxious and on edge this afternoon. Hard to concentrate on anything, not sure what triggered it."),
    (3, "Twisted my left ankle badly during football practice this evening. It's quite swollen and really painful to walk on."),
    (4, "Left ankle still very sore and swollen today. Took ibuprofen to manage the pain. Had to completely skip my workout."),
    (5, "Headache for most of the afternoon. Realized I'd only had one small glass of water the entire day."),
    (6, "This is the fourth night in a row with under 5 hours of sleep because of back-to-back work deadlines. Completely burnt out and can barely keep my eyes open."),
    (7, "My mood has been really low all week. No energy, zero motivation, been skipping workouts and eating junk food."),
    (9, "Sunday run done. Ankle is still a bit tender so I took it easy and only did about 3km."),
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


queries = [
    "Could the leftover chicken have caused my nausea?",
    "What should I know about my migraines recently?",
    "Is my Sunday running pattern connected to the ankle issue?",
    "What chronic issues should stay relevant in my health memory?",
]


def api(method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path} failed: {response.status_code} {response.text}")
    return response.json()


def main():
    base_date = datetime(2026, 4, 1, 9, 0, 0, tzinfo=timezone.utc)
    query_time = datetime(2026, 4, 26, 18, 0, tzinfo=timezone.utc)

    print(f"\nAegis API demo DB: {DB_PATH}")

    print("\nClearing graph through DELETE /graph")
    print(api("DELETE", "/graph"))

    print(f"\nIngesting {len(inputs)} entries through POST /ingest")
    for index, (day_offset, text) in enumerate(inputs, 1):
        reference_time = base_date + timedelta(days=day_offset)
        result = api("POST", "/ingest", json={
            "text": text,
            "reference_time": reference_time.isoformat(),
        })
        print(
            f"[{index:02d}] day={day_offset:02d} "
            f"event={result['event_id']} "
            f"nodes={result['node_count']} edges={result['edge_count']}"
        )

    print("\nSummary from GET /graph/summary")
    summary = api("GET", "/graph/summary")
    print(summary)

    print("\nRecent events from GET /graph/events")
    for event in api("GET", "/graph/events", params={"limit": 5}):
        print(f"- {event['event_time']} | {event['event_type']} | {event['raw_text']}")

    print("\nReloading SQLite file directly to verify persistence")
    reloaded = SQLiteHealthGraph(DB_PATH)
    print(f"Reloaded rows: nodes={len(reloaded.nodes)} edges={len(reloaded.edges)}")

    print("\n=== API QUERY DEMO ===")
    for question in queries:
        result = api("POST", "/query", json={
            "question": question,
            "limit": 10,
            "query_time": query_time.isoformat(),
        })
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print("Memory pulled:")
        for memory in result["memories"][:3]:
            matched = ", ".join(memory.get("matched", [])) or "query match"
            print(f"  - {memory['text']}  (matched: {matched})")

    print("\nDone. Run the API server with:")
    print("  ../venv/bin/uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
