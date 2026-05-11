from ingestion.extractor import extract_health_data, sanitize_output
from ingestion.canonicalizer import (
    resolve_or_create,
    normalize_concept,
    normalize_body_part,
    resolve_body_node_pairs
)
from ingestion.episodes import attach_episode
from datetime import datetime, timedelta, timezone

def _utc_now():
    return datetime.now(timezone.utc)


def get_event_time(data, reference_time=None):
    """
    Converts the LLM-provided day offset into real-world event time.
    """
    reference_time = reference_time or _utc_now()
    day_offset = data.get("time_reference")

    if isinstance(day_offset, int):
        return (reference_time + timedelta(days=day_offset)).isoformat()

    return reference_time.isoformat()


def get_temporal_metadata(data, reference_time=None):
    reference_time = reference_time or _utc_now()
    day_offset = data.get("time_reference")

    metadata = {
        "time_reference": day_offset,
        "event_day_offset": day_offset,
        "ingestion_time": reference_time.isoformat(),
        "event_time": get_event_time(data, reference_time),
        "temporal_status": "past_or_present",
        "duration_start": None,
    }

    if isinstance(day_offset, int) and day_offset > 0:
        metadata["temporal_status"] = "future"

    return metadata


def is_valid_event(data):
    event_type = data.get("event_type")
    if not event_type:
        return False

    if event_type == "symptom":
        return bool(data.get("symptom"))

    return bool(data.get("symptom") or data.get("body_parts") or data.get("trigger"))


def ingest_text(graph, user_input, reference_time=None):
    # Step 1: Extract
    data = sanitize_output(extract_health_data(user_input))

    if data is None:
        print("❌ Extraction failed")
        return None

    if not is_valid_event(data):
        print("❌ Extraction missing required event structure")
        return None
    
    temporal_metadata = get_temporal_metadata(data, reference_time)
    event_time = temporal_metadata["event_time"]
    symptom = normalize_concept(data["symptom"])
    body_parts = [
        normalize_body_part(body_part)
        for body_part in data.get("body_parts", [])
        if normalize_body_part(body_part)
    ]

    # Step 2: Resolve nodes
    symptom_id = resolve_or_create(graph, symptom, "state")

    body_pairs = resolve_body_node_pairs(
        graph,
        body_parts,
        data["laterality"]
    )

    trigger_id = resolve_or_create(graph, data["trigger"], "state")

    # Step 3: Create event node

    event_id = graph.add_node(
        node_type="event",
        name=f"{data['event_type']}_event",
        canonical_name=f"{data['event_type']}_event",
        metadata={
            "raw_text": user_input,
            **temporal_metadata,
            "severity": data["severity"]
        }
    )

    attach_episode(
        graph,
        symptom,
        body_parts,
        event_id,
        event_time
    )

    # Step 4: Create edges

    # event → symptom
    if symptom_id:
        graph.add_edge(event_id, symptom_id, "HAS_SYMPTOM")

    for pair in body_pairs:
        specific_body_id = pair["specific_id"]
        base_body_id = pair["base_id"]

        if specific_body_id:
            graph.add_edge(event_id, specific_body_id, "TARGETS")
        elif base_body_id:
            graph.add_edge(event_id, base_body_id, "TARGETS")

        if specific_body_id and base_body_id:
            graph.add_edge(specific_body_id, base_body_id, "PART_OF")

    # event → trigger
    if trigger_id:
        graph.add_edge(event_id, trigger_id, "TRIGGERED_BY")

    return event_id
