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


# ---------------------------------------------------------------------------
# Decay configuration
# ---------------------------------------------------------------------------

# Lambda (λ) per impact class: hourly decay constant for the formula
#   Relevance(t) = S0 * B * e^(-lambda_hr * t_hours)
#
# Values chosen so relevance drops below 5% of S0 at the practical
# end of each class's real-world lifespan:
#   C1 → gone in ~3 days     (λ = 0.030 → half-life 23 hrs)
#   C2 → gone in ~2 weeks    (λ = 0.006 → half-life ~5 days)
#   C3 → gone in ~4 months   (λ = 0.0009 → half-life ~32 days)
#   C4 → gone in ~3 years    (λ = 0.0001 → half-life ~9 months)
#   C5 → never decays        (λ = 0.0)
IMPACT_CLASS_CONFIG = {
    "C1": {"lambda_hr": 0.030,  "label": "Transient",  "half_life_hrs": 23},
    "C2": {"lambda_hr": 0.006,  "label": "Short-term", "half_life_hrs": 116},
    "C3": {"lambda_hr": 0.0009, "label": "Acute",      "half_life_hrs": 770},
    "C4": {"lambda_hr": 0.0001, "label": "Persistent", "half_life_hrs": 6931},
    "C5": {"lambda_hr": 0.0,    "label": "Chronic",    "half_life_hrs": None},
}

# S0 anchor values: initial relevance score at t=0
SEVERITY_BAND_S0 = {
    "Low":      0.30,
    "Moderate": 0.60,
    "High":     0.90,
}

# Used when severity_band is null — conservative mid value
DEFAULT_S0 = 0.50


def build_decay_metadata(data: dict) -> dict:
    """
    Derives all decay-related parameters from extracted impact_class
    and severity_band. Returns a flat dict to merge into event node metadata.

    Nothing is computed here — just the config values are looked up and
    stored on the node. The actual Relevance(t) calculation happens in
    the retriever (utils/relevance.py, added in Step 5).
    """
    impact_class = data.get("impact_class") or "C1"
    severity_band = data.get("severity_band")

    class_config = IMPACT_CLASS_CONFIG.get(impact_class, IMPACT_CLASS_CONFIG["C1"])
    S0 = SEVERITY_BAND_S0.get(severity_band, DEFAULT_S0)

    return {
        "impact_class":           impact_class,
        "impact_label":           class_config["label"],
        "severity_band":          severity_band,
        "S0":                     S0,
        "lambda_hr":              class_config["lambda_hr"],
        "cyclic_candidate":       data.get("cyclic_candidate", False),
        # C6 promotion fields — populated later by cyclic_detector.py
        "is_cyclic":              False,
        "cyclic_period_hrs":      None,
        # Accumulation boost — updated by cyclic_detector or episode logic
        "occurrence_count":       1,
        # Snapshot of relevance at the moment of ingestion (t=0 → e^0=1)
        "relevance_at_ingestion": S0,
    }


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

def get_event_time(data, reference_time=None):
    """Converts the LLM-provided integer day offset into an ISO timestamp."""
    reference_time = reference_time or _utc_now()
    day_offset = data.get("time_reference")

    if isinstance(day_offset, int):
        return (reference_time + timedelta(days=day_offset)).isoformat()

    return reference_time.isoformat()


def get_temporal_metadata(data, reference_time=None):
    reference_time = reference_time or _utc_now()
    day_offset = data.get("time_reference")

    metadata = {
        "time_reference":   day_offset,
        "event_day_offset": day_offset,
        "ingestion_time":   reference_time.isoformat(),
        "event_time":       get_event_time(data, reference_time),
        "temporal_status":  "past_or_present",
        "duration_start":   None,
    }

    if isinstance(day_offset, int) and day_offset > 0:
        metadata["temporal_status"] = "future"

    return metadata


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# These event types are valid on their own — the type itself carries meaning
# even when symptom / body_part / trigger are all null.
# e.g. "Slept 5 hours" → sleep event with no symptom is still useful.
SELF_CONTAINED_EVENT_TYPES = {"sleep", "mood", "food", "activity", "medication"}


def is_valid_event(data: dict) -> bool:
    event_type = data.get("event_type")
    if not event_type:
        return False

    if event_type in SELF_CONTAINED_EVENT_TYPES:
        return True

    if event_type == "symptom":
        return bool(data.get("symptom"))

    # "other" — needs at least one meaningful field
    return bool(data.get("symptom") or data.get("body_parts") or data.get("trigger"))


# ---------------------------------------------------------------------------
# Event node naming
# ---------------------------------------------------------------------------

def _build_event_canonical_name(data: dict) -> str:
    """
    Builds a specific canonical name for the event node.

    More useful for retrieval than the old flat "{type}_event" name.
    Examples:
      symptom + pain          → "symptom:pain"
      sleep  + insomnia       → "sleep:insomnia"
      food   + trigger=coffee → "food:coffee"
      activity (no symptom)   → "activity"
    """
    event_type = data.get("event_type", "other")
    symptom    = data.get("symptom")
    trigger    = data.get("trigger")

    if symptom:
        return f"{event_type}:{symptom}"
    if trigger:
        return f"{event_type}:{trigger}"
    return event_type


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_text(graph, user_input, reference_time=None):
    # Step 1: Extract and sanitize
    data = sanitize_output(extract_health_data(user_input))

    if data is None:
        print("❌ Extraction failed")
        return None

    if not is_valid_event(data):
        print("❌ Extraction missing required event structure")
        return None

    temporal_metadata = get_temporal_metadata(data, reference_time)
    event_time        = temporal_metadata["event_time"]
    decay_metadata    = build_decay_metadata(data)

    symptom    = normalize_concept(data.get("symptom"))
    body_parts = [
        normalize_body_part(bp)
        for bp in data.get("body_parts", [])
        if normalize_body_part(bp)
    ]

    # Step 2: Resolve state/entity nodes
    symptom_id = resolve_or_create(graph, symptom, "state")

    body_pairs = resolve_body_node_pairs(
        graph,
        body_parts,
        data.get("laterality")
    )

    trigger_id = resolve_or_create(graph, data.get("trigger"), "state")

    # Step 3: Create event node with full metadata
    canonical_name = _build_event_canonical_name(data)

    event_id = graph.add_node(
        node_type="event",
        name=canonical_name,
        canonical_name=canonical_name,
        metadata={
            "raw_text":   user_input,
            "event_type": data.get("event_type"),
            **temporal_metadata,
            **decay_metadata,
        }
    )

    # Step 4: Attach to episode (symptom-driven; gracefully skipped if no symptom)
    attach_episode(
        graph,
        symptom,
        body_parts,
        event_id,
        event_time
    )

    # Step 5: Create edges
    if symptom_id:
        graph.add_edge(event_id, symptom_id, "HAS_SYMPTOM")

    for pair in body_pairs:
        specific_body_id = pair["specific_id"]
        base_body_id     = pair["base_id"]

        if specific_body_id:
            graph.add_edge(event_id, specific_body_id, "TARGETS")
        elif base_body_id:
            graph.add_edge(event_id, base_body_id, "TARGETS")

        if specific_body_id and base_body_id:
            graph.add_edge(specific_body_id, base_body_id, "PART_OF")

    if trigger_id:
        graph.add_edge(event_id, trigger_id, "TRIGGERED_BY")

    return event_id