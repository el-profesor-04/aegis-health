from datetime import datetime, timezone

from ingestion.canonicalizer import normalize_body_part, normalize_concept


RELEVANT_RELATIONS = {"HAS_SYMPTOM", "TARGETS", "TRIGGERED_BY"}


def _metadata(row):
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _tokens(text):
    if not text:
        return set()

    cleaned = []
    for char in str(text).lower():
        cleaned.append(char if char.isalnum() else " ")

    return {token for token in "".join(cleaned).split() if len(token) > 2}


def _parse_time(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _state_context(graph, event_id):
    edges = graph.edges[
        (graph.edges["source_id"] == event_id) &
        (graph.edges["relation"].isin(RELEVANT_RELATIONS))
    ]

    states = []
    for _, edge in edges.iterrows():
        node = graph.get_node(edge["target_id"])
        if node.empty:
            continue

        row = node.iloc[0]
        states.append({
            "relation": edge["relation"],
            "node_id": row["node_id"],
            "name": row["canonical_name"],
        })

    return states


def _episode_ids(graph, event_id):
    edges = graph.edges[
        (graph.edges["target_id"] == event_id) &
        (graph.edges["relation"] == "HAS_EVENT")
    ]
    return list(edges["source_id"])


def _query_concepts(query):
    concepts = set()
    query_tokens = _tokens(query)

    for token in query_tokens:
        concept = normalize_concept(token)
        body_part = normalize_body_part(token)

        if concept:
            concepts.add(concept)
        if body_part:
            concepts.add(body_part)

    return concepts


def _event_score(graph, event, query, query_tokens, query_concepts, now):
    md = _metadata(event)
    raw_text = md.get("raw_text", "")
    raw_tokens = _tokens(raw_text)
    states = _state_context(graph, event["node_id"])
    state_names = {state["name"] for state in states}

    score = 0.0
    overlap = query_tokens & raw_tokens
    score += len(overlap) * 1.0

    direct_state_matches = query_concepts & state_names
    score += len(direct_state_matches) * 3.0

    for state in states:
        if any(token in state["name"] or state["name"] in token for token in query_tokens):
            score += 1.5

    event_time = _parse_time(md.get("event_time"))
    if event_time and now:
        age_days = abs((now - event_time).days)
        if age_days <= 1:
            score += 1.0
        elif age_days <= 7:
            score += 0.5

    return score


def retrieve_relevant_events(graph, query, limit=5, now=None):
    now = now or datetime.now(timezone.utc)
    query_tokens = _tokens(query)
    query_concepts = _query_concepts(query)

    events = graph.nodes[graph.nodes["type"] == "event"]
    scored = []

    for _, event in events.iterrows():
        score = _event_score(graph, event, query, query_tokens, query_concepts, now)
        if score <= 0:
            continue

        md = _metadata(event)
        scored.append({
            "score": score,
            "event_id": event["node_id"],
            "name": event["canonical_name"],
            "raw_text": md.get("raw_text"),
            "event_time": md.get("event_time"),
            "event_day_offset": md.get("event_day_offset"),
            "ingestion_time": md.get("ingestion_time"),
            "time_reference": md.get("time_reference"),
            "severity": md.get("severity"),
            "states": _state_context(graph, event["node_id"]),
            "episode_ids": _episode_ids(graph, event["node_id"]),
        })

    scored.sort(key=lambda item: (item["score"], item.get("event_time") or ""), reverse=True)
    return scored[:limit]
