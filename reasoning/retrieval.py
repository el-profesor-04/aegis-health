import json
from collections import defaultdict
from datetime import datetime, timezone

from openai import OpenAI

from ingestion.canonicalizer import normalize_concept
from utils.embedding import cosine_similarity, get_embedding
from utils.relevance import score_node_relevance


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    max_retries=0,
)


QUERY_EXTRACTION_PROMPT = """
You extract search entities from a user's health question.

Return ONLY valid JSON. No markdown or explanation.

Schema:
{
  "entities": [
    {
      "text": string,
      "type": "symptom | body_part | trigger | food | medication | activity | condition | time | other"
    }
  ],
  "intent": "cause | timeline | recurrence | status | general"
}

Rules:
- Use concise SNOMED-style clinical terms where possible.
- Prefer singular canonical terms: "migraine" not "migraines", "headache" not "headaches".
- Include likely graph search anchors: symptoms, body parts, foods, medications, activities, conditions, and triggers.
- Do not include filler words.
- Keep each entity short.

Example:
Input: "Could sushi from yesterday have caused my nausea?"
Output:
{
  "entities": [
    {"text": "sushi", "type": "food"},
    {"text": "nausea", "type": "symptom"}
  ],
  "intent": "cause"
}
"""


TRAVERSABLE_RELATIONS = {
    "HAS_SYMPTOM",
    "TARGETS",
    "TRIGGERED_BY",
    "HAS_EVENT",
    "PART_OF",
}

RELATION_WEIGHTS = {
    "HAS_SYMPTOM": 1.20,
    "TARGETS": 1.05,
    "TRIGGERED_BY": 1.15,
    "HAS_EVENT": 1.10,
    "PART_OF": 0.80,
}

MIN_SEED_SIMILARITY = 0.75

STOPWORDS = {
    "about", "after", "again", "could", "from", "have", "health", "know",
    "memory", "recently", "should", "that", "the", "this", "what", "with",
    "your", "caused", "issues", "relevant", "stay", "simple", "question",
}

CHRONIC_QUERY_TERMS = {"chronic", "permanent", "long", "longterm", "long-term", "ongoing"}


def _metadata(row):
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _tokens(text):
    if not text:
        return set()

    cleaned = []
    for char in str(text).lower():
        cleaned.append(char if char.isalnum() else " ")

    return {
        token for token in "".join(cleaned).split()
        if len(token) > 2 and token not in STOPWORDS
    }


def _extract_first_json_object(text):
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:index + 1]
    return text


def extract_query_entities(query, model="gemma-4-e2b"):
    """
    LLM-backed query understanding for retrieval.

    If the local model is unavailable, falls back to a conservative token-based
    entity list so tests and offline development still work.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_EXTRACTION_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.strip().startswith("```")
            ).strip()
        data = json.loads(_extract_first_json_object(text))
        entities = [
            {
                "text": normalize_concept(item.get("text")),
                "type": item.get("type") or "other",
            }
            for item in data.get("entities", [])
            if normalize_concept(item.get("text"))
        ]
        return {
            "entities": _dedupe_entities(entities),
            "intent": data.get("intent") or "general",
            "source": "llm",
        }
    except Exception:
        fallback_entities = [
            {"text": token, "type": "other"}
            for token in sorted(_tokens(query))
        ]
        return {
            "entities": _dedupe_entities(fallback_entities),
            "intent": "general",
            "source": "fallback",
        }


def _dedupe_entities(entities):
    seen = set()
    deduped = []
    for entity in entities:
        text = normalize_concept(entity.get("text"))
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append({"text": text, "type": entity.get("type") or "other"})
    return deduped


def _term_variants(text):
    text = normalize_concept(text)
    if not text:
        return set()

    variants = {text}
    tokens = text.split()

    def singularize(token):
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 3 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    singular_tokens = [singularize(token) for token in tokens]
    singular_text = " ".join(singular_tokens)
    variants.add(singular_text)

    return {variant for variant in variants if variant}


def _iter_embedding_nodes(graph):
    for _, row in graph.nodes.iterrows():
        if row.get("embedding") is not None:
            yield row


def find_embedding_seed_nodes(graph, entities, top_k_per_entity=10):
    """
    Convert query entities to embeddings and find the closest graph nodes.
    Returns unique seed nodes, preserving each seed's best entity match.
    """
    best_by_node = {}

    for entity in entities:
        entity_text = entity["text"]
        try:
            query_embedding = get_embedding(entity_text)
        except Exception:
            continue

        scored = []
        for row in _iter_embedding_nodes(graph):
            try:
                score = cosine_similarity(query_embedding, row["embedding"])
            except Exception:
                continue
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)

        for score, row in scored[:top_k_per_entity]:
            name = str(row["canonical_name"]).lower()
            exactish = any(
                variant in name or name in variant
                for variant in _term_variants(entity_text)
            )
            if score < MIN_SEED_SIMILARITY and not exactish:
                continue

            node_id = row["node_id"]
            existing = best_by_node.get(node_id)
            if existing is None or score > existing["seed_score"]:
                best_by_node[node_id] = {
                    "node_id": node_id,
                    "name": row["canonical_name"],
                    "type": row["type"],
                    "seed_score": float(score),
                    "matched_entity": entity_text,
                    "matched_entity_type": entity.get("type", "other"),
                }

    seeds = list(best_by_node.values())
    seeds.sort(key=lambda item: item["seed_score"], reverse=True)
    return seeds


def _node_row(graph, node_id):
    row = graph.get_node(node_id)
    if row.empty:
        return None
    return row.iloc[0]


def _neighbors(graph, node_id):
    edges = graph.edges[
        (
            (graph.edges["source_id"] == node_id) |
            (graph.edges["target_id"] == node_id)
        ) &
        (graph.edges["relation"].isin(TRAVERSABLE_RELATIONS))
    ]

    neighbors = []
    for _, edge in edges.iterrows():
        if edge["source_id"] == node_id:
            other_id = edge["target_id"]
            direction = "out"
        else:
            other_id = edge["source_id"]
            direction = "in"

        neighbors.append({
            "node_id": other_id,
            "relation": edge["relation"],
            "direction": direction,
        })

    return neighbors


def _event_context(graph, event_id):
    edges = graph.edges[graph.edges["source_id"] == event_id]
    states = []
    for _, edge in edges.iterrows():
        if edge["relation"] not in {"HAS_SYMPTOM", "TARGETS", "TRIGGERED_BY"}:
            continue
        node = _node_row(graph, edge["target_id"])
        if node is None:
            continue
        states.append({
            "relation": edge["relation"],
            "node_id": node["node_id"],
            "name": node["canonical_name"],
        })
    return states


def _episode_ids(graph, event_id):
    edges = graph.edges[
        (graph.edges["target_id"] == event_id) &
        (graph.edges["relation"] == "HAS_EVENT")
    ]
    return list(edges["source_id"])


def _event_to_evidence(graph, event_row, graph_score, relevance_score, paths):
    md = _metadata(event_row)
    return {
        "event_id": event_row["node_id"],
        "name": event_row["canonical_name"],
        "raw_text": md.get("raw_text"),
        "event_type": event_row.get("event_type") or md.get("event_type"),
        "event_time": event_row.get("event_time") or md.get("event_time"),
        "event_day_offset": md.get("event_day_offset"),
        "severity_band": event_row.get("severity_band") or md.get("severity_band"),
        "impact_class": event_row.get("impact_class") or md.get("impact_class"),
        "impact_label": md.get("impact_label"),
        "S0": event_row.get("S0") or md.get("S0"),
        "lambda_hr": event_row.get("lambda_hr") or md.get("lambda_hr"),
        "is_cyclic": bool(event_row.get("is_cyclic") or md.get("is_cyclic", False)),
        "cyclic_period_hrs": event_row.get("cyclic_period_hrs") or md.get("cyclic_period_hrs"),
        "occurrence_count": event_row.get("occurrence_count") or md.get("occurrence_count"),
        "relevance_score": float(relevance_score),
        "graph_score": float(graph_score),
        "score": float(graph_score + relevance_score),
        "entity_match_score": 0.0,
        "matched_entities": [],
        "states": _event_context(graph, event_row["node_id"]),
        "episode_ids": _episode_ids(graph, event_row["node_id"]),
        "paths": paths[:3],
    }


def beam_search_events(graph, seeds, beam_width=20, max_depth=4, now=None):
    """
    Multi-start beam traversal over the health graph.

    Starts from embedding-matched state/episode nodes, walks edges in both
    directions, and collects event nodes encountered along the way. Event rank
    combines graph proximity/path score with temporal impact relevance.
    """
    now = now or datetime.now(timezone.utc)
    beam = []

    for seed in seeds:
        beam.append({
            "node_id": seed["node_id"],
            "score": seed["seed_score"],
            "depth": 0,
            "path": [{
                "node_id": seed["node_id"],
                "name": seed["name"],
                "type": seed["type"],
                "matched_entity": seed["matched_entity"],
                "seed_score": seed["seed_score"],
            }],
        })

    event_hits = {}
    best_seen = defaultdict(float)

    for _ in range(max_depth + 1):
        next_beam = []

        for item in beam:
            row = _node_row(graph, item["node_id"])
            if row is None:
                continue

            if row["type"] == "event":
                relevance = score_node_relevance(row.to_dict(), now)
                existing = event_hits.get(row["node_id"])
                path = item["path"]
                if existing is None:
                    event_hits[row["node_id"]] = {
                        "row": row,
                        "graph_score": item["score"],
                        "relevance_score": relevance,
                        "paths": [path],
                    }
                else:
                    existing["graph_score"] = max(existing["graph_score"], item["score"])
                    existing["relevance_score"] = max(existing["relevance_score"], relevance)
                    existing["paths"].append(path)

            if item["depth"] >= max_depth:
                continue

            for neighbor in _neighbors(graph, item["node_id"]):
                relation_weight = RELATION_WEIGHTS.get(neighbor["relation"], 1.0)
                depth_penalty = 0.72
                next_score = item["score"] * relation_weight * depth_penalty
                neighbor_id = neighbor["node_id"]

                if next_score <= best_seen[neighbor_id]:
                    continue
                best_seen[neighbor_id] = next_score

                neighbor_row = _node_row(graph, neighbor_id)
                if neighbor_row is None:
                    continue

                next_beam.append({
                    "node_id": neighbor_id,
                    "score": next_score,
                    "depth": item["depth"] + 1,
                    "path": item["path"] + [{
                        "node_id": neighbor_id,
                        "name": neighbor_row["canonical_name"],
                        "type": neighbor_row["type"],
                        "relation": neighbor["relation"],
                        "direction": neighbor["direction"],
                        "score": next_score,
                    }],
                })

        next_beam.sort(key=lambda item: item["score"], reverse=True)
        beam = next_beam[:beam_width]
        if not beam:
            break

    evidence = []
    for hit in event_hits.values():
        evidence.append(_event_to_evidence(
            graph,
            hit["row"],
            hit["graph_score"],
            hit["relevance_score"],
            hit["paths"],
        ))

    evidence.sort(key=lambda item: item["score"], reverse=True)
    return evidence


def build_evidence_bundle(graph, query, limit=20, now=None):
    now = now or datetime.now(timezone.utc)
    query_plan = extract_query_entities(query)
    seeds = []
    events = []

    if query_plan.get("source") != "fallback":
        seeds = find_embedding_seed_nodes(graph, query_plan["entities"])
        events = beam_search_events(graph, seeds, beam_width=20, max_depth=4, now=now)

    if not events:
        events = lexical_event_fallback(graph, query, now=now)

    events.extend(class_based_events(graph, query, now=now))
    events = _dedupe_events(events)
    events = rerank_events(events, query_plan, query)

    top_events = events[:limit]
    return {
        "query": query,
        "query_time": now.isoformat(),
        "query_plan": query_plan,
        "seed_nodes": seeds[:20],
        "events": top_events,
        "missing_information": _infer_missing_information(query_plan, top_events),
    }


def _dedupe_events(events):
    best = {}
    for event in events:
        event_id = event["event_id"]
        existing = best.get(event_id)
        if existing is None or event["score"] > existing["score"]:
            best[event_id] = event
    return list(best.values())


def class_based_events(graph, query, now=None):
    """
    Retrieve by stored memory class when the query asks about durable memory,
    chronic facts, or long-term relevance. This uses the rich graph columns
    directly instead of relying only on semantic entity search.
    """
    query_tokens = _tokens(query)
    if not (query_tokens & CHRONIC_QUERY_TERMS):
        return []

    now = now or datetime.now(timezone.utc)
    events = []
    class_events = graph.nodes[
        (graph.nodes["type"] == "event") &
        (graph.nodes["impact_class"] == "C5")
    ]

    for _, event in class_events.iterrows():
        relevance = score_node_relevance(event.to_dict(), now)
        events.append(_event_to_evidence(
            graph,
            event,
            graph_score=1.0,
            relevance_score=relevance,
            paths=[[{
                "node_id": event["node_id"],
                "name": event["canonical_name"],
                "type": "event",
                "matched_entity": "impact_class:C5",
                "seed_score": 1.0,
            }]],
        ))

    return events


def rerank_events(events, query_plan, query):
    reranked = []
    for event in events:
        entity_score, matched_entities = _entity_match_score(event, query_plan, query)
        event["entity_match_score"] = entity_score
        event["matched_entities"] = matched_entities

        graph_component = min(event.get("graph_score", 0.0), 1.25)
        relevance_component = min(event.get("relevance_score", 0.0), 1.0)
        class_bonus = _class_query_bonus(event, query)

        # Query match should dominate. Relevance is a memory prior, not a
        # reason to retrieve unrelated events.
        event["score"] = (
            entity_score * 3.0 +
            graph_component * 0.8 +
            relevance_component * 0.35 +
            class_bonus
        )

        # For LLM-planned queries, discard events that have no entity/class
        # explanation. This prevents high-relevance chronic or cyclic records
        # from crowding out the actual question.
        if query_plan.get("source") == "llm" and entity_score == 0 and class_bonus == 0:
            continue

        reranked.append(event)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked


def _entity_match_score(event, query_plan, query):
    entities = [
        entity for entity in query_plan.get("entities", [])
        if entity.get("type") != "time" and normalize_concept(entity.get("text"))
    ]

    if not entities:
        entities = [{"text": token, "type": "other"} for token in _tokens(query)]

    if not entities:
        return 0.0, []

    searchable_text = _event_search_text(event)
    searchable_tokens = _tokens(searchable_text)
    matched = []

    for entity in entities:
        text = normalize_concept(entity["text"])
        for variant in _term_variants(text):
            entity_tokens = _tokens(variant)
            if not entity_tokens:
                continue

            phrase_match = variant in searchable_text
            token_match = entity_tokens.issubset(searchable_tokens)
            partial_match = bool(entity_tokens & searchable_tokens) and len(entity_tokens) == 1

            if phrase_match or token_match or partial_match:
                matched.append(variant)
                break

    if not matched:
        return 0.0, []

    return len(set(matched)) / max(len(entities), 1), sorted(set(matched))


def _event_search_text(event):
    state_names = " ".join(state["name"] for state in event.get("states", []))
    return " ".join([
        str(event.get("name") or ""),
        str(event.get("raw_text") or ""),
        str(event.get("event_type") or ""),
        state_names,
    ]).lower()


def _class_query_bonus(event, query):
    query_tokens = _tokens(query)
    if (query_tokens & CHRONIC_QUERY_TERMS) and event.get("impact_class") == "C5":
        return 2.0
    if "recurring" in query_tokens and event.get("is_cyclic"):
        return 1.0
    return 0.0


def _infer_missing_information(query_plan, events):
    missing = []
    if not events:
        return ["No matching events were found in the graph."]

    has_trigger = any(
        state["relation"] == "TRIGGERED_BY"
        for event in events
        for state in event.get("states", [])
    )
    if query_plan.get("intent") == "cause" and not has_trigger:
        missing.append("No explicit trigger or exposure was linked to the retrieved events.")

    if not any(event.get("event_time") for event in events):
        missing.append("Retrieved events do not have reliable event_time values.")

    return missing


def retrieve_relevant_events(graph, query, limit=5, now=None):
    """
    Backward-compatible API: returns the top event evidence list.
    Prefer build_evidence_bundle() when calling the generator.
    """
    bundle = build_evidence_bundle(graph, query, limit=limit, now=now)
    return bundle["events"]


def lexical_event_fallback(graph, query, now=None):
    """
    Last-resort retrieval when embedding seeds are unavailable or empty.
    This keeps local tests and degraded mobile/offline modes usable, while the
    primary path remains embedding seed search + graph traversal.
    """
    now = now or datetime.now(timezone.utc)
    query_tokens = _tokens(query)
    events = []

    for _, event in graph.nodes[graph.nodes["type"] == "event"].iterrows():
        md = _metadata(event)
        raw_tokens = _tokens(md.get("raw_text", ""))
        state_names = {
            state["name"]
            for state in _event_context(graph, event["node_id"])
        }
        event_type = event.get("event_type") or md.get("event_type")

        score = len(query_tokens & raw_tokens)
        if event_type in query_tokens:
            score += 2
        for state_name in state_names:
            if state_name in query_tokens:
                score += 3

        if score <= 0:
            continue

        relevance = score_node_relevance(event.to_dict(), now)
        events.append(_event_to_evidence(
            graph,
            event,
            float(score),
            relevance,
            [[{
                "node_id": event["node_id"],
                "name": event["canonical_name"],
                "type": "event",
                "matched_entity": "lexical fallback",
                "seed_score": score,
            }]],
        ))

    events.sort(key=lambda item: item["score"], reverse=True)
    return events
