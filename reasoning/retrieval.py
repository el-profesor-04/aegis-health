"""
reasoning/retrieval.py
──────────────────────
Three-phase retrieval over the Aegis health graph.

Phase 1 — Seed finding
    LLM extracts entities from the query. Each entity is embedded and matched
    against all graph nodes (state nodes, episode nodes) by cosine similarity.
    Lexical string matching supplements embedding search for direct name hits.

Phase 2 — Beam traversal
    Multi-start graph traversal from all seeds simultaneously. Maintains a
    priority beam of the top BEAM_WIDTH nodes at each hop. Event nodes
    encountered at any depth are collected as candidates. Traversal is
    bidirectional (follows edges in both directions).

Phase 3 — Reranking
    Candidates are scored by three independent signals:
      A. Graph score      — path quality from beam traversal
      B. Temporal decay   — Relevance(t) from utils/relevance.py
      C. Semantic score   — cosine similarity between query embedding and
                            each event's raw_text embedding  ← KEY NEW PIECE

    C5 (Chronic) events are always surfaced as a supplementary band,
    regardless of graph proximity or query keywords.

    Final score = A * w_graph + B * w_temporal + C * w_semantic

    Top LIMIT events are returned to the generator.
"""

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
    max_retries=1,
    timeout=60.0,
)


# ── Scoring weights ────────────────────────────────────────────────────────
W_GRAPH    = 0.30   # beam traversal proximity
W_TEMPORAL = 0.20   # time-decay relevance (Relevance(t) formula)
W_SEMANTIC = 0.50   # query ↔ raw_text embedding similarity  ← dominates

# ── Beam parameters ────────────────────────────────────────────────────────
BEAM_WIDTH  = 60    # max nodes to carry between hops (was 20, too small)
MAX_DEPTH   = 5     # max hops from seed
DEPTH_DECAY = 0.85  # per-hop score multiplier (was 0.72, too steep)

# ── Seed finding ───────────────────────────────────────────────────────────
MIN_SEED_SIMILARITY = 0.72   # minimum embedding sim for a node to be a seed
TOP_K_PER_ENTITY    = 12     # candidate nodes per entity before sim filter

# ── Edge traversal ─────────────────────────────────────────────────────────
TRAVERSABLE_RELATIONS = {
    "HAS_SYMPTOM",
    "TARGETS",
    "TRIGGERED_BY",
    "HAS_EVENT",
    "PART_OF",
}

RELATION_WEIGHTS = {
    "HAS_SYMPTOM":  1.25,   # symptom links are semantically strong
    "TARGETS":      1.10,
    "TRIGGERED_BY": 1.20,   # causal links are important
    "HAS_EVENT":    1.15,   # episode → event is the core cluster edge
    "PART_OF":      0.80,   # anatomical hierarchy: lower priority
}

STOPWORDS = {
    "about", "after", "again", "could", "from", "have", "health", "know",
    "memory", "recently", "should", "that", "the", "this", "what", "with",
    "your", "caused", "issues", "relevant", "stay", "simple", "question",
    "also", "been", "does", "into", "more", "some", "then", "when",
}


# ── Query entity extraction ────────────────────────────────────────────────

QUERY_EXTRACTION_PROMPT = """
Extract search entities from a health question. Return ONLY JSON.

Schema:
{
  "entities": [{"text": "clinical term", "type": "symptom|body_part|trigger|food|medication|activity|other"}],
  "intent": "cause | timeline | recurrence | status | general"
}

Example: "Could sushi from yesterday have caused my nausea?"
{"entities": [{"text": "sushi", "type": "food"}, {"text": "nausea", "type": "symptom"}], "intent": "cause"}
"""


def _extract_first_json_object(text):
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return text


def _tokens(text):
    if not text:
        return set()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in str(text).lower())
    return {t for t in cleaned.split() if len(t) > 2 and t not in STOPWORDS}


def _dedupe_entities(entities):
    seen, deduped = set(), []
    for e in entities:
        text = normalize_concept(e.get("text"))
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append({"text": text, "type": e.get("type") or "other"})
    return deduped


def extract_query_entities(query, model="gemma-4-e2b"):
    """LLM-backed query understanding. Falls back to token extraction."""
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
                l for l in text.splitlines() if not l.strip().startswith("```")
            ).strip()
        data = json.loads(_extract_first_json_object(text))
        entities = [
            {"text": normalize_concept(e.get("text")), "type": e.get("type") or "other"}
            for e in data.get("entities", [])
            if normalize_concept(e.get("text"))
        ]
        return {
            "entities": _dedupe_entities(entities),
            "intent": data.get("intent") or "general",
            "source": "llm",
        }
    except Exception:
        fallback = [{"text": t, "type": "other"} for t in sorted(_tokens(query))]
        return {
            "entities": _dedupe_entities(fallback),
            "intent": "general",
            "source": "fallback",
        }


# ── Seed finding ───────────────────────────────────────────────────────────

def _term_variants(text):
    text = normalize_concept(text)
    if not text:
        return set()
    variants = {text}
    tokens = text.split()

    def singularize(t):
        if len(t) > 4 and t.endswith("ies"):
            return t[:-3] + "y"
        if len(t) > 3 and t.endswith("es"):
            return t[:-2]
        if len(t) > 3 and t.endswith("s"):
            return t[:-1]
        return t

    variants.add(" ".join(singularize(t) for t in tokens))
    return {v for v in variants if v}


def find_seed_nodes(graph, entities, top_k=TOP_K_PER_ENTITY):
    """
    Find graph nodes that best match query entities.
    Uses TWO passes:
      Pass 1 — Embedding similarity (all nodes with stored embeddings)
      Pass 2 — Lexical string matching (all nodes, regardless of embedding)
    Results are merged and deduplicated, keeping the best score per node.
    """
    best_by_node = {}

    for entity in entities:
        entity_text = entity["text"]
        variants    = _term_variants(entity_text)

        # ── Pass 1: Embedding similarity ──────────────────────────────
        try:
            q_emb   = get_embedding(entity_text)
            scored  = []
            for _, row in graph.nodes.iterrows():
                emb = row.get("embedding")
                if emb is None:
                    continue
                try:
                    sim = float(cosine_similarity(q_emb, emb))
                except Exception:
                    continue
                scored.append((sim, row))

            scored.sort(key=lambda x: x[0], reverse=True)

            for sim, row in scored[:top_k]:
                name = str(row["canonical_name"]).lower()
                exact = any(v in name or name in v for v in variants)
                if sim < MIN_SEED_SIMILARITY and not exact:
                    continue
                _update_best(best_by_node, row, float(sim), entity_text, entity.get("type"))
        except Exception:
            pass

        # ── Pass 2: Lexical string matching ───────────────────────────
        # Catches nodes whose embeddings weren't stored (rare after the
        # canonicalizer fix) and provides a hard anchor for exact matches.
        for _, row in graph.nodes.iterrows():
            name = str(row.get("canonical_name") or "").lower()
            if not name:
                continue
            if any(v in name or name in v for v in variants):
                # Assign a fixed high score for direct string hits so they
                # aren't beaten by weak embedding matches.
                _update_best(best_by_node, row, 0.90, entity_text, entity.get("type"))

    seeds = list(best_by_node.values())
    seeds.sort(key=lambda s: s["seed_score"], reverse=True)
    return seeds


def _update_best(best_by_node, row, score, entity_text, entity_type):
    node_id  = row["node_id"]
    existing = best_by_node.get(node_id)
    if existing is None or score > existing["seed_score"]:
        best_by_node[node_id] = {
            "node_id":             node_id,
            "name":                row["canonical_name"],
            "type":                row["type"],
            "seed_score":          score,
            "matched_entity":      entity_text,
            "matched_entity_type": entity_type or "other",
        }


# ── Graph traversal ────────────────────────────────────────────────────────

def _node_row(graph, node_id):
    row = graph.get_node(node_id)
    return None if row.empty else row.iloc[0]


def _neighbors(graph, node_id):
    edges = graph.edges[
        (
            (graph.edges["source_id"] == node_id) |
            (graph.edges["target_id"] == node_id)
        ) &
        graph.edges["relation"].isin(TRAVERSABLE_RELATIONS)
    ]
    neighbors = []
    for _, edge in edges.iterrows():
        other_id  = edge["target_id"] if edge["source_id"] == node_id else edge["source_id"]
        direction = "out" if edge["source_id"] == node_id else "in"
        neighbors.append({
            "node_id":  other_id,
            "relation": edge["relation"],
            "direction": direction,
        })
    return neighbors


def beam_search_events(graph, seeds, beam_width=BEAM_WIDTH,
                        max_depth=MAX_DEPTH, now=None):
    """
    Multi-start beam traversal. Collects all event nodes reachable from seeds
    within max_depth hops. Returns scored event candidates.
    """
    now = now or datetime.now(timezone.utc)

    # Initialise beam from all seeds simultaneously
    beam = [
        {
            "node_id": s["node_id"],
            "score":   s["seed_score"],
            "depth":   0,
            "path":    [{
                "node_id":        s["node_id"],
                "name":           s["name"],
                "type":           s["type"],
                "matched_entity": s["matched_entity"],
                "seed_score":     s["seed_score"],
            }],
        }
        for s in seeds
    ]

    event_hits  = {}     # node_id → best hit dict
    best_seen   = defaultdict(float)   # node_id → best score seen

    for _depth in range(max_depth + 1):
        next_beam = []

        for item in beam:
            row = _node_row(graph, item["node_id"])
            if row is None:
                continue

            # Collect event nodes encountered at any depth
            if row["type"] == "event":
                relevance = score_node_relevance(row.to_dict(), now)
                hit       = event_hits.get(row["node_id"])
                if hit is None:
                    event_hits[row["node_id"]] = {
                        "row":             row,
                        "graph_score":     item["score"],
                        "relevance_score": relevance,
                        "paths":           [item["path"]],
                    }
                else:
                    if item["score"] > hit["graph_score"]:
                        hit["graph_score"] = item["score"]
                    if relevance > hit["relevance_score"]:
                        hit["relevance_score"] = relevance
                    hit["paths"].append(item["path"])

            if item["depth"] >= max_depth:
                continue

            for neighbor in _neighbors(graph, item["node_id"]):
                rel_weight  = RELATION_WEIGHTS.get(neighbor["relation"], 1.0)
                next_score  = item["score"] * rel_weight * DEPTH_DECAY
                neighbor_id = neighbor["node_id"]

                if next_score <= best_seen[neighbor_id]:
                    continue
                best_seen[neighbor_id] = next_score

                n_row = _node_row(graph, neighbor_id)
                if n_row is None:
                    continue

                next_beam.append({
                    "node_id": neighbor_id,
                    "score":   next_score,
                    "depth":   item["depth"] + 1,
                    "path":    item["path"] + [{
                        "node_id":   neighbor_id,
                        "name":      n_row["canonical_name"],
                        "type":      n_row["type"],
                        "relation":  neighbor["relation"],
                        "direction": neighbor["direction"],
                        "score":     next_score,
                    }],
                })

        next_beam.sort(key=lambda x: x["score"], reverse=True)
        beam = next_beam[:beam_width]
        if not beam:
            break

    return list(event_hits.values())


# ── Event context helpers ──────────────────────────────────────────────────

def _metadata(row):
    v = row.get("metadata", {})
    return v if isinstance(v, dict) else {}


def _event_context(graph, event_id):
    edges  = graph.edges[graph.edges["source_id"] == event_id]
    states = []
    for _, edge in edges.iterrows():
        if edge["relation"] not in {"HAS_SYMPTOM", "TARGETS", "TRIGGERED_BY"}:
            continue
        node = _node_row(graph, edge["target_id"])
        if node is None:
            continue
        states.append({
            "relation": edge["relation"],
            "node_id":  node["node_id"],
            "name":     node["canonical_name"],
        })
    return states


def _episode_ids(graph, event_id):
    edges = graph.edges[
        (graph.edges["target_id"] == event_id) &
        (graph.edges["relation"] == "HAS_EVENT")
    ]
    return list(edges["source_id"])


def _build_evidence(graph, hit, query_sim=0.0):
    """Build the evidence dict the generator consumes."""
    row = hit["row"]
    md  = _metadata(row)
    return {
        "event_id":         row["node_id"],
        "name":             row["canonical_name"],
        "raw_text":         md.get("raw_text"),
        "event_type":       row.get("event_type") or md.get("event_type"),
        "event_time":       row.get("event_time") or md.get("event_time"),
        "event_day_offset": md.get("event_day_offset"),
        "severity_band":    row.get("severity_band") or md.get("severity_band"),
        "impact_class":     row.get("impact_class") or md.get("impact_class"),
        "impact_label":     md.get("impact_label"),
        "S0":               row.get("S0") or md.get("S0"),
        "lambda_hr":        row.get("lambda_hr") or md.get("lambda_hr"),
        "is_cyclic":        bool(row.get("is_cyclic") or md.get("is_cyclic", False)),
        "cyclic_period_hrs": row.get("cyclic_period_hrs") or md.get("cyclic_period_hrs"),
        "occurrence_count": row.get("occurrence_count") or md.get("occurrence_count"),
        "relevance_score":  float(hit["relevance_score"]),
        "graph_score":      float(hit["graph_score"]),
        "query_sim":        float(query_sim),
        "score":            0.0,    # filled in by rerank
        "entity_match_score": 0.0,
        "matched_entities": [],
        "states":           _event_context(graph, row["node_id"]),
        "episode_ids":      _episode_ids(graph, row["node_id"]),
        "paths":            hit["paths"][:3],
    }


# ── Semantic reranking ─────────────────────────────────────────────────────

def _score_semantic(events, query):
    """
    Embed the full query ONCE, then score each event's raw_text by cosine
    similarity. This is the key signal that was previously missing.
    Falls back gracefully if embedding fails.
    """
    try:
        q_emb = get_embedding(query)
    except Exception:
        for e in events:
            e["query_sim"] = 0.0
        return events

    for event in events:
        raw = event.get("raw_text") or ""
        if not raw:
            event["query_sim"] = 0.0
            continue
        try:
            r_emb = get_embedding(raw)
            event["query_sim"] = float(cosine_similarity(q_emb, r_emb))
        except Exception:
            event["query_sim"] = 0.0

    return events


def _entity_match_score(event, entities, query):
    """
    Token-based entity matching — kept as a lightweight supplement to
    semantic scoring. Only used as a bonus signal, not a gate.
    """
    if not entities:
        entities = [{"text": t, "type": "other"} for t in _tokens(query)]
    if not entities:
        return 0.0, []

    searchable = " ".join([
        str(event.get("name") or ""),
        str(event.get("raw_text") or ""),
        str(event.get("event_type") or ""),
        " ".join(s["name"] for s in event.get("states", [])),
    ]).lower()
    s_tokens = _tokens(searchable)

    matched = []
    for entity in entities:
        text = normalize_concept(entity["text"])
        if not text:
            continue
        for variant in _term_variants(text):
            v_tokens = _tokens(variant)
            if not v_tokens:
                continue
            if variant in searchable or v_tokens.issubset(s_tokens) or (
                len(v_tokens) == 1 and bool(v_tokens & s_tokens)
            ):
                matched.append(variant)
                break

    return (len(set(matched)) / max(len(entities), 1), sorted(set(matched)))


def rerank_events(events, query_plan, query):
    """
    Final scoring combining all three signals.
    No event is hard-discarded here — everything that made it through
    traversal gets a chance. Low-scoring events naturally fall to the bottom.
    """
    entities = [
        e for e in query_plan.get("entities", [])
        if e.get("type") != "time" and normalize_concept(e.get("text"))
    ]

    reranked = []
    for event in events:
        entity_score, matched = _entity_match_score(event, entities, query)
        event["entity_match_score"] = entity_score
        event["matched_entities"]   = matched

        g  = min(event.get("graph_score", 0.0), 1.5)
        t  = min(event.get("relevance_score", 0.0), 1.0)
        s  = event.get("query_sim", 0.0)
        em = entity_score   # 0–1 bonus for literal entity hits

        # C5 safety floor: guarantees chronic facts appear for direct
        # C5 queries (e.g. "what conditions do I have") but only adds
        # a small bump — semantic relevance still dominates.
        # Formula: floor is scaled by query_sim so it's only meaningful
        # when the C5 event is actually related to the query.
        is_c5 = event.get("impact_class") == "C5"
        c5_bonus = (0.10 + s * 0.15) if is_c5 else 0.0

        event["score"] = (
            s  * W_SEMANTIC  +
            t  * W_TEMPORAL  +
            g  * W_GRAPH     +
            em * 0.20        +
            c5_bonus
        )

        reranked.append(event)

    reranked.sort(key=lambda e: e["score"], reverse=True)

    # Drop events that scored below the noise floor.
    # Relevant events typically score 0.25–0.80; irrelevant ones cluster below 0.12.
    # This prevents ankle events appearing in headache answers etc.
    MIN_FINAL_SCORE = 0.10
    reranked = [e for e in reranked if e["score"] >= MIN_FINAL_SCORE]

    return reranked


# ── Chronic fact supplementation ───────────────────────────────────────────

def _always_on_c5_events(graph, now, query):
    """
    C5 (Chronic) events are always included in the bundle as a supplementary
    band. These are permanent health facts (diagnoses, allergies, long-term
    meds) that a health assistant should always be aware of.
    Previously gated by "chronic" keyword — now always runs.
    """
    now    = now or datetime.now(timezone.utc)
    q_emb  = None
    try:
        q_emb = get_embedding(query)
    except Exception:
        pass

    results = []
    c5_nodes = graph.nodes[
        (graph.nodes["type"] == "event") &
        (graph.nodes["impact_class"] == "C5")
    ]

    for _, row in c5_nodes.iterrows():
        md        = _metadata(row)
        relevance = score_node_relevance(row.to_dict(), now)
        query_sim = 0.0
        if q_emb is not None:
            raw = md.get("raw_text", "")
            try:
                query_sim = float(cosine_similarity(q_emb, get_embedding(raw)))
            except Exception:
                pass

        hit = {
            "row":             row,
            # C5 graph_score is driven purely by query similarity.
            # No free 0.80 — that was causing irrelevant chronic events
            # (e.g. peanut allergy when asking about nausea) to rank
            # above actually relevant events.
            "graph_score":     query_sim,
            "relevance_score": relevance,
            "paths":           [],
        }
        ev = _build_evidence(graph, hit, query_sim)
        results.append(ev)

    return results


# ── Lexical fallback ───────────────────────────────────────────────────────

def _lexical_fallback(graph, query, now):
    """Last resort when embedding seeds are empty (offline / model down)."""
    q_tokens = _tokens(query)
    results  = []

    for _, event in graph.nodes[graph.nodes["type"] == "event"].iterrows():
        md       = _metadata(event)
        raw_tok  = _tokens(md.get("raw_text", ""))
        score    = len(q_tokens & raw_tok)
        if score <= 0:
            continue
        relevance = score_node_relevance(event.to_dict(), now)
        results.append(_build_evidence(
            graph,
            {"row": event, "graph_score": float(score), "relevance_score": relevance, "paths": []},
            query_sim=0.0,
        ))

    results.sort(key=lambda e: e["graph_score"], reverse=True)
    return results


def _dedupe(events):
    best = {}
    for e in events:
        eid = e["event_id"]
        if eid not in best or e["score"] > best[eid]["score"]:
            best[eid] = e
    return list(best.values())


def _infer_missing(query_plan, events):
    missing = []
    if not events:
        return ["No matching events found in the health graph."]
    has_trigger = any(
        s["relation"] == "TRIGGERED_BY"
        for e in events
        for s in e.get("states", [])
    )
    if query_plan.get("intent") == "cause" and not has_trigger:
        missing.append("No explicit trigger was linked to the retrieved events.")
    return missing


# ── Public API ─────────────────────────────────────────────────────────────

def build_evidence_bundle(graph, query, limit=20, now=None):
    """
    Full retrieval pipeline. Returns a bundle dict consumed by the generator.

    The bundle separates two kinds of context:
      events          — top ranked, query-relevant events (beam + semantic)
      background_facts — C5 chronic facts, always included as context but
                         NOT competing in the main ranked list. This prevents
                         diabetes/allergy records from crowding out the events
                         that actually answer the current question.
    """
    now        = now or datetime.now(timezone.utc)
    query_plan = extract_query_entities(query)

    # ── Phase 1: Seeds ──────────────────────────────────────────────────
    seeds = find_seed_nodes(graph, query_plan["entities"])

    # ── Phase 2: Beam traversal ─────────────────────────────────────────
    hits  = beam_search_events(graph, seeds, now=now) if seeds else []

    if not hits:
        events = _lexical_fallback(graph, query, now)
    else:
        events = [_build_evidence(graph, h) for h in hits]

    # ── Phase 3: Semantic scoring + reranking ───────────────────────────
    events = _score_semantic(events, query)
    events = rerank_events(events, query_plan, query)
    top    = events[:limit]

    # ── Phase 4: C5 background facts (separate band, not ranked) ────────
    # Score them semantically so the generator knows which ones are most
    # relevant to the current question, but keep them out of the main list.
    background = _always_on_c5_events(graph, now, query)
    background = _score_semantic(background, query)
    background.sort(key=lambda e: e.get("query_sim", 0.0), reverse=True)

    return {
        "query":               query,
        "query_time":          now.isoformat(),
        "query_plan":          query_plan,
        "seed_nodes":          seeds[:20],
        "events":              top,
        "background_facts":    background,
        "missing_information": _infer_missing(query_plan, top),
    }


def retrieve_relevant_events(graph, query, limit=5, now=None):
    """Backward-compatible shim."""
    return build_evidence_bundle(graph, query, limit=limit, now=now)["events"]