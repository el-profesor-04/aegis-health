"""
reasoning/engine.py
────────────────────
Query routing + answer generation for Aegis.

Three query types:
  PERSONAL  — question about the user's own health history → graph retrieval
  GENERAL   — general health/medical question → knowledge base
  HYBRID    — both (e.g. "is my ankle sprain normal?") → graph + knowledge base

The generator prompt is designed to:
  - Lead with the direct answer first
  - Surface patterns explicitly (frequency, trends, correlations)
  - Be proactive — connect dots the user didn't ask about
  - Stay concise: 3–5 sentences for simple questions, short paragraphs for complex ones
  - Ground every claim in the retrieved memories
  - Add a brief safety note when symptoms warrant it
"""

import json

from openai import OpenAI

from reasoning.retrieval import build_evidence_bundle
from reasoning.knowledge import query_knowledge_base

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    max_retries=1,
    timeout=60.0,  # Increase timeout to 60s for complex reasoning
)

# ── Generator prompt ───────────────────────────────────────────────────────

GENERATOR_PROMPT = """
You are Aegis, a health assistant. Give sharp, pattern-aware answers.

Structure:
1. Direct answer first.
2. Context/Pattern (logged events, frequency).
3. Likely connection (suggest possibilities, don't diagnose).
4. Proactive insight (optional).
5. Safety note (if serious).

Rules:
- Speak directly ("You").
- Weave medical knowledge naturally.
- Max 120 words.
- Don't say "I found" or "based on records".
"""

# ── Query router ───────────────────────────────────────────────────────────

ROUTER_PROMPT = """
Classify health question:
PERSONAL: user's history/patterns
GENERAL: medical condition/drug info
HYBRID: both personal and general

Return ONLY one word: PERSONAL, GENERAL, or HYBRID.
"""


def classify_query(query: str) -> str:
    query_l = query.lower()
    personal_memory_terms = [
        "my health memory",
        "my health",
        "my chronic",
        "chronic health issues",
        "should i always keep in mind",
        "what chronic",
        "what conditions do i have",
    ]
    if any(term in query_l for term in personal_memory_terms):
        return "PERSONAL"

    try:
        response = client.chat.completions.create(
            model="gemma-4-e2b",
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().upper()
        if result in {"PERSONAL", "GENERAL", "HYBRID"}:
            return result
        return "HYBRID"   # safe default
    except Exception:
        return "PERSONAL"  # safe fallback when LLM is unavailable


# ── Event summariser ───────────────────────────────────────────────────────

def _event_summary(event):
    states    = event.get("states", [])
    symptoms  = [s["name"] for s in states if s["relation"] == "HAS_SYMPTOM"]
    triggers  = [s["name"] for s in states if s["relation"] == "TRIGGERED_BY"]

    summary = {
        "text":                  event.get("raw_text"),
        "time":                  event.get("event_time")[:10] if event.get("event_time") else None,
        "type":                  _memory_type(event),
        "symptoms":              symptoms,
        "triggers":              triggers,
    }
    # Strip None or empty lists to save tokens
    return {k: v for k, v in summary.items() if v is not None and v != []}


def _memory_type(event):
    if event.get("is_cyclic"):
        return "recurring pattern"
    cls = event.get("impact_class")
    return {
        "C5": "chronic",
        "C4": "recurring",
        "C3": "acute",
        "C2": "recent-pattern",
        "C1": "transient",
    }.get(cls, "transient")


# ── Generator ──────────────────────────────────────────────────────────────

def _build_payload(bundle, knowledge_hits=None):
    events = bundle.get("events", [])
    background_facts = bundle.get("background_facts", [])

    # Aggressive truncation for small context windows (e.g. 4096)
    payload = {
        "query":             bundle["query"],
        "query_time":        bundle["query_time"],
        "personal_memories": [_event_summary(e) for e in events[:6]],
        "chronic_context":   [_event_summary(e) for e in background_facts[:3]],
    }

    if knowledge_hits:
        # Only take top 2 hits and truncate to 800 chars
        payload["medical_knowledge"] = [
            hit[:800] + "..." if len(hit) > 800 else hit
            for hit in knowledge_hits[:2]
        ]

    return payload


def generate_answer(bundle, knowledge_hits=None, model="gemma-4-e2b"):
    payload = _build_payload(bundle, knowledge_hits)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GENERATOR_PROMPT},
                {"role": "user",   "content": json.dumps(payload, indent=2)},
            ],
            temperature=0.25,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Generator error: {e}")
        return _fallback_answer(payload)


def _fallback_answer(payload):
    memories = payload.get("personal_memories", [])
    if not memories:
        return "I don't have enough information in your health history to answer this question."

    triggers = [t for m in memories for t in m.get("triggers", [])]
    top      = memories[0]
    parts    = []
    if top.get("raw_text"):
        parts.append(f"Your most relevant logged event: {top['raw_text']}")
    if triggers:
        parts.append(f"Possible related trigger(s): {', '.join(set(triggers))}.")
    return " ".join(parts) or "I found relevant history but couldn't generate a full answer."


# ── Clinical Summarizer ───────────────────────────────────────────────────

CLINICAL_SUMMARY_PROMPT = """
You are Aegis, preparing a clinical summary for a doctor's visit.
Your goal is to transform the user's messy health memory into structured, objective evidence.

sections:
1. CHRONIC BASELINE: List all C5 conditions and their associated medications.
2. ACTIVE CONCERNS: List ongoing C3/C4/C6 episodes, duration, and frequency.
3. RECENT ANOMALIES: Highlight any new C1/C2 symptoms or high-severity events from the last 14 days.
4. OBSERVED CORRELATIONS: Explicitly state links between triggers (food, stress, activity) and symptoms.

Rules:
- Be clinical, objective, and concise.
- Use bullet points.
- "The patient reports..." or "Data shows..."
- Do NOT diagnose. State "Pattern suggests potential link between X and Y."
- Max 250 words.
"""

def generate_clinical_summary(graph):
    # 1. Gather all high-level context from the graph
    events = graph.get_event_nodes()
    if events.empty:
        return "No health data recorded yet."

    # Sort and filter for efficiency
    chronic = events[events["impact_class"] == "C5"].to_dict('records')
    active  = events[events["impact_class"].isin(["C3", "C4", "C6"])].tail(10).to_dict('records')
    recent  = events.tail(15).to_dict('records')
    
    # 2. Get episodes for better grouping
    episodes = graph.nodes[graph.nodes["type"] == "episode"].tail(10).to_dict('records')

    payload = {
        "chronic_conditions": [_event_summary(e) for e in chronic],
        "active_episodes": [e.get("metadata") for e in episodes],
        "recent_history": [_event_summary(e) for e in recent],
    }

    try:
        response = client.chat.completions.create(
            model="gemma-4-e2b",
            messages=[
                {"role": "system", "content": CLINICAL_SUMMARY_PROMPT},
                {"role": "user",   "content": json.dumps(payload, indent=2)},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

def reason_about_query(graph, query, limit=20, now=None):
    # Step 1: Classify
    query_type = classify_query(query)

    # Step 2: Retrieve personal history (unless purely general)
    bundle = None
    if query_type in {"PERSONAL", "HYBRID"}:
        bundle = build_evidence_bundle(graph, query, limit=limit, now=now)
    else:
        # GENERAL: build a minimal bundle so the pipeline doesn't break
        bundle = {
            "query":               query,
            "query_time":          str(now) if now else "",
            "query_plan":          {"intent": "general", "entities": []},
            "events":              [],
            "background_facts":    [],
            "missing_information": [],
            "seed_nodes":          [],
        }

    # Step 3: Retrieve medical knowledge (for GENERAL and HYBRID)
    knowledge_hits = None
    if query_type in {"GENERAL", "HYBRID"}:
        knowledge_hits = query_knowledge_base(query, top_k=4)

    # Step 4: Generate
    answer = generate_answer(bundle, knowledge_hits=knowledge_hits)

    # Step 5: Collate metadata for caller
    candidate_triggers, symptoms, body_parts = [], [], []
    for event in bundle.get("events", []):
        for state in event.get("states", []):
            if state["relation"] == "TRIGGERED_BY" and state["name"] not in candidate_triggers:
                candidate_triggers.append(state["name"])
            elif state["relation"] == "HAS_SYMPTOM" and state["name"] not in symptoms:
                symptoms.append(state["name"])
            elif state["relation"] == "TARGETS" and state["name"] not in body_parts:
                body_parts.append(state["name"])

    return {
        "query":               query,
        "query_type":          query_type,
        "answer":              answer,
        "candidate_triggers":  candidate_triggers,
        "symptoms":            symptoms,
        "body_parts":          body_parts,
        "events":              bundle.get("events", []),
        "knowledge_hits":      knowledge_hits,
        "evidence_bundle":     bundle,
    }
