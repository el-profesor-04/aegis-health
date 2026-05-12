import json

from openai import OpenAI

from reasoning.retrieval import build_evidence_bundle


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    max_retries=0,
)


GENERATOR_PROMPT = """
You are Aegis, a personal health memory assistant.

Answer the user's question using ONLY the provided memory context.

Rules:
- Be helpful, concise, and grounded.
- Speak directly to the user as "you".
- Do not diagnose.
- State uncertainty clearly.
- Synthesize the strongest relevant facts; do not dump a raw event list unless the user asks for one.
- If causal language is requested, frame it as a possibility, not proof.
- Include missing information when it affects confidence.
- If evidence is weak or absent, say so directly.
- Do not mention internal fields such as scores, graph_score, relevance_score, S0, lambda, impact_class, or event ids.
- Do not say "evidence", "evidence bundle", "provided evidence", "records indicate", or "there is a memory of someone".
- Prefer phrases like "You logged...", "You mentioned...", "Your recent notes show...", and "This suggests..."

Return plain text, not JSON.
"""


def _event_summary(event):
    states = event.get("states", [])
    symptoms = [s["name"] for s in states if s["relation"] == "HAS_SYMPTOM"]
    body_parts = [s["name"] for s in states if s["relation"] == "TARGETS"]
    triggers = [s["name"] for s in states if s["relation"] == "TRIGGERED_BY"]

    return {
        "raw_text": event.get("raw_text"),
        "event_type": event.get("event_type"),
        "event_time": event.get("event_time"),
        "severity_band": event.get("severity_band"),
        "memory_type": _memory_type(event),
        "matched_query_entities": event.get("matched_entities", []),
        "symptoms": symptoms,
        "body_parts": body_parts,
        "triggers": triggers,
    }


def _memory_type(event):
    if event.get("is_cyclic"):
        return "recurring pattern"

    impact_class = event.get("impact_class")
    if impact_class == "C5":
        return "long-term health fact"
    if impact_class == "C4":
        return "persistent or recurring issue"
    if impact_class == "C3":
        return "acute issue"
    if impact_class == "C2":
        return "short-term pattern"
    return "recent transient event"


def _generator_payload(bundle):
    return {
        "query": bundle["query"],
        "query_time": bundle["query_time"],
        "query_plan": bundle["query_plan"],
        "top_events": [
            _event_summary(event)
            for event in bundle.get("events", [])[:12]
        ],
        "missing_information": bundle.get("missing_information", []),
    }


def generate_answer(bundle, model="gemma-4-e2b"):
    payload = _generator_payload(bundle)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GENERATOR_PROMPT},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return _fallback_answer(payload)


def _fallback_answer(payload):
    events = payload.get("top_events", [])
    if not events:
        return "I could not find matching events in the graph for this question."

    triggers = []
    symptoms = []
    for event in events:
        for trigger in event.get("triggers", []):
            if trigger not in triggers:
                triggers.append(trigger)
        for symptom in event.get("symptoms", []):
            if symptom not in symptoms:
                symptoms.append(symptom)

    parts = ["I found relevant graph history."]
    if triggers:
        parts.append("Possible related trigger(s): " + ", ".join(triggers) + ".")
    if symptoms:
        parts.append("Related symptom(s): " + ", ".join(symptoms) + ".")

    top = events[0]
    if top.get("raw_text"):
        parts.append(f"Strongest event: {top['raw_text']}")

    missing = payload.get("missing_information", [])
    if missing:
        parts.append("Limits: " + " ".join(missing))

    return " ".join(parts)


def reason_about_query(graph, query, limit=20, now=None):
    bundle = build_evidence_bundle(graph, query, limit=limit, now=now)
    answer = generate_answer(bundle)

    candidate_triggers = []
    symptoms = []
    body_parts = []

    for event in bundle.get("events", []):
        for state in event.get("states", []):
            if state["relation"] == "TRIGGERED_BY" and state["name"] not in candidate_triggers:
                candidate_triggers.append(state["name"])
            elif state["relation"] == "HAS_SYMPTOM" and state["name"] not in symptoms:
                symptoms.append(state["name"])
            elif state["relation"] == "TARGETS" and state["name"] not in body_parts:
                body_parts.append(state["name"])

    return {
        "query": query,
        "answer": answer,
        "candidate_triggers": candidate_triggers,
        "symptoms": symptoms,
        "body_parts": body_parts,
        "events": bundle.get("events", []),
        "evidence_bundle": bundle,
    }
