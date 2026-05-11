from reasoning.retrieval import retrieve_relevant_events


def _group_state_values(event, relation):
    return [
        state["name"]
        for state in event.get("states", [])
        if state["relation"] == relation
    ]


def reason_about_query(graph, query, limit=5):
    relevant_events = retrieve_relevant_events(graph, query, limit=limit)

    candidate_triggers = []
    symptoms = []
    body_parts = []

    for event in relevant_events:
        for trigger in _group_state_values(event, "TRIGGERED_BY"):
            if trigger not in candidate_triggers:
                candidate_triggers.append(trigger)

        for symptom in _group_state_values(event, "HAS_SYMPTOM"):
            if symptom not in symptoms:
                symptoms.append(symptom)

        for body_part in _group_state_values(event, "TARGETS"):
            if body_part not in body_parts:
                body_parts.append(body_part)

    if candidate_triggers:
        answer = "Relevant history points to possible trigger(s): " + ", ".join(candidate_triggers)
    elif relevant_events:
        answer = "Relevant history found, but no explicit trigger is linked yet."
    else:
        answer = "No relevant history found in the graph."

    return {
        "query": query,
        "answer": answer,
        "candidate_triggers": candidate_triggers,
        "symptoms": symptoms,
        "body_parts": body_parts,
        "events": relevant_events,
    }
