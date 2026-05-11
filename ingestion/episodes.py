from utils.embedding import get_embedding, cosine_similarity
from datetime import datetime


EPISODE_SIM_THRESHOLD = 0.80
TIME_WINDOW_DAYS = 10

def safe_metadata(row):
    md = row.get("metadata", None)

    if isinstance(md, dict):
        return md

    return {}

def within_time_window(event_time, episode_time):
    if event_time is None or episode_time is None:
        return False

    event_time = datetime.fromisoformat(event_time)
    episode_time = datetime.fromisoformat(episode_time)

    return abs((event_time - episode_time).days) <= TIME_WINDOW_DAYS


def normalize_body_parts(body_parts):
    if body_parts is None:
        return []

    if isinstance(body_parts, str):
        body_parts = [body_parts]

    return sorted({part for part in body_parts if part})

def score_episode(graph, episode, symptom, body_parts, event_time):
    if not symptom:
        return -1

    score = 0.0

    md = safe_metadata(episode)

    ep_symptom = md.get("symptom")
    ep_bodies = normalize_body_parts(md.get("body_parts") or md.get("body_part"))
    ep_time = md.get("created_at")
    event_bodies = normalize_body_parts(body_parts)

    # 1. Symptom similarity (weighted)
    if ep_symptom and symptom:
        sim = cosine_similarity(
            get_embedding(ep_symptom),
            get_embedding(symptom)
        )
        score += sim * 0.5

    # 2. Body overlap (strong constraint)
    if ep_bodies and event_bodies:
        overlap = set(ep_bodies) & set(event_bodies)
        if overlap:
            score += 0.4
        elif any(a in b or b in a for a in ep_bodies for b in event_bodies):
            score += 0.2

    # 3. Time proximity (hard filter)
    if ep_time and event_time:
        if within_time_window(event_time, ep_time):
            score += 0.1
        else:
            return -1  # reject old episodes entirely

    return score

def find_best_episode(graph, symptom, body_parts, event_time):
    if not symptom:
        return None

    episodes = graph.nodes[graph.nodes["type"] == "episode"]

    best_ep = None
    best_score = 0

    for _, ep in episodes.iterrows():
        score = score_episode(
            graph,
            ep,
            symptom,
            body_parts,
            event_time
        )

        if score > best_score:
            best_score = score
            best_ep = ep

    if best_score < 0.6:
        return None

    return best_ep

def create_episode(graph, symptom, body_parts, event_id, event_time):
    if not symptom:
        return None

    body_parts = normalize_body_parts(body_parts)
    episode_id = graph.add_node(
        node_type="episode",
        name=f"{symptom}_episode",
        canonical_name=f"{symptom}_episode",
        embedding=get_embedding(symptom),
        metadata={
            "symptom": symptom,
            "body_part": body_parts[0] if body_parts else None,
            "body_parts": body_parts,
            "created_at": event_time,
            "last_event_at": event_time,
            "status": "active",
            "event_count": 1
        }
    )

    graph.add_edge(episode_id, event_id, "HAS_EVENT")

    return episode_id

def update_episode(graph, episode_id, event_id):
    graph.add_edge(episode_id, event_id, "HAS_EVENT")

    ep_row = graph.nodes[graph.nodes["node_id"] == episode_id]
    md = ep_row["metadata"].iloc[0]

    if not isinstance(md, dict):
        md = {}

    count = md.get("event_count", 0) + 1

    new_md = md.copy()
    new_md["event_count"] = count
    event_row = graph.nodes[graph.nodes["node_id"] == event_id]
    if not event_row.empty:
        event_md = event_row["metadata"].iloc[0]
        if isinstance(event_md, dict) and event_md.get("event_time"):
            new_md["last_event_at"] = event_md["event_time"]

    graph.nodes.loc[
        graph.nodes["node_id"] == episode_id,
        "metadata"
    ] = [new_md]

def attach_episode(graph, symptom, body_parts, event_id, event_time):
    if not symptom:
        return None

    ep = find_best_episode(graph, symptom, body_parts, event_time)

    if ep is None:
        return create_episode(graph, symptom, body_parts, event_id, event_time)

    update_episode(graph, ep["node_id"], event_id)
    return ep["node_id"]
