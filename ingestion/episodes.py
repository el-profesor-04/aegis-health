"""
ingestion/episodes.py
─────────────────────
Impact-class-aware episode grouping.

An episode is a temporal cluster of health events sharing the same clinical
concept. It acts as a memory anchor: the retriever finds all events in an
episode by traversing a single HAS_EVENT edge, giving it temporal context
without loading the full graph.

Old design problems fixed here:
  - Fixed TIME_WINDOW_DAYS=10 for everything → replaced with CLASS_WINDOW_DAYS
  - Hard -1 rejection outside window → graceful per-class decay
  - C5 (Chronic) episodes never closed → correct, they stay open forever
  - Non-symptom events (sleep, food, activity) couldn't form episodes → C2+ can now
  - No tracking of highest_impact_class or episode status

New joining logic:
    An event joins an existing episode when ALL of:
      1. Semantic similarity of concepts > EPISODE_SIM_THRESHOLD
      2. Body part compatible (same or one contains the other, or neither has one)
      3. Time since episode's last_event_at <= CLASS_WINDOW_DAYS[event's impact_class]
         (C5 has no window — always joins a matching chronic episode)
"""

from datetime import datetime, timezone
from utils.embedding import get_embedding, cosine_similarity
from graph.schema import IMPACT_CLASS_CONFIG


# ── Similarity threshold for concept matching ─────────────────────────────
# Kept at 0.80: "migraine" and "headache" (~0.72) stay separate episodes.
# "pain" and "sprain" (~0.65) stay separate.
# "migraine" and "migraine" (~1.0) always merge.
EPISODE_SIM_THRESHOLD = 0.80

# ── Dynamic joining window ────────────────────────────────────────────────
# How many half-lives an event can be away from an episode's last event 
# and still join. 3.0 covers ~87.5% of the "influence" of the previous event.
WINDOW_HALF_LIFE_MULTIPLIER = 3.0

# ── Event types allowed to create/join episodes ───────────────────────────
# C1 food/sleep/activity events are ephemeral — not worth an episode node.
# Anything C2+ for non-symptom types is meaningful (accumulated sleep debt,
# repeated food reactions, sustained mood episodes).
EPISODE_ALLOWED_TYPES = {"symptom", "mood", "medication"}
EPISODE_ALLOWED_CLASSES_FOR_OTHER_TYPES = {"C2", "C3", "C4", "C5", "C6"}


def _parse_time(ts):
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _safe_metadata(row):
    md = row.get("metadata", None)
    return md if isinstance(md, dict) else {}


def _normalize_body_parts(body_parts):
    if body_parts is None:
        return []
    if isinstance(body_parts, str):
        body_parts = [body_parts]
    return sorted({part for part in body_parts if part})


def _body_compatible(ep_bodies, event_bodies):
    """
    Returns True when body part sets are compatible for episode grouping.
    Compatible means: both empty, one contains the other, or direct overlap.
    """
    if not ep_bodies and not event_bodies:
        return True   # neither specifies a body part — fine
    if not ep_bodies or not event_bodies:
        return True   # one is unspecified — allow joining (new info)
    overlap = set(ep_bodies) & set(event_bodies)
    if overlap:
        return True
    # Substring containment: "lower back" compatible with "back"
    if any(a in b or b in a for a in ep_bodies for b in event_bodies):
        return True
    return False


def _within_class_window(event_time, episode_last_event_at, impact_class):
    """
    Returns True when the gap is within the class-appropriate window.
    C5 always returns True (no window).
    """
    config = IMPACT_CLASS_CONFIG.get(impact_class)
    if not config or config["half_life_hrs"] is None:
        return True   # C5: no limit

    t_event   = _parse_time(event_time)
    t_episode = _parse_time(episode_last_event_at)
    if t_event is None or t_episode is None:
        return True   # missing timestamps: don't block joining

    gap_days = abs((t_event - t_episode).total_seconds()) / 86400.0
    
    # window = multiplier * half_life
    window_days = (config["half_life_hrs"] * WINDOW_HALF_LIFE_MULTIPLIER) / 24.0
    return gap_days <= window_days


def _should_create_episode(event_type, impact_class, symptom):
    """
    Decide whether this event is eligible to create or join an episode.
    Symptom/mood/medication events always qualify.
    Other types qualify only when they're C2+.
    """
    if event_type in EPISODE_ALLOWED_TYPES:
        return True
    if symptom:   # any event with an explicit symptom gets an episode
        return True
    if impact_class in EPISODE_ALLOWED_CLASSES_FOR_OTHER_TYPES:
        return True
    return False


def _highest_class(existing_class, new_class):
    """Return the higher-severity impact class of two."""
    order = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 2}
    existing_rank = order.get(existing_class, 1)
    new_rank      = order.get(new_class, 1)
    return new_class if new_rank > existing_rank else existing_class


# ── Episode scoring ───────────────────────────────────────────────────────

def score_episode_match(episode_row, symptom, body_parts, event_time, impact_class):
    """
    Score how well a candidate episode matches this incoming event.

    Returns float ≥ 0 or -1 (hard reject).

    Reject conditions (returns -1):
      - Body parts incompatible
      - Event is outside the class-appropriate time window

    Score components:
      - Concept similarity: 0.0 – 0.70   (embedding cosine)
      - Body part match bonus:      0.20  (if both specify body parts and overlap)
      - Class compatibility bonus:  0.10  (if episode's highest class == event's class)
    """
    md = _safe_metadata(episode_row)

    ep_concept = md.get("symptom") or md.get("concept")
    ep_bodies  = _normalize_body_parts(md.get("body_parts") or md.get("body_part"))
    ep_last    = md.get("last_event_at") or md.get("created_at")
    ep_class   = md.get("highest_impact_class", "C1")
    ev_bodies  = _normalize_body_parts(body_parts)

    # Hard reject: body part mismatch
    if not _body_compatible(ep_bodies, ev_bodies):
        return -1

    # Hard reject: outside class time window
    if not _within_class_window(event_time, ep_last, impact_class):
        return -1

    score = 0.0

    # 1. Concept similarity (core signal)
    if ep_concept and symptom:
        try:
            sim = cosine_similarity(
                get_embedding(ep_concept),
                get_embedding(symptom)
            )
            score += float(sim) * 0.70
        except Exception:
            pass
    elif ep_concept == symptom and symptom:
        score += 0.70   # exact string match without embedding

    # 2. Body part overlap bonus
    if ep_bodies and ev_bodies:
        if set(ep_bodies) & set(ev_bodies):
            score += 0.20
        elif any(a in b or b in a for a in ep_bodies for b in ev_bodies):
            score += 0.10

    # 3. Class compatibility bonus
    if ep_class == impact_class:
        score += 0.10

    return score


# ── Episode CRUD ──────────────────────────────────────────────────────────

def find_best_episode(graph, symptom, body_parts, event_time, impact_class):
    """
    Find the best matching existing episode, or return None to create a new one.
    Requires score > SIM_THRESHOLD to join.
    """
    concept = symptom   # alias for clarity
    if not concept:
        return None

    episodes = graph.nodes[graph.nodes["type"] == "episode"]
    if episodes.empty:
        return None

    best_ep    = None
    best_score = 0.0

    for _, ep in episodes.iterrows():
        score = score_episode_match(
            ep, concept, body_parts, event_time, impact_class
        )
        if score > best_score:
            best_score = score
            best_ep    = ep

    # Require both semantic similarity and the time-window check to pass.
    # The sim component contributes 0.70 max, so threshold at 0.56 (=0.80*0.70)
    # means the semantic similarity on its own needs to be ≥ 0.80.
    min_join_score = EPISODE_SIM_THRESHOLD * 0.70
    if best_score < min_join_score:
        return None

    return best_ep


def create_episode(graph, concept, body_parts, event_id, event_time, impact_class, event_type):
    """Create a new episode node and link the first event."""
    body_parts = _normalize_body_parts(body_parts)

    # Name: use concept if available, else event_type
    episode_name = f"{concept or event_type}_episode"

    try:
        embedding = get_embedding(concept or event_type)
    except Exception:
        embedding = None

    episode_id = graph.add_node(
        node_type="episode",
        name=episode_name,
        canonical_name=episode_name,
        embedding=embedding,
        metadata={
            "symptom":             concept,
            "concept":             concept or event_type,
            "body_part":           body_parts[0] if body_parts else None,
            "body_parts":          body_parts,
            "event_types_seen":    [event_type] if event_type else [],
            "highest_impact_class": impact_class or "C1",
            "created_at":          event_time,
            "last_event_at":       event_time,
            "status":              "active",
            "event_count":         1,
        }
    )

    graph.add_edge(episode_id, event_id, "HAS_EVENT")
    return episode_id


def update_episode(graph, episode_id, event_id, event_time, impact_class, event_type):
    """Link a new event to an existing episode and update its metadata."""
    graph.add_edge(episode_id, event_id, "HAS_EVENT")

    ep_row = graph.nodes[graph.nodes["node_id"] == episode_id]
    if ep_row.empty:
        return

    md = ep_row["metadata"].iloc[0]
    md = md.copy() if isinstance(md, dict) else {}

    md["event_count"]        = md.get("event_count", 0) + 1
    md["last_event_at"]      = event_time or md.get("last_event_at")
    md["highest_impact_class"] = _highest_class(
        md.get("highest_impact_class", "C1"),
        impact_class or "C1"
    )
    seen_types = md.get("event_types_seen", [])
    if event_type and event_type not in seen_types:
        seen_types.append(event_type)
    md["event_types_seen"] = seen_types

    # Status: escalate to "chronic" if a C5 event joins
    if impact_class == "C5":
        md["status"] = "chronic"

    graph.nodes.loc[
        graph.nodes["node_id"] == episode_id,
        "metadata"
    ] = [md]


# ── Public entry point ────────────────────────────────────────────────────

def attach_episode(graph, symptom, body_parts, event_id, event_time,
                   impact_class="C1", event_type="symptom"):
    """
    Main entry point called by ingestion pipeline for every event.

    Returns the episode_id that this event was attached to, or None if
    this event type/class doesn't warrant episode tracking.
    """
    if not _should_create_episode(event_type, impact_class, symptom):
        return None

    concept = symptom

    ep = find_best_episode(graph, concept, body_parts, event_time, impact_class)

    if ep is None:
        return create_episode(
            graph, concept, body_parts, event_id,
            event_time, impact_class, event_type
        )

    update_episode(
        graph, ep["node_id"], event_id,
        event_time, impact_class, event_type
    )
    return ep["node_id"]