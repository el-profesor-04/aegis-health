"""
cyclic_detector.py
──────────────────
Post-ingestion pass that looks for recurring patterns in the health graph
and promotes confirmed cyclic event nodes to impact class C6.

Design:
  - Groups event nodes by canonical_name (e.g. "activity:tennis")
  - Needs ≥ MIN_OCCURRENCES events of the same name to even attempt detection
  - Uses the Coefficient of Variation (CV) of inter-event deltas to determine
    whether the spacing is consistent enough to call "cyclic"
  - CV < CV_THRESHOLD  → consistent → promote to C6 with mean period
  - CV ≥ CV_THRESHOLD  → irregular → leave as-is, clear cyclic_candidate flag
    so we don't re-evaluate on every ingest

Call run_cyclic_detection(graph) after every ingest_text() call.
It is intentionally cheap: only evaluates nodes still flagged as
cyclic_candidate=True and is_cyclic=False (the "pending" pool).
"""

import math
from datetime import datetime, timezone
from collections import defaultdict


# ─── Tuneable constants ────────────────────────────────────────────────────

# Minimum number of observed occurrences before we attempt detection.
# 2 events gives 1 delta — not enough. 3 gives 2 deltas — minimum viable.
MIN_OCCURRENCES = 3

# Coefficient of variation ceiling for "consistent enough to be cyclic".
# CV = std(deltas) / mean(deltas). Lower = more consistent.
# 0.30 means spacing can vary by up to ~30% and still count as regular.
# e.g. weekly tennis that sometimes slips to 6 or 8 days → CV ≈ 0.15 → passes.
# "Sometimes daily, sometimes every 4 days" → CV > 0.5 → fails.
CV_THRESHOLD = 0.30

# Decay constant for confirmed cyclic nodes (λ_c).
# Relevance peaks right after the event and again as the next one approaches,
# decaying in the middle of the cycle.
# 0.008 hr⁻¹  → half-life ≈ 87 hrs (good for sub-weekly cycles).
CYCLIC_LAMBDA_HR = 0.008

# C6 impact class label for promoted nodes.
CYCLIC_IMPACT_CLASS = "C6"


# ─── Time helpers ─────────────────────────────────────────────────────────

def _parse_time(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        dt = datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _hours_between(t1: datetime, t2: datetime) -> float:
    return abs((t2 - t1).total_seconds()) / 3600.0


# ─── Statistics ───────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _coefficient_of_variation(deltas: list[float]) -> float:
    """
    CV = std / mean.
    Returns float('inf') if mean is 0 (pathological case — all events
    at the same timestamp).
    """
    if not deltas:
        return float("inf")
    m = _mean(deltas)
    if m == 0:
        return float("inf")
    s = _std(deltas, m)
    return s / m


# ─── Core detection logic ─────────────────────────────────────────────────

def _analyse_group(event_rows) -> dict | None:
    """
    Given a list of event node rows (all sharing a canonical_name),
    determine whether they form a consistent cycle.

    Returns a result dict if cyclic, None otherwise:
    {
        "is_cyclic":         True,
        "cyclic_period_hrs": float,   # mean inter-event gap in hours
        "cv":                float,   # coefficient of variation
        "n":                 int,     # number of occurrences used
    }
    """
    # Parse and sort timestamps
    times = []
    for row in event_rows:
        t = _parse_time(row.get("event_time"))
        if t is not None:
            times.append(t)

    times.sort()

    if len(times) < MIN_OCCURRENCES:
        return None

    # Inter-event deltas in hours
    deltas = [
        _hours_between(times[i], times[i + 1])
        for i in range(len(times) - 1)
    ]

    # Reject pathological cases (all on the same day → delta ≈ 0)
    if all(d < 1.0 for d in deltas):
        return None

    cv = _coefficient_of_variation(deltas)
    mean_period = _mean(deltas)

    if cv < CV_THRESHOLD:
        return {
            "is_cyclic":         True,
            "cyclic_period_hrs": round(mean_period, 2),
            "cv":                round(cv, 4),
            "n":                 len(times),
        }

    return None


# ─── Graph promotion / demotion ───────────────────────────────────────────

def _promote_to_cyclic(graph, node_ids: list[str], period_hrs: float):
    """
    Flip all nodes in node_ids to C6 and write cyclic period.
    Also updates occurrence_count to the total number of confirmed instances.
    """
    n = len(node_ids)
    for node_id in node_ids:
        graph.update_node_fields(
            node_id,
            impact_class=CYCLIC_IMPACT_CLASS,
            is_cyclic=True,
            cyclic_period_hrs=period_hrs,
            lambda_hr=CYCLIC_LAMBDA_HR,
            occurrence_count=n,
        )


def _clear_cyclic_candidate(graph, node_ids: list[str]):
    """
    Pattern was irregular — unset cyclic_candidate so we stop re-evaluating
    these nodes on every future ingest (saves work, avoids noise).
    """
    for node_id in node_ids:
        graph.update_node_fields(node_id, cyclic_candidate=False)


# ─── Public API ───────────────────────────────────────────────────────────

def run_cyclic_detection(graph) -> list[dict]:
    """
    Main entry point. Call after every ingest_text().

    Evaluates all event nodes where cyclic_candidate=True and is_cyclic=False.
    Groups them by canonical_name, runs _analyse_group on each group, and
    either promotes or clears the candidate flag.

    Returns a list of promotion reports (one per promoted group):
    [
      {
        "canonical_name": "activity:tennis",
        "period_hrs":     168.0,
        "period_label":   "~7.0 days",
        "cv":             0.12,
        "n":              5,
        "node_ids":       ["uuid1", "uuid2", ...],
      },
      ...
    ]
    Returns [] if no promotions occurred (the common case).
    """
    candidates = graph.get_cyclic_candidates()

    if candidates.empty:
        return []

    # Group by canonical_name (e.g. "activity:tennis", "food:coffee")
    groups = defaultdict(list)
    for _, row in candidates.iterrows():
        groups[row["canonical_name"]].append(row.to_dict())

    promotions = []

    for canonical_name, rows in groups.items():
        result = _analyse_group(rows)

        node_ids = [r["node_id"] for r in rows]

        if result is not None:
            _promote_to_cyclic(graph, node_ids, result["cyclic_period_hrs"])

            period_days = result["cyclic_period_hrs"] / 24
            promotions.append({
                "canonical_name": canonical_name,
                "period_hrs":     result["cyclic_period_hrs"],
                "period_label":   f"~{period_days:.1f} days",
                "cv":             result["cv"],
                "n":              result["n"],
                "node_ids":       node_ids,
            })
            print(
                f"✅ Cyclic promotion: '{canonical_name}' → C6 "
                f"(period ≈ {period_days:.1f} days, CV={result['cv']}, n={result['n']})"
            )
        elif len(rows) >= MIN_OCCURRENCES:
            # Only clear when we have ENOUGH data and spacing is still
            # irregular. If n < MIN_OCCURRENCES, do nothing — leave
            # cyclic_candidate=True so future ingests of the same entity
            # are evaluated together with these nodes.
            _clear_cyclic_candidate(graph, node_ids)
            print(
                f"⏭  '{canonical_name}': {len(rows)} occurrences, "
                f"spacing too irregular — cyclic_candidate cleared"
            )
        # else: fewer than MIN_OCCURRENCES — stay quiet, wait for more data

    return promotions


def cyclic_relevance(node_row, query_time: datetime) -> float:
    """
    Relevance formula for a confirmed C6 (cyclic) node.

    Unlike the standard exponential decay from ingestion time, a cyclic event
    has two relevance peaks per cycle:
      - right after it last happened (t_since is small)
      - as the next occurrence approaches (t_until is small)

    We use the minimum of those two distances as the proximity metric,
    then apply exponential decay on that.

        t_since   = hours since last event_time
        t_until   = period - (t_since mod period)
        t_prox    = min(t_since mod period, t_until)
        R         = S0 * exp(-lambda_hr * t_prox)

    This means relevance is high right after "tennis Sunday" and climbs
    again by Friday as next Sunday approaches.
    """
    event_time = _parse_time(node_row.get("event_time"))
    if event_time is None:
        return 0.0

    period_hrs = node_row.get("cyclic_period_hrs")
    S0         = node_row.get("S0", 0.5)
    lambda_hr  = node_row.get("lambda_hr", CYCLIC_LAMBDA_HR)

    if not period_hrs or period_hrs <= 0:
        return 0.0

    t_since = _hours_between(event_time, query_time)
    t_since_in_cycle = t_since % period_hrs
    t_until          = period_hrs - t_since_in_cycle
    t_prox           = min(t_since_in_cycle, t_until)

    return S0 * math.exp(-lambda_hr * t_prox)