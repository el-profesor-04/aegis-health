"""
utils/relevance.py
──────────────────
Unified relevance scoring for health graph event nodes.

This is the single function the retriever will call to rank candidates.
It implements the full formula designed in the architecture:

  Standard (C1–C5):
      R(t) = S0 * B * exp(-lambda_hr * t_hours)

  Cyclic (C6):
      t_prox = min(t_since % period, period - (t_since % period))
      R(t)   = S0 * exp(-lambda_hr * t_prox)

  Chronic (C5):
      R = S0  (no decay, lambda_hr ≈ 0 handles this automatically,
               but we short-circuit for clarity)

The accumulation boost B rewards patterns:
    B = min(1 + (occurrence_count - 1) * BOOST_PER_OCCURRENCE, MAX_BOOST)

Import and call score_node_relevance(node_row, query_time).
"""

import math
from datetime import datetime, timezone


# ─── Accumulation boost constants ─────────────────────────────────────────

# How much each additional occurrence adds to B (multiplicative boost).
# 1 occurrence → B = 1.0   (no boost)
# 2 occurrences → B = 1.25
# 3 occurrences → B = 1.50
# 5 occurrences → B = 2.00 (capped)
BOOST_PER_OCCURRENCE = 0.25
MAX_BOOST            = 2.0


# ─── Helpers ──────────────────────────────────────────────────────────────

def _parse_time(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _hours_since(event_time: datetime, query_time: datetime) -> float:
    delta = query_time - event_time
    return max(delta.total_seconds() / 3600.0, 0.0)


def _accumulation_boost(occurrence_count) -> float:
    n = int(occurrence_count) if occurrence_count else 1
    n = max(n, 1)
    return min(1.0 + (n - 1) * BOOST_PER_OCCURRENCE, MAX_BOOST)


# ─── Main scorer ──────────────────────────────────────────────────────────

def score_node_relevance(node_row, query_time: datetime = None) -> float:
    """
    Compute the current relevance of an event node.

    Parameters
    ----------
    node_row : dict or pandas Series
        A single event node row from HealthGraph.nodes.
        Must contain: impact_class, S0, lambda_hr, event_time,
                      occurrence_count, is_cyclic, cyclic_period_hrs.

    query_time : datetime, optional
        The moment of the query. Defaults to now (UTC).
        Injecting this makes the function pure and unit-testable.

    Returns
    -------
    float in [0.0, ~2.0]
        Higher = more relevant right now.
        C5 (Chronic) nodes return S0 (their fixed relevance floor).
        Returns 0.0 if event_time is missing or unparseable.
    """
    if query_time is None:
        query_time = datetime.now(timezone.utc)
    elif query_time.tzinfo is None:
        query_time = query_time.replace(tzinfo=timezone.utc)

    # Pull fields — works for both dict and pandas Series
    impact_class     = node_row.get("impact_class") or "C1"
    S0               = float(node_row.get("S0") or 0.5)
    lambda_hr        = float(node_row.get("lambda_hr") or 0.0)
    is_cyclic        = bool(node_row.get("is_cyclic", False))
    cyclic_period    = node_row.get("cyclic_period_hrs")
    occurrence_count = node_row.get("occurrence_count", 1)

    event_time = _parse_time(node_row.get("event_time"))
    if event_time is None:
        return 0.0

    # ── C5 Chronic: permanent, no decay ────────────────────────────────
    if impact_class == "C5":
        return S0   # always relevant, no boost (it's a baseline fact)

    # ── C6 Cyclic: proximity-to-cycle relevance ─────────────────────────
    if is_cyclic and cyclic_period and float(cyclic_period) > 0:
        period = float(cyclic_period)
        t_since           = _hours_since(event_time, query_time)
        t_since_in_cycle  = t_since % period
        t_until           = period - t_since_in_cycle
        t_prox            = min(t_since_in_cycle, t_until)
        return S0 * math.exp(-lambda_hr * t_prox)

    # ── C1–C4 Standard: exponential decay with accumulation boost ───────
    t_hours = _hours_since(event_time, query_time)
    B       = _accumulation_boost(occurrence_count)
    return S0 * B * math.exp(-lambda_hr * t_hours)


def score_nodes(nodes_df, query_time: datetime = None) -> "pd.Series":
    """
    Vectorised wrapper: score all rows in a DataFrame and return a Series
    of relevance floats, indexed the same as nodes_df.

    Usage:
        event_nodes = graph.get_event_nodes()
        event_nodes["relevance"] = score_nodes(event_nodes)
        top = event_nodes.nlargest(10, "relevance")
    """
    if query_time is None:
        query_time = datetime.now(timezone.utc)

    return nodes_df.apply(
        lambda row: score_node_relevance(row.to_dict(), query_time),
        axis=1
    )


# ─── Debug / inspection helper ────────────────────────────────────────────

def explain_relevance(node_row, query_time: datetime = None) -> str:
    """
    Human-readable breakdown of the relevance score for a single node.
    Useful during development to verify the formula is behaving as expected.

    Example output:
        [activity:tennis] C6 cyclic | S0=0.60 | period=168.0h
        t_since=52.3h → t_prox=52.3h | R = 0.60 * exp(-0.008 * 52.3) = 0.385
    """
    if query_time is None:
        query_time = datetime.now(timezone.utc)

    name         = node_row.get("canonical_name", "?")
    impact_class = node_row.get("impact_class", "?")
    S0           = float(node_row.get("S0") or 0.5)
    lambda_hr    = float(node_row.get("lambda_hr") or 0.0)
    is_cyclic    = bool(node_row.get("is_cyclic", False))
    cyclic_p     = node_row.get("cyclic_period_hrs")
    n            = node_row.get("occurrence_count", 1)
    event_time   = _parse_time(node_row.get("event_time"))

    score = score_node_relevance(node_row, query_time)

    if impact_class == "C5":
        return (
            f"[{name}] C5 chronic | S0={S0} | R = {score:.4f} (no decay)"
        )

    if is_cyclic and cyclic_p:
        period = float(cyclic_p)
        t_since = _hours_since(event_time, query_time) if event_time else 0
        t_prox  = min(t_since % period, period - (t_since % period))
        return (
            f"[{name}] C6 cyclic | S0={S0} | period={period}h\n"
            f"  t_since={t_since:.1f}h → t_prox={t_prox:.1f}h\n"
            f"  R = {S0} * exp(-{lambda_hr} * {t_prox:.1f}) = {score:.4f}"
        )

    t_hours = _hours_since(event_time, query_time) if event_time else 0
    B       = _accumulation_boost(n)
    return (
        f"[{name}] {impact_class} | S0={S0} | λ={lambda_hr} | n={n} → B={B:.2f}\n"
        f"  t={t_hours:.1f}h\n"
        f"  R = {S0} * {B:.2f} * exp(-{lambda_hr} * {t_hours:.1f}) = {score:.4f}"
    )