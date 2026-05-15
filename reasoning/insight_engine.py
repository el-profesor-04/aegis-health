"""
reasoning/insight_engine.py
───────────────────────────
The "Active" brain of Aegis. Handles:
1. Correlation Detection: Statistical links between triggers and symptoms.
2. Gap Detection: Identifying missing clinical details in new logs.
3. Cyclic Verification: Checking if scheduled activities are still happening.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from graph.schema import HealthGraph

# ── Correlation Settings ───────────────────────────────────────────────────
MIN_CORRELATION_STRENGTH = 0.4  
MIN_OCCURRENCES = 2             
LOOKBACK_WINDOW_HOURS = 48      

# ── Insight Engine ─────────────────────────────────────────────────────────

def detect_correlations(graph: HealthGraph) -> list[dict]:
    events = graph.get_event_nodes()
    if events.empty: return []

    episodes = graph.nodes[graph.nodes["type"] == "episode"]
    correlations = []
    
    for _, ep_row in episodes.iterrows():
        ep_id = ep_row["node_id"]
        ep_name = ep_row["name"].replace("_episode", "")
        
        event_edges = graph.edges[
            (graph.edges["source_id"] == ep_id) & 
            (graph.edges["relation"] == "HAS_EVENT")
        ]
        if event_edges.empty: continue
        
        e_ids = event_edges["target_id"].tolist()
        s_instances = events[events["node_id"].isin(e_ids)]
        
        trigger_counts = {}
        for _, s_row in s_instances.iterrows():
            # 1. Direct triggers in metadata
            md = s_row.get("metadata", {})
            direct_trigger = md.get("trigger")
            if direct_trigger:
                t_name = normalize_trigger_name(direct_trigger)
                if t_name:
                    trigger_counts[t_name] = trigger_counts.get(t_name, 0) + 1
                    continue
            
            # 2. Contextual triggers in window
            s_time = datetime.fromisoformat(s_row["event_time"])
            window_start = s_time - timedelta(hours=LOOKBACK_WINDOW_HOURS)
            
            # Find candidate triggers, EXCLUDING the symptom itself to avoid self-correlation
            potential_triggers = events[
                (events["event_type"].isin(["food", "activity", "medication", "sleep", "mood"])) &
                (events["event_time"] >= window_start.isoformat()) &
                (events["event_time"] < s_row["event_time"]) &
                (~events["name"].str.contains(ep_name, case=False, na=False))
            ]
            
            seen_in_this_window = set()
            for _, t_row in potential_triggers.iterrows():
                t_md = t_row.get("metadata", {})
                raw_t_name = t_md.get("trigger") or t_md.get("symptom") or t_row["name"].split(":")[-1]
                t_name = normalize_trigger_name(raw_t_name)
                if t_name and t_name not in seen_in_this_window:
                    trigger_counts[t_name] = trigger_counts.get(t_name, 0) + 1
                    seen_in_this_window.add(t_name)
                    
        total_s_count = len(s_instances)
        if total_s_count < MIN_OCCURRENCES: continue
        
        for t_name, count in trigger_counts.items():
            strength = count / total_s_count
            if strength >= MIN_CORRELATION_STRENGTH and count >= MIN_OCCURRENCES:
                correlations.append({
                    "type": "correlation",
                    "symptom": ep_name,
                    "trigger": t_name,
                    "strength": round(strength, 2),
                    "occurrences": count,
                    "message": f"I've noticed {int(strength*100)}% of your {ep_name} episodes happen after you log {t_name}."
                })
                
    return correlations

def normalize_trigger_name(name):
    if not name: return ""
    name = str(name).lower().strip()
    if any(k in name for k in ["cheese", "milk", "ice cream", "dairy", "yogurt", "pizza", "shake", "feta"]):
        return "dairy"
    if any(k in name for k in ["stress", "work", "deadline", "tantrum", "busy"]):
        return "stress"
    if any(k in name for k in ["sleep", "rough night", "crying", "stayed up", "restless"]):
        return "poor sleep"
    # Filter out common symptoms being used as triggers
    if name in ["bloating", "headache", "fatigue", "pain", "nausea", "throb", "stiffness"]:
        return ""
    return name

def detect_gaps(graph: HealthGraph) -> list[dict]:
    events = graph.get_event_nodes()
    if events.empty: return []
        
    latest_event = events.iloc[-1]
    e_id = latest_event["node_id"]
    
    edge = graph.edges[
        (graph.edges["target_id"] == e_id) & 
        (graph.edges["relation"] == "HAS_EVENT")
    ]
    if edge.empty: return []
        
    episode_id = edge.iloc[0]["source_id"]
    other_event_edges = graph.edges[
        (graph.edges["source_id"] == episode_id) &
        (graph.edges["relation"] == "HAS_EVENT") &
        (graph.edges["target_id"] != e_id)
    ]
    if other_event_edges.empty: return []
        
    other_event_ids = other_event_edges["target_id"].tolist()
    other_events = graph.nodes[graph.nodes["node_id"].isin(other_event_ids)]
    
    gaps = []
    fields_to_check = ["severity_band", "laterality"]
    current_md = latest_event["metadata"] or {}
    
    for field in fields_to_check:
        if current_md.get(field) is None:
            prev_vals = []
            for _, e in other_events.iterrows():
                md = e.get("metadata", {})
                if md and md.get(field):
                    prev_vals.append(md.get(field))
            
            if prev_vals:
                most_common = max(set(prev_vals), key=prev_vals.count)
                gaps.append({
                    "type": "gap",
                    "field": field,
                    "event_id": e_id,
                    "suggested_value": most_common,
                    "message": f"You logged a {latest_event['name'].split(':')[-1]}, but didn't mention {field}. Is it {most_common} like last time?"
                })
                
    return gaps

def check_cyclic_compliance(graph: HealthGraph, now=None) -> list[dict]:
    if isinstance(now, str):
        now = datetime.fromisoformat(now.replace(' ', '+'))
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None: now = now.replace(tzinfo=timezone.utc)
    
    cyclic_events = graph.get_cyclic_nodes()
    if cyclic_events.empty: return []
        
    notifications = []
    for name in cyclic_events["canonical_name"].unique():
        instances = cyclic_events[cyclic_events["canonical_name"] == name]
        latest_instance = instances.iloc[-1]
        last_time = datetime.fromisoformat(latest_instance["event_time"])
        period_hrs = latest_instance["cyclic_period_hrs"]
        if not period_hrs: continue
        
        expected_time = last_time + timedelta(hours=period_hrs)
        buffer_hrs = period_hrs * 0.2
        if now > (expected_time + timedelta(hours=buffer_hrs)):
            activity_name = name.split(":")[-1] if ":" in name else name
            notifications.append({
                "type": "cyclic_check",
                "activity": activity_name,
                "last_seen": latest_instance["event_time"],
                "message": f"I haven't heard about your {activity_name} lately. Did you go this week?"
            })
    return notifications

def get_all_insights(graph: HealthGraph, now=None) -> dict:
    return {
        "correlations": detect_correlations(graph),
        "gaps": detect_gaps(graph),
        "cyclic_checks": check_cyclic_compliance(graph, now=now)
    }
