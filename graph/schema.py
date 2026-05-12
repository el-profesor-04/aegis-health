import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timezone


def generate_id():
    return str(uuid.uuid4())


def current_time():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Columns that are promoted out of the metadata dict into first-class columns.
# These are the fields the cyclic detector and retriever query / filter / sort
# on directly. Keeping them as real DataFrame columns avoids iterating over
# nested dicts at query time.
#
# Convention: if a column is not applicable for a node type (e.g. impact_class
# on a "state" node) it is stored as None / NaN.
# ---------------------------------------------------------------------------

EVENT_COLUMNS = [
    "event_type",        # symptom | activity | food | medication | sleep | mood | other
    "event_time",        # ISO timestamp of when the event occurred
    "impact_class",      # C1 – C5  (C6 if promoted by cyclic detector)
    "severity_band",     # Low | Moderate | High | None
    "S0",                # float 0.0–1.0  initial relevance anchor
    "lambda_hr",         # float  hourly decay constant
    "cyclic_candidate",  # bool   LLM flagged this as a habitual event
    "is_cyclic",         # bool   confirmed by cyclic_detector
    "cyclic_period_hrs", # float | None  period if is_cyclic is True
    "occurrence_count",  # int    incremented by cyclic_detector / episodes
]


class HealthGraph:
    def __init__(self):
        # ------------------------------------------------------------------
        # Nodes table
        # Core identity columns + all promoted event decay columns.
        # Non-event nodes (state, episode) will have None in the event cols.
        # ------------------------------------------------------------------
        node_columns = [
            "node_id",
            "type",              # state | event | episode
            "name",
            "canonical_name",
            "embedding",         # np.array or None
            "metadata",          # full raw dict (always kept for completeness)
            "created_at",
        ] + EVENT_COLUMNS

        self.nodes = pd.DataFrame(columns=node_columns)

        # ------------------------------------------------------------------
        # Edges table — unchanged from original
        # ------------------------------------------------------------------
        self.edges = pd.DataFrame(columns=[
            "edge_id",
            "source_id",
            "target_id",
            "relation",
            "timestamp",
            "metadata",
        ])

    # -----------------------------------------------------------------------
    # Helpers: extract promoted fields from metadata dict
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_event_columns(metadata: dict) -> dict:
        """
        Pull the known event-column values out of a metadata dict so they can
        be written to their dedicated DataFrame columns.
        Returns a dict with exactly the keys in EVENT_COLUMNS.
        Missing keys default to None (or sensible typed defaults).
        """
        if not isinstance(metadata, dict):
            return {k: None for k in EVENT_COLUMNS}

        return {
            "event_type":        metadata.get("event_type"),
            "event_time":        metadata.get("event_time"),
            "impact_class":      metadata.get("impact_class"),
            "severity_band":     metadata.get("severity_band"),
            "S0":                metadata.get("S0"),
            "lambda_hr":         metadata.get("lambda_hr"),
            "cyclic_candidate":  metadata.get("cyclic_candidate", False),
            "is_cyclic":         metadata.get("is_cyclic", False),
            "cyclic_period_hrs": metadata.get("cyclic_period_hrs"),
            "occurrence_count":  metadata.get("occurrence_count", 1),
        }

    # -----------------------------------------------------------------------
    # Core write operations
    # -----------------------------------------------------------------------

    def add_node(self, node_type, name, canonical_name=None,
                 embedding=None, metadata=None):
        node_id = generate_id()
        metadata = metadata or {}

        # Promoted columns: only populated for event nodes; None for others.
        event_cols = (
            self._extract_event_columns(metadata)
            if node_type == "event"
            else {k: None for k in EVENT_COLUMNS}
        )

        new_node = {
            "node_id":        node_id,
            "type":           node_type,
            "name":           name,
            "canonical_name": canonical_name or name,
            "embedding":      embedding,
            "metadata":       metadata,
            "created_at":     current_time(),
            **event_cols,
        }

        self.nodes = pd.concat(
            [self.nodes, pd.DataFrame([new_node])],
            ignore_index=True
        )
        return node_id

    def add_edge(self, source_id, target_id, relation, metadata=None):
        if source_id == target_id:
            return None

        existing = self.edges[
            (self.edges["source_id"] == source_id) &
            (self.edges["target_id"] == target_id) &
            (self.edges["relation"]  == relation)
        ]
        if not existing.empty:
            return existing.iloc[0]["edge_id"]

        edge_id = generate_id()
        new_edge = {
            "edge_id":   edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation":  relation,
            "timestamp": current_time(),
            "metadata":  metadata or {},
        }

        self.edges = pd.concat(
            [self.edges, pd.DataFrame([new_edge])],
            ignore_index=True
        )
        return edge_id

    # -----------------------------------------------------------------------
    # Update helpers
    # Called by cyclic_detector (Steps 4) and episode logic to mutate nodes
    # after creation without touching unrelated fields.
    # -----------------------------------------------------------------------

    def update_node_fields(self, node_id: str, **kwargs):
        """
        Update one or more first-class columns AND mirror the same values
        into the metadata dict so the two are always in sync.

        Usage:
            graph.update_node_fields(
                node_id,
                is_cyclic=True,
                cyclic_period_hrs=168.0,
                impact_class="C6",
            )
        """
        mask = self.nodes["node_id"] == node_id
        if not mask.any():
            return

        for field, value in kwargs.items():
            if field in self.nodes.columns:
                self.nodes.loc[mask, field] = value

        # Mirror into metadata dict
        row_idx = self.nodes.index[mask][0]
        md = self.nodes.at[row_idx, "metadata"]
        if not isinstance(md, dict):
            md = {}
        md.update(kwargs)
        self.nodes.at[row_idx, "metadata"] = md

    def increment_occurrence_count(self, node_id: str):
        """Atomically bump occurrence_count by 1 on the column and in metadata."""
        mask = self.nodes["node_id"] == node_id
        if not mask.any():
            return
        current = self.nodes.loc[mask, "occurrence_count"].iloc[0]
        current = int(current) if pd.notna(current) else 0
        self.update_node_fields(node_id, occurrence_count=current + 1)

    # -----------------------------------------------------------------------
    # Query helpers
    # All return DataFrames (possibly empty), never raise on missing data.
    # -----------------------------------------------------------------------

    def get_node(self, node_id: str) -> pd.DataFrame:
        return self.nodes[self.nodes["node_id"] == node_id]

    def find_nodes_by_name(self, name: str) -> pd.DataFrame:
        return self.nodes[self.nodes["canonical_name"] == name]

    def get_edges_from(self, node_id: str) -> pd.DataFrame:
        return self.edges[self.edges["source_id"] == node_id]

    def get_edges_to(self, node_id: str) -> pd.DataFrame:
        return self.edges[self.edges["target_id"] == node_id]

    def get_event_nodes(self) -> pd.DataFrame:
        """All event nodes, sorted oldest → newest by event_time."""
        events = self.nodes[self.nodes["type"] == "event"].copy()
        events = events.sort_values("event_time", ascending=True, na_position="last")
        return events

    def get_events_by_type(self, event_type: str) -> pd.DataFrame:
        """All event nodes of a specific event_type (sleep, food, symptom…)."""
        return self.nodes[
            (self.nodes["type"] == "event") &
            (self.nodes["event_type"] == event_type)
        ]

    def get_cyclic_candidates(self) -> pd.DataFrame:
        """
        Event nodes where cyclic_candidate=True but is_cyclic is still False.
        These are the nodes cyclic_detector needs to evaluate.
        """
        return self.nodes[
            (self.nodes["type"] == "event") &
            (self.nodes["cyclic_candidate"] == True) &   # noqa: E712
            (self.nodes["is_cyclic"] == False)            # noqa: E712
        ]

    def get_cyclic_nodes(self) -> pd.DataFrame:
        """Event nodes already confirmed as cyclic (is_cyclic=True)."""
        return self.nodes[
            (self.nodes["type"] == "event") &
            (self.nodes["is_cyclic"] == True)             # noqa: E712
        ]

    def get_events_by_canonical_name(self, canonical_name: str) -> pd.DataFrame:
        """
        All event nodes sharing a canonical_name.
        Useful for the cyclic detector to group by entity:
            e.g. all "activity:tennis" events.
        """
        return self.nodes[
            (self.nodes["type"] == "event") &
            (self.nodes["canonical_name"] == canonical_name)
        ]

    # -----------------------------------------------------------------------
    # Debug / inspection
    # -----------------------------------------------------------------------

    def print_graph(self):
        print("\n=== NODES ===")
        cols = ["node_id", "type", "canonical_name", "impact_class",
                "severity_band", "S0", "is_cyclic", "occurrence_count"]
        available = [c for c in cols if c in self.nodes.columns]
        print(self.nodes[available].to_string(index=False))

        print("\n=== EDGES ===")
        print(self.edges[["source_id", "relation", "target_id"]].to_string(index=False))

    def summary(self):
        """Quick stats — useful after a batch ingest."""
        events = self.get_event_nodes()
        print(f"\n{'─'*40}")
        print(f"  Nodes total    : {len(self.nodes)}")
        print(f"  Event nodes    : {len(events)}")
        print(f"  Edges total    : {len(self.edges)}")
        if not events.empty and "impact_class" in events.columns:
            print(f"\n  Impact class breakdown:")
            for cls, count in events["impact_class"].value_counts().items():
                print(f"    {cls}: {count}")
            print(f"\n  Cyclic candidates : {events['cyclic_candidate'].sum()}")
            print(f"  Confirmed cyclic  : {events['is_cyclic'].sum()}")
        print(f"{'─'*40}\n")