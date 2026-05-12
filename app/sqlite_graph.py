import json
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from graph.schema import EVENT_COLUMNS, HealthGraph


NODE_COLUMNS = [
    "node_id",
    "type",
    "name",
    "canonical_name",
    "embedding",
    "metadata",
    "created_at",
] + EVENT_COLUMNS

EDGE_COLUMNS = [
    "edge_id",
    "source_id",
    "target_id",
    "relation",
    "timestamp",
    "metadata",
]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_json(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return json.dumps(value, default=_json_default)


def _from_json(value, default):
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _embedding_from_json(value):
    data = _from_json(value, None)
    if data is None:
        return None
    return np.array(data)


def _clean_scalar(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


class SQLiteHealthGraph(HealthGraph):
    """
    SQLite-backed HealthGraph with the same public interface as HealthGraph.

    Existing ingestion/retrieval code can keep using graph.nodes and
    graph.edges as DataFrames. Writes are mirrored to SQLite. If legacy code
    mutates the DataFrames directly, call persist_all() after the operation.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        super().__init__()
        self._init_db()
        self._load()

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT,
                    canonical_name TEXT,
                    embedding_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT,
                    event_type TEXT,
                    event_time TEXT,
                    impact_class TEXT,
                    severity_band TEXT,
                    S0 REAL,
                    lambda_hr REAL,
                    cyclic_candidate INTEGER,
                    is_cyclic INTEGER,
                    cyclic_period_hrs REAL,
                    occurrence_count INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    timestamp TEXT,
                    metadata_json TEXT,
                    UNIQUE(source_id, target_id, relation)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_canonical ON nodes(canonical_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")

    def _load(self):
        with self._lock, self._connect() as conn:
            node_rows = conn.execute(
                """
                SELECT node_id, type, name, canonical_name, embedding_json,
                       metadata_json, created_at, event_type, event_time,
                       impact_class, severity_band, S0, lambda_hr,
                       cyclic_candidate, is_cyclic, cyclic_period_hrs,
                       occurrence_count
                FROM nodes
                """
            ).fetchall()
            edge_rows = conn.execute(
                """
                SELECT edge_id, source_id, target_id, relation, timestamp,
                       metadata_json
                FROM edges
                """
            ).fetchall()

        nodes = []
        for row in node_rows:
            record = dict(zip([
                "node_id", "type", "name", "canonical_name", "embedding_json",
                "metadata_json", "created_at", "event_type", "event_time",
                "impact_class", "severity_band", "S0", "lambda_hr",
                "cyclic_candidate", "is_cyclic", "cyclic_period_hrs",
                "occurrence_count",
            ], row))
            nodes.append({
                "node_id": record["node_id"],
                "type": record["type"],
                "name": record["name"],
                "canonical_name": record["canonical_name"],
                "embedding": _embedding_from_json(record["embedding_json"]),
                "metadata": _from_json(record["metadata_json"], {}),
                "created_at": record["created_at"],
                "event_type": record["event_type"],
                "event_time": record["event_time"],
                "impact_class": record["impact_class"],
                "severity_band": record["severity_band"],
                "S0": record["S0"],
                "lambda_hr": record["lambda_hr"],
                "cyclic_candidate": bool(record["cyclic_candidate"]),
                "is_cyclic": bool(record["is_cyclic"]),
                "cyclic_period_hrs": record["cyclic_period_hrs"],
                "occurrence_count": record["occurrence_count"],
            })

        edges = []
        for row in edge_rows:
            record = dict(zip([
                "edge_id", "source_id", "target_id", "relation",
                "timestamp", "metadata_json",
            ], row))
            edges.append({
                "edge_id": record["edge_id"],
                "source_id": record["source_id"],
                "target_id": record["target_id"],
                "relation": record["relation"],
                "timestamp": record["timestamp"],
                "metadata": _from_json(record["metadata_json"], {}),
            })

        self.nodes = pd.DataFrame(nodes, columns=NODE_COLUMNS)
        self.edges = pd.DataFrame(edges, columns=EDGE_COLUMNS)

    def _persist_node_row(self, row):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO nodes (
                    node_id, type, name, canonical_name, embedding_json,
                    metadata_json, created_at, event_type, event_time,
                    impact_class, severity_band, S0, lambda_hr,
                    cyclic_candidate, is_cyclic, cyclic_period_hrs,
                    occurrence_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["node_id"],
                    row["type"],
                    row["name"],
                    row["canonical_name"],
                    _to_json(row["embedding"]),
                    _to_json(row["metadata"]),
                    row["created_at"],
                    _clean_scalar(row.get("event_type")),
                    _clean_scalar(row.get("event_time")),
                    _clean_scalar(row.get("impact_class")),
                    _clean_scalar(row.get("severity_band")),
                    _clean_scalar(row.get("S0")),
                    _clean_scalar(row.get("lambda_hr")),
                    int(bool(row.get("cyclic_candidate"))) if row.get("cyclic_candidate") is not None else 0,
                    int(bool(row.get("is_cyclic"))) if row.get("is_cyclic") is not None else 0,
                    _clean_scalar(row.get("cyclic_period_hrs")),
                    _clean_scalar(row.get("occurrence_count")),
                ),
            )

    def _persist_edge_row(self, row):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO edges (
                    edge_id, source_id, target_id, relation, timestamp,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["edge_id"],
                    row["source_id"],
                    row["target_id"],
                    row["relation"],
                    row["timestamp"],
                    _to_json(row["metadata"]),
                ),
            )

    def add_node(self, node_type, name, canonical_name=None, embedding=None, metadata=None):
        node_id = super().add_node(node_type, name, canonical_name, embedding, metadata)
        row = self.nodes[self.nodes["node_id"] == node_id].iloc[0].to_dict()
        self._persist_node_row(row)
        return node_id

    def add_edge(self, source_id, target_id, relation, metadata=None):
        edge_id = super().add_edge(source_id, target_id, relation, metadata)
        if edge_id is None:
            return None
        row = self.edges[self.edges["edge_id"] == edge_id].iloc[0].to_dict()
        self._persist_edge_row(row)
        return edge_id

    def update_node_fields(self, node_id: str, **kwargs):
        super().update_node_fields(node_id, **kwargs)
        row = self.nodes[self.nodes["node_id"] == node_id]
        if not row.empty:
            self._persist_node_row(row.iloc[0].to_dict())

    def persist_all(self):
        """Rewrite SQLite from the current DataFrames."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")

        for _, row in self.nodes.iterrows():
            self._persist_node_row(row.to_dict())
        for _, row in self.edges.iterrows():
            self._persist_edge_row(row.to_dict())

    def reload(self):
        self._load()

    def clear(self):
        """Delete all graph rows from SQLite and reset in-memory DataFrames."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")

        self.nodes = pd.DataFrame(columns=NODE_COLUMNS)
        self.edges = pd.DataFrame(columns=EDGE_COLUMNS)
