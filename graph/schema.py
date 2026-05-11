import pandas as pd
import numpy as np
import uuid
from datetime import datetime


def generate_id():
    return str(uuid.uuid4())


def current_time():
    return datetime.utcnow().isoformat()


class HealthGraph:
    def __init__(self):
        # Nodes table
        self.nodes = pd.DataFrame(columns=[
            "node_id",
            "type",              # state / event / episode
            "name",              # raw or display name
            "canonical_name",    # normalized name
            "embedding",         # vector (np.array or list)
            "metadata",          # dict
            "created_at"
        ])

        # Edges table
        self.edges = pd.DataFrame(columns=[
            "edge_id",
            "source_id",
            "target_id",
            "relation",          # e.g., TARGETS, TRIGGERED_BY
            "timestamp",
            "metadata"
        ])
    
    def add_node(self, node_type, name, canonical_name=None, embedding=None, metadata=None):
        node_id = generate_id()

        new_node = {
            "node_id": node_id,
            "type": node_type,
            "name": name,
            "canonical_name": canonical_name or name,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": current_time()
        }

        self.nodes = pd.concat([self.nodes, pd.DataFrame([new_node])], ignore_index=True)

        return node_id
    
    def add_edge(self, source_id, target_id, relation, metadata=None):
        if source_id == target_id:
            return None
        # prevent duplicate edges
        existing = self.edges[
            (self.edges["source_id"] == source_id) &
            (self.edges["target_id"] == target_id) &
            (self.edges["relation"] == relation)
        ]

        if not existing.empty:
            return existing.iloc[0]["edge_id"]

        edge_id = generate_id()

        new_edge = {
            "edge_id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation,
            "timestamp": current_time(),
            "metadata": metadata or {}
        }

        self.edges = pd.concat([self.edges, pd.DataFrame([new_edge])], ignore_index=True)

        return edge_id
    
    def get_node(self, node_id):
        return self.nodes[self.nodes["node_id"] == node_id]

    def find_nodes_by_name(self, name):
        return self.nodes[self.nodes["canonical_name"] == name]

    def get_edges_from(self, node_id):
        return self.edges[self.edges["source_id"] == node_id]

    def get_edges_to(self, node_id):
        return self.edges[self.edges["target_id"] == node_id]
    
    def print_graph(self):
        print("\n=== NODES ===")
        print(self.nodes[["node_id", "type", "canonical_name"]])

        print("\n=== EDGES ===")
        print(self.edges[["source_id", "relation", "target_id"]])