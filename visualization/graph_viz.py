from pyvis.network import Network


def visualize_graph(graph, output_file="graph.html"):
    net = Network(height="800px", width="100%", bgcolor="#111111", font_color="white")

    net.force_atlas_2based()

    # --- Nodes ---
    for _, row in graph.nodes.iterrows():
        node_id = row["node_id"]
        label = row["canonical_name"]
        node_type = row["type"]

        color = {
            "event": "#ff6b6b",
            "state": "#4dabf7",
            "episode": "#51cf66"
        }.get(node_type, "#ffffff")

        net.add_node(
            node_id,
            label=label,
            title=str(row.get("metadata", {})),
            color=color
        )

    # --- Edges ---
    for _, row in graph.edges.iterrows():
        net.add_edge(
            row["source_id"],
            row["target_id"],
            label=row["relation"]
        )

    # ✅ FIX HERE
    net.write_html(output_file)

    print(f"Saved graph to {output_file}")