from utils.embedding import get_embedding, cosine_similarity

# Threshold from the LLM-standardized phrase embedding experiment.
SIM_THRESHOLD = 0.75
TOP_K = 5

LATERALITY_VALUES = {"left", "right", "bilateral"}


def normalize_concept(name):
    if name is None:
        return None

    name = str(name).lower().strip()
    if name in {"", "none", "null", "unknown", "n/a"}:
        return None

    return name

def find_candidates(embedding, nodes_df, node_type):
    candidates = nodes_df[nodes_df["type"] == node_type]

    scored = []
    for _, row in candidates.iterrows():
        sim = cosine_similarity(embedding, row["embedding"])
        scored.append((sim, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[:TOP_K]

def resolve_or_create(graph, name, node_type):
    name = normalize_concept(name)

    if name is None:
        return None

    exact = graph.nodes[
        (graph.nodes["type"] == node_type) &
        (graph.nodes["canonical_name"] == name)
    ]
    if not exact.empty:
        return exact.iloc[0]["node_id"]

    embedding = get_embedding(name)

    candidates = find_candidates(embedding, graph.nodes, node_type)

    for sim, row in candidates:
        if sim >= SIM_THRESHOLD:
            return row["node_id"]

    # Create new node
    return graph.add_node(
        node_type=node_type,
        name=name,
        canonical_name=name,
        embedding=embedding
    )

def normalize_body_part(part):
    if part is None:
        return None

    return normalize_concept(part)

def resolve_body_nodes(graph, body_part, laterality):
    if body_part is None:
        return None, None

    base_name = normalize_body_part(body_part)
    base_id = resolve_or_create(graph, base_name, "state")

    laterality = normalize_concept(laterality)

    if laterality in LATERALITY_VALUES:
        specific_name = f"{laterality} {base_name}"
        specific_id = resolve_or_create(graph, specific_name, "state")

        return specific_id, base_id

    return base_id, None


def resolve_body_node_pairs(graph, body_parts, laterality):
    if body_parts is None:
        return []

    if isinstance(body_parts, str):
        body_parts = [body_parts]

    pairs = []
    seen = set()
    for body_part in body_parts:
        base_name = normalize_body_part(body_part)
        if base_name is None or base_name in seen:
            continue

        seen.add(base_name)
        specific_id, base_id = resolve_body_nodes(graph, base_name, laterality)
        pairs.append({
            "specific_id": specific_id,
            "base_id": base_id,
            "base_name": base_name,
        })

    return pairs
