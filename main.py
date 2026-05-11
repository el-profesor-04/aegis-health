from graph.schema import HealthGraph
from ingestion.pipeline import ingest_text
from visualization.graph_viz import visualize_graph

graph = HealthGraph()

inputs = [
    "Had way too much coffee trying to finish work, and now even though I'm exhausted I feel jittery and my heart's kind of racing."
]

for text in inputs:
    ingest_text(graph, text)

graph.print_graph()
visualize_graph(graph, "user_graph.html")
