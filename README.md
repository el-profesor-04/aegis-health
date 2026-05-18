# Aegis

> **Frontend & Mobile App:** See [aegis-mobile](https://github.com/el-profesor-04/aegis-mobile) for the Flutter Android application with on-device model execution and UI.

Python backend, ML infrastructure, and temporal health graph database for the Aegis personal health reasoning system. Contains data processing pipelines, embedding generation, and the core SQLite-based temporal graph engine.

## Overview

Aegis Backend provides the foundational ML and data infrastructure for the sovereign, offline health agent. It includes:

- **Temporal Health Graph Database**: Custom semantic graph stored in SQLite with causal relationships and temporal decay
- **Gemma 4 E2B & MediaPipe BERT**: LiteRT models for entity extraction, query routing, and semantic embeddings
- **Entity Extraction & Embeddings**: Natural language processing for health event extraction and semantic retrieval
- **Graph Reasoning Engine**: Three-phase retrieval pipeline for intelligent pattern discovery
- **First Aid Knowledge Base**: 55 curated Mayo Clinic first aid articles embedded and searchable offline

All processing is designed to run locally on-device without cloud connectivity.

## Architecture

### Temporal Health Graph

The core of Aegis is a semantic health graph stored in SQLite. Every logged health event creates a node with:

**Impact Classification**:
- **C1 Transient**: Acute, one-time events (e.g., single dose of medication)
- **C2 Short-term**: Events lasting hours to days
- **C3 Medium-term**: Events lasting days to weeks
- **C4 Long-term**: Events lasting weeks to months
- **C5 Chronic**: Persistent conditions, indefinite lifespan
- **C6 Cyclic**: Recurring patterns with predictable periods (e.g., hormonal cycles, weekly stressors)

**Relevance Scoring**:
- Linear decay based on impact class for C1-C5
- Modulo-arithmetic temporal alignment for C6 (cyclic nodes re-surface in ranking when current time aligns with predicted period)
- Exponential decay function: `relevance(t) = initial_score × e^(-decay_rate × time_elapsed)`

**Graph Edges**:
- `TRIGGERED_BY`: Causal antecedent relationship
- `HAS_SYMPTOM`: Symptom manifestation
- `TARGETS`: Intervention or medication target
- `PART_OF`: Temporal grouping or event composition

### Gemma 4 E2B & LiteRT Integration

Aegis uses Gemma 4 E2B via LiteRT-LM for three core NLP tasks:

1. **Entity Extraction**: Parses user-logged health events into structured entities (symptom, medication, food, sleep hours, exercise duration, mood)
2. **Query Routing**: Classifies user queries to determine which reasoning pathway (ingestion, history search, first aid RAG)
3. **Answer Generation**: Generates natural language responses based on retrieved graph context

The Gemma 4 E2B model is sourced pre-converted in LiteRT-LM format directly from Hugging Face.

### Offline Embeddings & Retrieval

**Phase 1 - Seed Finding**: 
- User query embedded via MediaPipe BERT embedder
- Cosine similarity search in SQLite vector store identifies top-k seed nodes

**Phase 2 - Beam Traversal**:
- Beam width: 50 candidates
- Traversal depth: 5 hops
- Explores causal edges (TRIGGERED_BY, HAS_SYMPTOM, TARGETS, PART_OF)
- Maintains frontier of most-relevant nodes at each depth

**Phase 3 - Reranking**:
- Composite scoring: `score = semantic_similarity × temporal_relevance × causal_strength`
- Final ranked results returned for context generation

### First Aid RAG

55 curated Mayo Clinic first aid articles are embedded and stored in SQLite:
- Titles embedded via MediaPipe BERT
- Retrieval via cosine similarity at query time
- All processing fully offline, no external dependencies

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/el-profesor-04/aegis-health.git
   cd aegis-health
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
aegis-health/
├── graph/
│   ├── temporal_graph.py            # Core temporal graph engine
│   ├── impact_classes.py            # Impact class definitions and decay
│   └── schema.sql                   # SQLite schema
├── models/
│   ├── bert_embedder.py             # MediaPipe BERT embedder wrapper
│   └── quantization.py              # Quantization utilities
├── retrieval/
│   ├── seed_finder.py               # Phase 1: seed finding
│   ├── beam_search.py               # Phase 2: beam traversal
│   └── reranker.py                  # Phase 3: reranking
├── rag/
│   ├── first_aid_db.py              # First aid knowledge base
│   └── articles/                    # 55 Mayo Clinic articles
├── processing/
│   ├── entity_extractor.py          # Health event entity extraction
│   └── graph_builder.py             # Graph node/edge construction
├── scripts/
│   ├── build_first_aid_db.py        # First aid DB initialization
│   └── process_health_logs.py       # Health event batch processing
├── tests/
│   └── ...                          # Unit and integration tests
├── requirements.txt                 # Python dependencies
└── README.md
```

## SQLite Schema Overview

The temporal health graph is persisted in SQLite with the following primary tables:

- `nodes`: Health events with impact class, timestamp, and semantic embedding
- `edges`: Causal relationships between nodes (TRIGGERED_BY, HAS_SYMPTOM, TARGETS, PART_OF)
- `first_aid_articles`: Curated first aid knowledge base with embeddings

Indices are created on timestamp, impact_class, and embedding vectors for efficient retrieval.

## Usage

### Building the Graph from Health Logs

```python
from graph.temporal_graph import TemporalHealthGraph

graph = TemporalHealthGraph(db_path="health.db")
graph.add_node(
    event_type="symptom",
    description="headache",
    impact_class="C2",
    timestamp=1234567890
)
graph.add_edge(from_node_id=1, to_node_id=2, relationship="TRIGGERED_BY")
```

### Querying the Graph

```python
from retrieval.seed_finder import SeedFinder
from retrieval.beam_search import BeamSearch
from retrieval.reranker import Reranker

seed_finder = SeedFinder(graph)
seeds = seed_finder.find(query="Why do I get headaches after eating cheese?")

beam_search = BeamSearch(graph, width=50, depth=5)
candidates = beam_search.traverse(seeds)

reranker = Reranker()
final_results = reranker.rerank(candidates, query_embedding)
```

## Privacy & Security

- **Offline-First**: All processing occurs on-device without cloud connectivity
- **No Data Egress**: Health data never leaves the user's device
- **Sovereign Computation**: Users retain complete control over their health data
- **Local SQLite**: All graph data persists locally in encrypted SQLite

## Testing

Run tests with:
```bash
pytest tests/
```

## License

Proprietary — Aegis Health
