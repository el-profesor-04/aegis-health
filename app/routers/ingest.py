from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_graph
from app.schemas import IngestRequest, IngestResponse
from ingestion.cyclic_detector import run_cyclic_detection
from ingestion.pipeline import ingest_text


router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def ingest(request: IngestRequest, graph=Depends(get_graph)):
    try:
        event_id = ingest_text(
            graph,
            request.text,
            reference_time=request.reference_time,
        )

        # Extraction failed (LLM returned nothing parseable or validation
        # rejected the output). Return 200 with success=False instead of
        # crashing — the caller can log the skip and continue.
        if event_id is None:
            return IngestResponse(
                event_id=None,
                success=False,
                skip_reason="Input could not be converted into a valid event.",
                cyclic_promotions=[],
                node_count=len(graph.nodes),
                edge_count=len(graph.edges),
            )

        promotions = run_cyclic_detection(graph)
        graph.persist_all()

        return IngestResponse(
            event_id=event_id,
            success=True,
            cyclic_promotions=promotions,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc