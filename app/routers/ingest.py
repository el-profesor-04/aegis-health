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
        if event_id is None:
            raise HTTPException(status_code=422, detail="Input could not be converted into a valid event.")

        promotions = run_cyclic_detection(graph)
        graph.persist_all()

        return IngestResponse(
            event_id=event_id,
            cyclic_promotions=promotions,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
