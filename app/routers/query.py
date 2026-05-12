from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_graph
from app.schemas import QueryRequest, QueryResponse
from reasoning.engine import reason_about_query


router = APIRouter(prefix="/query", tags=["query"])


def _memory_preview(event):
    return {
        "event_id": event.get("event_id"),
        "text": event.get("raw_text"),
        "event_time": event.get("event_time"),
        "event_type": event.get("event_type"),
        "severity_band": event.get("severity_band"),
        "memory_type": event.get("impact_label") or event.get("impact_class"),
        "matched": event.get("matched_entities", []),
    }


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest, graph=Depends(get_graph)):
    try:
        result = reason_about_query(
            graph,
            request.question,
            limit=request.limit,
            now=request.query_time,
        )
        bundle = result.get("evidence_bundle", {})
        return QueryResponse(
            answer=result["answer"],
            memories=[
                _memory_preview(event)
                for event in result.get("events", [])[: min(request.limit, 10)]
            ],
            query_plan=bundle.get("query_plan", {}),
            missing_information=bundle.get("missing_information", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
