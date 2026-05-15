from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_graph
from app.schemas import QueryRequest, QueryResponse
from reasoning.engine import reason_about_query


router = APIRouter(prefix="/query", tags=["query"])


def _memory_preview(event):
    # impact_label is set by pipeline ("Transient", "Acute" etc)
    # fall back to impact_class code (C1–C5) if label missing
    impact = (
        event.get("impact_label")
        or event.get("impact_class")
        or "?"
    )
    return {
        "event_id":    event.get("event_id"),
        "text":        event.get("raw_text"),
        "event_time":  event.get("event_time"),
        "event_type":  event.get("event_type"),
        "impact_class": event.get("impact_class"),
        "severity_band": event.get("severity_band"),
        "memory_type": impact,
        "matched":     event.get("matched_entities", []),
        "score":       round(event.get("score", 0.0), 3),
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
                for event in result.get("events", [])[:min(request.limit, 10)]
            ],
            query_plan=bundle.get("query_plan", {}),
            missing_information=bundle.get("missing_information", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc