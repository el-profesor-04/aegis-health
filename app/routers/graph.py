from fastapi import APIRouter, Depends, Query

from app.dependencies import get_graph
from app.schemas import EventResponse, GraphClearResponse, GraphSummaryResponse
from reasoning.knowledge import clear_knowledge_base
from reasoning.engine import generate_clinical_summary
from reasoning.insight_engine import get_all_insights


router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/summary")
def get_clinical_summary(graph=Depends(get_graph)):
    summary = generate_clinical_summary(graph)
    return {"summary": summary}


@router.get("/insights")
def get_insights(now: str = None, graph=Depends(get_graph)):
    return get_all_insights(graph, now=now)


@router.delete("", response_model=GraphClearResponse)
def clear_graph(
    include_knowledge_base: bool = Query(False, description="Also delete the first aid knowledge base"),
    graph=Depends(get_graph)
):
    graph.clear()
    if include_knowledge_base:
        clear_knowledge_base()
        
    return GraphClearResponse(
        ok=True,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
    )


@router.get("/summary", response_model=GraphSummaryResponse)
def summary(graph=Depends(get_graph)):
    events = graph.get_event_nodes()
    impact_counts = {}
    if not events.empty and "impact_class" in events.columns:
        impact_counts = {
            str(key): int(value)
            for key, value in events["impact_class"].value_counts().items()
        }

    return GraphSummaryResponse(
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        event_count=len(events),
        cyclic_candidate_count=int(events["cyclic_candidate"].sum()) if not events.empty else 0,
        cyclic_count=int(events["is_cyclic"].sum()) if not events.empty else 0,
        impact_classes=impact_counts,
    )


@router.get("/events", response_model=list[EventResponse])
def events(limit: int = 50, graph=Depends(get_graph)):
    rows = graph.get_event_nodes().tail(limit)
    output = []
    for _, row in rows.iterrows():
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
        output.append(EventResponse(
            event_id=row["node_id"],
            name=row["canonical_name"],
            event_type=row.get("event_type"),
            event_time=row.get("event_time"),
            impact_class=row.get("impact_class"),
            severity_band=row.get("severity_band"),
            raw_text=metadata.get("raw_text"),
        ))
    return output
