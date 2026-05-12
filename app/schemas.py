from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    reference_time: datetime | None = None


class IngestResponse(BaseModel):
    event_id: str
    cyclic_promotions: list[dict[str, Any]]
    node_count: int
    edge_count: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=50)
    query_time: datetime | None = None


class QueryResponse(BaseModel):
    answer: str
    memories: list[dict[str, Any]]
    query_plan: dict[str, Any]
    missing_information: list[str]


class GraphSummaryResponse(BaseModel):
    node_count: int
    edge_count: int
    event_count: int
    cyclic_candidate_count: int
    cyclic_count: int
    impact_classes: dict[str, int]


class GraphClearResponse(BaseModel):
    ok: bool
    node_count: int
    edge_count: int


class EventResponse(BaseModel):
    event_id: str
    name: str | None
    event_type: str | None
    event_time: str | None
    impact_class: str | None
    severity_band: str | None
    raw_text: str | None
