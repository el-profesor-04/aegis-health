from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.routers import graph, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-embed medical knowledge base at startup so first query is fast
    try:
        from reasoning.knowledge import warmup
        warmup()
    except Exception as e:
        print(f"Knowledge base warmup failed (non-fatal): {e}")
    yield


app = FastAPI(
    title="Aegis Health Graph API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(graph.router)