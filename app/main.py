from fastapi import FastAPI

from app.routers import graph, ingest, query


app = FastAPI(
    title="Aegis Health Graph API",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(graph.router)
