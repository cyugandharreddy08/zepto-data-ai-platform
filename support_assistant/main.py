import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from support_assistant.models import QueryRequest, QueryResponse
from support_assistant.graph import run_query
from support_assistant.rag import ingest_corpus, get_chroma_client, COLLECTION_NAME

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure ChromaDB collection is ingested on startup
    try:
        client = get_chroma_client()
        col = client.get_or_create_collection(COLLECTION_NAME)
        if col.count() == 0:
            print("ChromaDB collection is empty. Ingesting documents...")
            ingest_corpus()
        else:
            print(f"ChromaDB ready with {col.count()} documents.")
    except Exception as e:
        print(f"Warning during startup ingestion: {e}")
    yield

app = FastAPI(
    title="Zepto Support Assistant API",
    description="Grounded GenAI Support Assistant for Zepto policies powered by LangGraph, ChromaDB, and FastAPI.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Zepto Policy Support Assistant",
        "endpoints": {"ask": "/ask", "health": "/health", "docs": "/docs"}
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=QueryResponse)
def ask_policy_question(request: QueryRequest) -> QueryResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        response = run_query(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal graph error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("support_assistant.main:app", host="0.0.0.0", port=7860, reload=False)
