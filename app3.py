# app3.py
"""
FastAPI backend for the Caterer RAG API.

Endpoints:
  POST /search  — accepts a query + top_k, returns structured caterer results
  GET  /health  — simple health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.search import RAGSearch, SearchResponse


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Caterer RAG API",
    description="API for searching and recommending caterers using FAISS + Groq LLM",
    version="1.0.0",
)


# ============================================================
# CORS — allow Streamlit (and any local dev origin) to call us
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Startup — initialise RAG pipeline once at server boot
# ============================================================

rag: RAGSearch | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global rag
    print("[INFO] Initialising RAG pipeline...")
    rag = RAGSearch(
        persist_dir="caterer_faiss_store",
        embedding_model="BAAI/bge-base-en-v1.5",
        llm_model="llama-3.3-70b-versatile",
    )
    print("[INFO] RAG pipeline ready.")


# ============================================================
# Request / Response schemas
# ============================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


# ============================================================
# Routes
# ============================================================

@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_caterers(request: SearchRequest) -> SearchResponse:
    """
    Run the RAG pipeline and return structured caterer results.

    - **query**: Natural language search (e.g. "Find wedding caterers in Karnal for 500 guests")
    - **top_k**: How many caterers to return (1–10)
    """

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready yet. Please try again in a moment.",
        )

    try:
        response: SearchResponse = rag.search_and_summarize(
            query=request.query,
            top_k=request.top_k,
        )
        return response

    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )
