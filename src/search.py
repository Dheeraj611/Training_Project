# src/search.py
"""
Full RAG pipeline: FAISS retrieval → Groq LLM → Structured Pydantic output.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore

load_dotenv()


# ==============================================================
# OUTPUT SCHEMAS
# ==============================================================

class CatererResult(BaseModel):
    """Structured schema for a single caterer match."""

    name: str = Field(
        description="Full business name of the caterer"
    )
    location: str = Field(
        description="Full address or city/area of the caterer"
    )
    region: str = Field(
        description="Region or zone (e.g., North India, NCR)"
    )
    specialization: str = Field(
        description="Cuisine type or event specialization (e.g., Mughlai, Punjabi, Multi-cuisine)"
    )
    budget_tier: str = Field(
        description="Budget tier (e.g., Premium, Mid-Range, Economy)"
    )
    guest_capacity: str = Field(
        description="Maximum guest capacity (e.g., 500, 200-1000)"
    )
    rating: str = Field(
        description="Customer rating (e.g., 4.8/5)"
    )
    contact_phone: str = Field(
        default="",
        description="Phone number for the caterer, empty string if not available"
    )
    website: str = Field(
        default="",
        description="Website URL, empty string if not available"
    )
    match_reason: str = Field(
        description=(
            "A concise 1–2 sentence explanation of why this caterer "
            "matches the user's specific query."
        )
    )
    similarity_score: float = Field(
        description="Semantic similarity score from vector search (0.0 – 1.0+)"
    )


class SearchResponse(BaseModel):
    """Top-level structured response for a catering search query."""

    query: str = Field(
        description="The original user query"
    )
    total_found: int = Field(
        description="Number of matching caterers returned"
    )
    summary: str = Field(
        description=(
            "A brief 2–3 sentence conversational summary addressing "
            "the user's query and the key highlights from the results found."
        )
    )
    caterers: List[CatererResult] = Field(
        description="Ranked list of matching caterers with full details."
    )


# ==============================================================
# RAG SEARCH CLASS
# ==============================================================

class RAGSearch:
    """
    Full RAG pipeline:
      1. Embed query with BGE-large
      2. Smart hybrid FAISS retrieval + re-ranking
      3. Groq LLM (llama-3.3-70b-versatile) with structured output
    """

    def __init__(
        self,
        persist_dir: str = "caterer_faiss_store",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        llm_model: str = "llama-3.3-70b-versatile",
    ):
        # ── Vector store ──────────────────────────────────────
        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )

        if not self.vectorstore.load():
            print(
                "[INFO] No existing index found. "
                "Building from documents..."
            )
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)

        # ── Groq LLM ──────────────────────────────────────────
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "[ERROR] GROQ_API_KEY is missing. "
                "Please add it to your .env file."
            )

        llm = ChatGroq(
            api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.1,       # Low temp → deterministic, factual
            max_tokens=4096,
        )

        # Use json_mode — more reliable than tool-calling for complex schemas
        # tool_use mode fails when context has empty/ambiguous fields
        self.structured_llm = llm.with_structured_output(
            SearchResponse,
            method="json_mode",
        )

        print(
            f"[INFO] RAGSearch ready — "
            f"LLM: {llm_model} | "
            f"Embeddings: {embedding_model}"
        )

    # ----------------------------------------------------------
    # SEARCH AND SUMMARIZE
    # ----------------------------------------------------------

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 25,
    ) -> SearchResponse:
        """
        Run the full RAG pipeline and return a structured SearchResponse.

        Args:
            query:       Natural language user query.
            top_k:       Number of final caterers to return.
            candidate_k: How many FAISS candidates to retrieve before re-ranking.

        Returns:
            SearchResponse Pydantic model.
        """

        print(f"\n[RAG] Retrieving top {candidate_k} candidates...")

        # 1. Smart hybrid retrieval
        raw_results = self.vectorstore.smart_query(
            query_text=query,
            top_k=candidate_k,
        )

        if not raw_results:
            return SearchResponse(
                query=query,
                total_found=0,
                summary="No caterers found matching your query.",
                caterers=[],
            )

        # 2. Build enriched context for LLM
        def _val(v) -> str:
            """Return 'N/A' for falsy/empty values so LLM sees clean data."""
            return str(v).strip() if v else "N/A"

        context_blocks = []
        for i, r in enumerate(raw_results[:top_k * 3], start=1):
            meta = r.get("metadata", {})
            score = r.get("score", 0.0)

            block = (
                f"--- Caterer #{i} (Similarity: {score:.4f}) ---\n"
                f"Name:            {_val(meta.get('caterer_name'))}\n"
                f"Address:         {_val(meta.get('address'))}\n"
                f"State / Area:    {_val(meta.get('state_area'))}\n"
                f"Region / Zone:   {_val(meta.get('region_zone'))}\n"
                f"Specialization:  {_val(meta.get('specialization'))}\n"
                f"Budget Tier:     {_val(meta.get('budget_tier'))}\n"
                f"Guest Capacity:  {_val(meta.get('guest_capacity'))}\n"
                f"Rating:          {_val(meta.get('rating_raw'))} "
                f"(score: {_val(meta.get('rating_score'))})\n"
                f"Status:          {_val(meta.get('status'))}\n"
                f"Phone:           {_val(meta.get('contact_phone'))}\n"
                f"Website:         {_val(meta.get('website'))}\n"
                f"Details:         {r.get('content', '')[:300]}"
            )
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)

        # 3. Build structured prompt
        prompt = f"""You are an expert AI Catering Intelligence Assistant.
Your job is to analyze the user query and context data, then return a valid JSON response.

RESPONSE FORMAT — you MUST return JSON that matches this exact schema:
{{
  "query": "<the original user query>",
  "total_found": <integer count of caterers in the list>,
  "summary": "<2-3 sentence summary of findings>",
  "caterers": [
    {{
      "name": "<caterer name>",
      "location": "<address or N/A>",
      "region": "<region/zone or N/A>",
      "specialization": "<cuisine/event type>",
      "budget_tier": "<tier>",
      "guest_capacity": "<capacity>",
      "rating": "<rating string>",
      "contact_phone": "<phone or empty string>",
      "website": "<url or empty string>",
      "match_reason": "<1-2 sentences why this matches the query>",
      "similarity_score": <float from the context>
    }}
  ]
}}

SELECTION RULES:
1. Prefer caterers whose State/Area or Region matches the user's requested location.
2. If the user specifies guest capacity, prefer caterers that meet it; exclude those that clearly cannot.
3. If no caterers have location data (State/Area = N/A), return the top matches by rating and similarity.
4. Return at most {top_k} caterers. Set total_found = the number in the list.
5. For location/region fields: use the State/Area and Region/Zone values from context, or "N/A" if missing.
6. Do NOT invent names, phones, or ratings. Use only exact values from the context.
7. Write a concise, honest summary — note if location data is sparse.

Context Data:
{context}

User Query: {query}

Remember: respond ONLY with valid JSON matching the schema above.
"""

        print("[RAG] Invoking Groq LLM for structured output...")

        try:
            response: SearchResponse = self.structured_llm.invoke(prompt)
            # Patch query field in case LLM omits it
            response.query = query
            response.total_found = len(response.caterers)
            return response

        except Exception as e:
            print(f"[ERROR] LLM structured output failed: {e}")
            # Graceful fallback: build response from raw FAISS results
            return self._fallback_response(query, raw_results[:top_k])

    # ----------------------------------------------------------
    # FALLBACK (no LLM)
    # ----------------------------------------------------------

    def _fallback_response(
        self,
        query: str,
        results: list,
    ) -> SearchResponse:
        """Build a SearchResponse directly from FAISS results (no LLM)."""

        caterers = []
        for r in results:
            meta = r.get("metadata", {})
            caterers.append(
                CatererResult(
                    name=meta.get("caterer_name", "N/A"),
                    location=meta.get("address", "N/A"),
                    region=meta.get("region_zone", "N/A"),
                    specialization=meta.get("specialization", "N/A"),
                    budget_tier=meta.get("budget_tier", "N/A"),
                    guest_capacity=meta.get("guest_capacity", "N/A"),
                    rating=meta.get("rating_raw", "N/A"),
                    contact_phone=meta.get("contact_phone", ""),
                    website=meta.get("website", ""),
                    match_reason="Matched via semantic similarity search.",
                    similarity_score=float(r.get("score", 0.0)),
                )
            )

        return SearchResponse(
            query=query,
            total_found=len(caterers),
            summary=(
                f"Found {len(caterers)} caterers matching your query. "
                "(LLM summarization unavailable — showing raw results.)"
            ),
            caterers=caterers,
        )