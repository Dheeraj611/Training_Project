# src/vectorstore.py
"""
FAISS vector store with BGE-large embeddings, hybrid smart re-ranking,
and optimized chunking for single-record documents.
"""

import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import faiss
import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ==============================================================
# EMBEDDING PIPELINE
# ==============================================================

class EmbeddingPipeline:
    """
    Wraps BAAI/bge-large-en-v1.5 for document and query embedding.

    BGE models require a special query prefix at inference time
    (not at indexing time) for best retrieval performance.
    """

    # BGE query prefix — injected ONLY during query embedding
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name

        # Select best available device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

        print(
            f"[INFO] Loaded embedding model: "
            f"{model_name} on {self.device.upper()}"
        )

    # ----------------------------------------------------------

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Split documents into chunks.
        For single-record caterer docs, chunk_size=512 keeps each
        record intact while still fitting within model token limits.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        print(
            f"[INFO] Split {len(documents)} documents "
            f"into {len(chunks)} chunks."
        )

        return chunks

    # ----------------------------------------------------------

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts:     List of strings to embed.
            batch_size: Batch size for encoding.
            is_query:  If True, prepend the BGE query prefix.
                       Always False when indexing documents.
        """
        if not texts:
            return np.empty(
                (0, self.model.get_sentence_embedding_dimension()),
                dtype=np.float32,
            )

        if is_query:
            texts = [
                self.BGE_QUERY_PREFIX + t
                for t in texts
            ]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalize at source
        )

        return embeddings.astype("float32")


# ==============================================================
# FAISS VECTOR STORE
# ==============================================================

class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

        self.emb_pipe = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ============================================================
    # BUILD
    # ============================================================

    def build_from_documents(
        self,
        documents: List[Document],
        batch_size: int = 32,
    ):
        if not documents:
            print("[WARN] No documents provided.")
            return

        print(
            f"[INFO] Building vector store from "
            f"{len(documents)} raw documents..."
        )

        chunks = self.emb_pipe.chunk_documents(documents)

        if not chunks:
            print("[WARN] No chunks generated.")
            return

        texts = [chunk.page_content for chunk in chunks]

        # Documents are NOT queries — is_query=False
        embeddings = self.emb_pipe.embed_texts(
            texts,
            batch_size=batch_size,
            is_query=False,
        )

        metadatas = []
        for chunk in chunks:
            metadata = {
                "page_content": chunk.page_content,
                **chunk.metadata,
            }
            metadatas.append(metadata)

        self.add_embeddings(embeddings, metadatas)
        self.save()

    # ============================================================
    # ADD EMBEDDINGS
    # ============================================================

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
    ):
        if embeddings.size == 0:
            print("[WARN] Empty embeddings.")
            return

        embeddings = np.ascontiguousarray(
            embeddings, dtype=np.float32
        )

        # Embeddings already normalized by encode(normalize_embeddings=True)
        # but re-normalize here as a safety measure for IndexFlatIP cosine sim
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)
        self.metadata.extend(metadatas)

        print(
            f"[INFO] Index now contains "
            f"{self.index.ntotal} total vectors."
        )

    # ============================================================
    # SAVE
    # ============================================================

    def save(self):
        if self.index is None:
            print("[WARN] Nothing to save.")
            return

        faiss_path = self.persist_dir / "faiss.index"
        meta_path = self.persist_dir / "metadata.pkl"

        faiss.write_index(self.index, str(faiss_path))

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(
            f"[INFO] Saved FAISS index "
            f"({self.index.ntotal} entries) "
            f"to {self.persist_dir}"
        )

    # ============================================================
    # LOAD
    # ============================================================

    def load(self) -> bool:
        faiss_path = self.persist_dir / "faiss.index"
        meta_path = self.persist_dir / "metadata.pkl"

        if not faiss_path.exists() or not meta_path.exists():
            print(
                f"[WARN] No existing FAISS store "
                f"found at {self.persist_dir}"
            )
            return False

        try:
            loaded_index = faiss.read_index(str(faiss_path))

            # ── Dimension guard ───────────────────────────────────
            # If the saved index was built with a different model,
            # its vector dimension will differ from the current model.
            # Detect this and force a rebuild rather than crash later.
            expected_dim = self.emb_pipe.model.get_embedding_dimension()
            if loaded_index.d != expected_dim:
                print(
                    f"[WARN] Dimension mismatch: index has {loaded_index.d}-dim "
                    f"vectors but current model produces {expected_dim}-dim. "
                    f"Rebuilding index..."
                )
                return False

            self.index = loaded_index

            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

            print(
                f"[INFO] Loaded "
                f"{self.index.ntotal} vectors "
                f"from {self.persist_dir}"
            )
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load FAISS store: {e}")
            return False

    # ============================================================
    # BASIC VECTOR SEARCH
    # ============================================================

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_fn: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:

        if self.index is None or self.index.ntotal == 0:
            print("[WARN] Search on empty index.")
            return []

        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        query_embedding = np.ascontiguousarray(
            query_embedding, dtype=np.float32
        )
        faiss.normalize_L2(query_embedding)

        fetch_k = min(
            top_k * 3 if filter_fn else top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(query_embedding, fetch_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            if score < score_threshold:
                continue
            if idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            if filter_fn and not filter_fn(meta):
                continue

            results.append({
                "score": round(float(score), 4),
                "content": meta.get("page_content", ""),
                "metadata": {
                    k: v
                    for k, v in meta.items()
                    if k != "page_content"
                },
            })

            if len(results) >= top_k:
                break

        return results

    # ============================================================
    # BASIC TEXT QUERY
    # ============================================================

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.2,
        filter_fn: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed query (with BGE prefix) and search FAISS.
        score_threshold=0.2 filters low-confidence matches.
        """
        query_embedding = self.emb_pipe.embed_texts(
            [query_text],
            is_query=True,   # Inject BGE query prefix
        )[0]

        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_fn=filter_fn,
        )

    # ============================================================
    # SMART QUERY — Hybrid re-ranking
    # ============================================================

    def smart_query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with multi-signal re-ranking:

        Signals:
          1. Semantic similarity (BGE-large cosine score)
          2. Location boost   — +0.35 for word-boundary match
          3. Rating boost     — up to +0.25 when top/best intent detected
          4. Capacity boost   — +0.15 when capacity keyword present & matches
          5. Budget boost     — +0.10 for budget/tier keyword match

        Returns top_k results sorted by final composite score.
        """

        if self.index is None or self.index.ntotal == 0:
            print("[WARN] Smart query on empty index.")
            return []

        query_lower = query_text.lower().strip()

        # ── Intent detection ────────────────────────────────────
        top_keywords = [
            "top", "best", "highest rated", "highest rating",
            "popular", "top rated", "best rated", "recommended",
        ]
        is_top_requested = any(kw in query_lower for kw in top_keywords)

        # Capacity number detection  (e.g. "500 guests", "1000+")
        capacity_match = re.search(
            r"(\d[\d,]*)\s*(?:\+)?\s*(?:guests?|people|persons?|pax)?",
            query_lower,
        )
        capacity_requested = (
            int(capacity_match.group(1).replace(",", ""))
            if capacity_match
            else None
        )

        # Budget intent
        budget_keywords = ["luxury", "premium", "budget", "economy",
                           "affordable", "cheap", "high-end"]
        budget_intent = next(
            (kw for kw in budget_keywords if kw in query_lower), None
        )

        # ── Over-fetch candidates ───────────────────────────────
        candidate_k = min(max(top_k * 8, 30), self.index.ntotal)

        base_results = self.query(
            query_text,
            top_k=candidate_k,
            score_threshold=0.0,   # Let re-ranking handle filtering
        )

        if not base_results:
            return []

        # ── Re-rank ─────────────────────────────────────────────
        location_fields = [
            "state_area", "region_zone", "address",
            "location", "area", "state", "city",
        ]

        reranked = []
        for item in base_results:
            meta = item.get("metadata", {})
            original_score = float(item.get("score", 0.0))
            final_score = original_score

            # 1. LOCATION BOOST — word-boundary match for precision
            location_match = False
            for field in location_fields:
                value = str(meta.get(field, "") or "").lower()
                if not value:
                    continue
                # Word-boundary regex prevents "punjab" matching "punjabi"
                # when searching for "Punjab"
                words = re.findall(r"[a-z]+", value)
                for word in words:
                    if (
                        len(word) >= 3
                        and re.search(rf"\b{re.escape(word)}\b", query_lower)
                    ):
                        location_match = True
                        break
                if location_match:
                    break

            if location_match:
                final_score += 0.35

            # 2. RATING BOOST — only when top/best intent detected
            if is_top_requested:
                rating = meta.get("rating_score", meta.get("rating", 0.0))
                try:
                    rating = float(rating)
                except (ValueError, TypeError):
                    rating = 0.0
                rating = max(0.0, min(rating, 5.0))
                final_score += (rating / 5.0) * 0.25

            # 3. CAPACITY BOOST
            if capacity_requested:
                raw_cap = str(meta.get("guest_capacity", "") or "")
                # Extract first number from capacity string
                cap_match = re.search(r"(\d[\d,]*)", raw_cap)
                if cap_match:
                    caterer_cap = int(
                        cap_match.group(1).replace(",", "")
                    )
                    if caterer_cap >= capacity_requested:
                        final_score += 0.15

            # 4. BUDGET BOOST
            if budget_intent:
                caterer_budget = str(
                    meta.get("budget_tier", "") or ""
                ).lower()
                if budget_intent in caterer_budget:
                    final_score += 0.10

            reranked.append({
                "score": round(final_score, 4),
                "content": item.get("content", ""),
                "metadata": meta,
            })

        # Sort descending
        reranked.sort(key=lambda x: x["score"], reverse=True)

        return reranked[:top_k]