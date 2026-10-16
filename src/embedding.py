# src/embedding.py
"""
Standalone embedding utilities (legacy module — core logic
has moved into vectorstore.EmbeddingPipeline).
Kept for backward compatibility.
"""

from typing import List, Dict, Any

import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:
    """
    BGE-large-en-v1.5 based embedding pipeline.

    Uses BAAI/bge-large-en-v1.5 — currently the best free open-source
    model on the MTEB retrieval benchmark (1024-dim, outperforms OpenAI ada-002).

    BGE query prefix is injected at query time (not indexing time)
    for optimal asymmetric retrieval performance.
    """

    # BGE standard query prefix for asymmetric retrieval
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Hardware acceleration
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = SentenceTransformer(model_name, device=self.device)
        print(
            f"[INFO] Loaded embedding model: "
            f"{model_name} on {self.device.upper()}"
        )

    def chunk_documents(
        self, documents: List[Document]
    ) -> List[Document]:
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

    def embed_chunks(
        self,
        chunks: List[Document],
        batch_size: int = 32,
        is_query: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Embed chunks and bundle text + metadata + vector together.

        Args:
            chunks:     List of LangChain Document objects.
            batch_size: Encoding batch size.
            is_query:   If True, prepend BGE query prefix.
        """
        texts = [chunk.page_content for chunk in chunks]

        if is_query:
            texts = [self.BGE_QUERY_PREFIX + t for t in texts]

        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalize for cosine similarity
        )

        processed_data = []
        for chunk, emb in zip(chunks, embeddings):
            processed_data.append({
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": emb.tolist(),
            })

        print(
            f"[INFO] Successfully packaged "
            f"{len(processed_data)} embedded chunks."
        )
        return processed_data