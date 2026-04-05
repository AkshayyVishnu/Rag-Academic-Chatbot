"""
Retrieval module — BM25, Hybrid, and Cross-Encoder Reranking
=============================================================
Strategies
----------
  semantic      Pure vector similarity (existing behaviour — no extra deps)
  hybrid_bm25   Semantic + BM25, merged via weighted score fusion
  hybrid_rrf    Semantic + BM25, merged via Reciprocal Rank Fusion (RRF)

Cross-encoder reranking is applied on top of any strategy when enabled.

Dependencies (install only what you need):
  pip install rank-bm25                           # BM25 / hybrid
  pip install sentence-transformers               # cross-encoder reranking

Quick start
-----------
  from retrieval import build_retriever

  retriever = build_retriever(
      vectorstore,
      strategy="hybrid_rrf",   # "semantic" | "hybrid_bm25" | "hybrid_rrf"
      top_k=4,
      use_reranker=True,
      reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
  )
  docs = retriever(query_text, embed_fn=make_embeddings(key).embed_query)
"""

from __future__ import annotations

import math
import re
import time
from typing import Callable, List, Optional

from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BM25 Index
# ─────────────────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Thin wrapper around rank_bm25.BM25Okapi that stores the original
    LangChain Document objects so retrieval returns them directly.

    Build once per vectorstore (expensive), then reuse.
    """

    def __init__(self, docs: List[Document]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for BM25 retrieval.\n"
                "Install with: pip install rank-bm25"
            )

        self.docs = docs
        tokenized = [_tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> List[tuple[Document, float]]:
        """Return (doc, raw_bm25_score) pairs, highest score first."""
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Pair each score with its index, sort descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.docs[i], float(s)) for i, s in ranked[:top_k]]


def build_bm25_index(vectorstore) -> BM25Index:
    """
    Pull all documents from a Chroma vectorstore and build a BM25 index.
    Call this once after loading the vectorstore; pass the result to build_retriever().
    """
    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    print(f"  [BM25] Indexed {len(docs)} documents.")
    return BM25Index(docs)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Fusion strategies
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_fuse(
    semantic_docs: List[Document],
    bm25_docs: List[tuple[Document, float]],
    top_k: int,
    k: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion.
    Each list contributes 1/(k + rank) to the document's score.
    Documents are matched by their page_content.
    """
    scores: dict[str, float] = {}
    content_to_doc: dict[str, Document] = {}

    for rank, doc in enumerate(semantic_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        content_to_doc[key] = doc

    for rank, (doc, _) in enumerate(bm25_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        content_to_doc[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [content_to_doc[k] for k, _ in ranked[:top_k]]


def _weighted_fuse(
    semantic_docs: List[Document],
    bm25_docs: List[tuple[Document, float]],
    top_k: int,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Document]:
    """
    Weighted score fusion.
    Semantic scores are approximated as 1/(1+rank); BM25 scores are
    min-max normalized then weighted.
    """
    scores: dict[str, float] = {}
    content_to_doc: dict[str, Document] = {}

    # Semantic: rank-based proxy score (1 = best)
    for rank, doc in enumerate(semantic_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + semantic_weight * (1.0 / rank)
        content_to_doc[key] = doc

    # BM25: min-max normalize then weight
    raw_scores = [s for _, s in bm25_docs]
    lo, hi = (min(raw_scores), max(raw_scores)) if raw_scores else (0.0, 1.0)
    span = hi - lo if hi != lo else 1.0

    for rank, (doc, s) in enumerate(bm25_docs, start=1):
        key = doc.page_content
        norm = (s - lo) / span
        scores[key] = scores.get(key, 0.0) + bm25_weight * norm
        content_to_doc[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [content_to_doc[k] for k, _ in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Cross-Encoder Reranker
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for post-retrieval reranking.
    The model is loaded lazily on first use.
    """

    _DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self._DEFAULT_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for reranking.\n"
                    "Install with: pip install sentence-transformers"
                )
            print(f"  [Reranker] Loading {self.model_name} …")
            self._model = CrossEncoder(self.model_name)
            print(f"  [Reranker] Ready.")

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """Score (query, passage) pairs and return top_k docs by score."""
        if not docs:
            return docs
        self._load()
        pairs = [(query, d.page_content) for d in docs]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  High-level retriever factory
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Stateful retriever that holds the vectorstore, optional BM25 index,
    and optional reranker.  Call it like a function:

        docs = retriever(query_text, embed_fn=embed_fn)

    Parameters
    ----------
    vectorstore     Chroma (or any LangChain vectorstore with similarity_search_by_vector)
    strategy        "semantic" | "hybrid_bm25" | "hybrid_rrf"
    top_k           Final number of docs to return
    fetch_k         Candidates fetched from each sub-retriever before fusion/reranking
    bm25_index      Pre-built BM25Index (required for hybrid strategies)
    use_reranker    Whether to apply cross-encoder reranking as a final step
    reranker_model  HuggingFace model ID for the cross-encoder
    semantic_weight Weight for semantic scores in weighted fusion (hybrid_bm25 only)
    bm25_weight     Weight for BM25 scores in weighted fusion (hybrid_bm25 only)
    """

    def __init__(
        self,
        vectorstore,
        strategy: str = "semantic",
        top_k: int = 4,
        fetch_k: int = 20,
        bm25_index: Optional[BM25Index] = None,
        use_reranker: bool = False,
        reranker_model: Optional[str] = None,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        valid = ("semantic", "hybrid_bm25", "hybrid_rrf")
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}, got {strategy!r}")

        if strategy.startswith("hybrid") and bm25_index is None:
            raise ValueError(
                "bm25_index is required for hybrid strategies. "
                "Call build_bm25_index(vectorstore) first."
            )

        self.vectorstore = vectorstore
        self.strategy = strategy
        self.top_k = top_k
        self.fetch_k = max(fetch_k, top_k * 3)  # always fetch more than needed
        self.bm25_index = bm25_index
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        self._reranker: Optional[CrossEncoderReranker] = (
            CrossEncoderReranker(reranker_model) if use_reranker else None
        )

    # ------------------------------------------------------------------
    def __call__(
        self,
        query: str,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> List[Document]:
        """
        Retrieve top_k documents for query.

        Parameters
        ----------
        query       The user question (plain text).
        embed_fn    A callable that takes a string and returns a vector.
                    Required for semantic / hybrid strategies.
                    Signature: embed_fn(text: str) -> List[float]
        """
        if self.strategy == "semantic":
            docs = self._semantic(query, embed_fn, k=self.top_k)
        elif self.strategy == "hybrid_bm25":
            docs = self._hybrid_weighted(query, embed_fn)
        else:  # hybrid_rrf
            docs = self._hybrid_rrf(query, embed_fn)

        if self._reranker is not None:
            docs = self._reranker.rerank(query, docs, self.top_k)

        return docs

    # ------------------------------------------------------------------
    # Private retrieval methods
    # ------------------------------------------------------------------

    def _semantic(
        self,
        query: str,
        embed_fn: Optional[Callable],
        k: int,
    ) -> List[Document]:
        if embed_fn is None:
            raise ValueError("embed_fn is required for semantic retrieval")
        vec = embed_fn(query)
        return self.vectorstore.similarity_search_by_vector(vec, k=k)

    def _hybrid_weighted(
        self,
        query: str,
        embed_fn: Optional[Callable],
    ) -> List[Document]:
        sem_docs = self._semantic(query, embed_fn, k=self.fetch_k)
        bm25_docs = self.bm25_index.search(query, top_k=self.fetch_k)
        return _weighted_fuse(
            sem_docs, bm25_docs, self.top_k,
            self.semantic_weight, self.bm25_weight,
        )

    def _hybrid_rrf(
        self,
        query: str,
        embed_fn: Optional[Callable],
    ) -> List[Document]:
        sem_docs = self._semantic(query, embed_fn, k=self.fetch_k)
        bm25_docs = self.bm25_index.search(query, top_k=self.fetch_k)
        return _rrf_fuse(sem_docs, bm25_docs, self.top_k)


def build_retriever(
    vectorstore,
    strategy: str = "semantic",
    top_k: int = 4,
    fetch_k: int = 20,
    use_reranker: bool = False,
    reranker_model: Optional[str] = None,
    bm25_index: Optional[BM25Index] = None,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> Retriever:
    """
    Convenience factory.  Automatically builds a BM25 index from the
    vectorstore when a hybrid strategy is requested and no index is supplied.

    Example
    -------
    retriever = build_retriever(
        vectorstore,
        strategy="hybrid_rrf",
        top_k=4,
        use_reranker=True,
    )
    docs = retriever(question, embed_fn=make_embeddings(key).embed_query)
    """
    if strategy.startswith("hybrid") and bm25_index is None:
        print("  [Retriever] Building BM25 index from vectorstore…")
        bm25_index = build_bm25_index(vectorstore)

    return Retriever(
        vectorstore=vectorstore,
        strategy=strategy,
        top_k=top_k,
        fetch_k=fetch_k,
        bm25_index=bm25_index,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
    )
