"""
Pipeline Configuration — version-controlled benchmark settings
==============================================================
HOW TO USE
----------
1. Before each evaluation run, bump VERSION and update any changed params.
2. If chunking or embedding params change, set a new CHROMA_COLLECTION so the
   old and new vector stores don't collide.
3. Run:  python benchmarks/evaluate.py
4. Compare results in:  benchmarks/evaluation_log.csv

VERSIONING CONVENTION
---------------------
v1_baseline          — initial setup, no optimisations
v2_chunk<size>       — changed CHUNK_SIZE
v3_hybrid_bm25       — added BM25 hybrid retrieval
v4_reranker          — added cross-encoder reranker
v5_contextual_embed  — contextual / late-chunking embeddings
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Active pipeline version — change this before every evaluation run
# ─────────────────────────────────────────────────────────────────────────────

class PipelineConfig:

    # ── Version ───────────────────────────────────────────────────────────────
    VERSION     = "v2_hybrid_rrf"
    DESCRIPTION = (
        "Hybrid RRF retrieval: recursive char chunking (1000/200), "
        "Gemini embedding-001 (768-dim), Groq LLaMA-3.3-70b, top-4 semantic+BM25 RRF"
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE        = 1000          # characters per chunk
    CHUNK_OVERLAP     = 200           # character overlap between adjacent chunks
    CHUNKING_STRATEGY = "recursive"   # "recursive" | "sentence" | "semantic"

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL        = "models/gemini-embedding-001"
    EMBEDDING_DIMENSIONS   = 768      # 768 (efficient) or 3072 (full)
    USE_CONTEXTUAL_EMBEDDINGS = False # prepend chunk context before embedding

    # ── Vector store ──────────────────────────────────────────────────────────
    # IMPORTANT: change CHROMA_COLLECTION whenever CHUNK_SIZE, CHUNK_OVERLAP,
    # EMBEDDING_MODEL, or EMBEDDING_DIMENSIONS changes — otherwise old vectors
    # will pollute the new collection.
    CHROMA_DB_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    CHROMA_COLLECTION = "nitw_documents"   # v1 uses the existing production collection

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K             = 4             # chunks retrieved per query
    RETRIEVAL_TYPE    = "hybrid_rrf"  # "semantic" | "hybrid_bm25" | "hybrid_rrf"
    USE_RERANKER      = False         # cross-encoder reranking after initial retrieval
    RERANKER_MODEL    = None          # e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_MULTI_QUERY   = False         # generate query variants and merge results

    # ── LLM ───────────────────────────────────────────────────────────────────
    # Provider is read from .env (LLM_PROVIDER=groq|google).
    # The evaluate script will pick the right model name automatically.
    LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "google").lower()
    LLM_MODEL_GROQ    = "llama-3.3-70b-versatile"
    LLM_MODEL_GOOGLE  = "gemini-2.0-flash"
    LLM_TEMPERATURE   = 0.1
    LLM_MAX_RETRIES   = 2

    # Convenience: active model name (used in logs)
    @classmethod
    def active_llm_model(cls) -> str:
        return cls.LLM_MODEL_GROQ if cls.LLM_PROVIDER == "groq" else cls.LLM_MODEL_GOOGLE

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # Inter-request delay (seconds) during batch evaluation.
    # Formula: 60 / (RPM_per_key × number_of_keys)
    # Groq free tier: ~30 RPM/key.  Google free tier: ~15 RPM/key.
    REQUEST_DELAY_SECS = 1.0          # conservative default; tune per key count

    # ── Evaluation paths ──────────────────────────────────────────────────────
    TEST_DATASET      = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_dataset", "golden_set.json"
    )
    EVAL_LOG          = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmarks", "evaluation_log.csv"
    )
    EVAL_DETAILS_DIR  = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmarks", "details"
    )

    # ── System prompt ─────────────────────────────────────────────────────────
    # Must stay in sync with Pipeline/rag_chain.py SYSTEM_PROMPT.
    SYSTEM_PROMPT = (
        "You are an AI assistant for NIT Warangal students. Your job is to answer\n"
        "questions about academic regulations, policies, syllabi, and circulars based\n"
        "ONLY on the provided context from official NIT Warangal documents.\n\n"
        "RULES YOU MUST FOLLOW:\n"
        "1. Answer ONLY based on the provided context. Do NOT use any outside knowledge.\n"
        "2. If the context does not contain enough information to answer the question, say:\n"
        '   "I don\'t have enough information in the available documents to answer this question."\n'
        "3. Always cite the source document at the end of your answer like:\n"
        "   (Source: <document name>)\n"
        "4. Keep answers clear, concise, and directly relevant to the question.\n"
        "5. If the question is ambiguous, briefly explain what interpretations are possible\n"
        "   and answer the most likely one.\n"
        "6. Use bullet points for lists, but keep explanations in paragraph form.\n"
        "7. If the context contains specific numbers, rules, or deadlines, include them exactly.\n\n"
        "CONTEXT FROM NIT WARANGAL DOCUMENTS:\n"
        "{context}"
    )

    # ── Utility methods ───────────────────────────────────────────────────────
    @classmethod
    def to_dict(cls) -> dict:
        """Export all uppercase config keys as a plain dict (for logging)."""
        return {
            k: v for k, v in cls.__dict__.items()
            if k.isupper() and not callable(v)
        }

    @classmethod
    def summary(cls) -> str:
        """One-line description used in log rows and console output."""
        return (
            f"{cls.VERSION} | "
            f"chunk={cls.CHUNK_SIZE}/{cls.CHUNK_OVERLAP} | "
            f"embed={cls.EMBEDDING_DIMENSIONS}d | "
            f"retrieval={cls.RETRIEVAL_TYPE} | "
            f"top_k={cls.TOP_K} | "
            f"reranker={cls.USE_RERANKER} | "
            f"multi_query={cls.USE_MULTI_QUERY} | "
            f"llm={cls.active_llm_model()}"
        )

    @classmethod
    def validate(cls) -> None:
        """Raise if critical settings are missing or inconsistent."""
        assert os.path.isdir(cls.CHROMA_DB_DIR), (
            f"ChromaDB directory not found: {cls.CHROMA_DB_DIR}\n"
            "Run Pipeline/embedding.py first."
        )
        assert os.path.isfile(cls.TEST_DATASET), (
            f"Golden dataset not found: {cls.TEST_DATASET}\n"
            "Expected: test_dataset/golden_set.json"
        )
        assert cls.EMBEDDING_DIMENSIONS in (768, 1024, 3072), (
            f"Unusual embedding dimension: {cls.EMBEDDING_DIMENSIONS}"
        )
        assert cls.LLM_TEMPERATURE >= 0.0, "LLM_TEMPERATURE must be >= 0"
        assert cls.TOP_K >= 1, "TOP_K must be >= 1"
        if cls.LLM_PROVIDER == "groq":
            assert os.getenv("GROQ_API_KEY"), "LLM_PROVIDER=groq but GROQ_API_KEY not set in .env"
        assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY not set in .env (required for embeddings)"


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PIPELINE CONFIG — VALIDATION")
    print("=" * 60)
    try:
        PipelineConfig.validate()
        print("  All checks passed.\n")
    except AssertionError as e:
        print(f"  FAILED: {e}\n")

    print(f"  Version     : {PipelineConfig.VERSION}")
    print(f"  Description : {PipelineConfig.DESCRIPTION}")
    print()
    print(f"  Summary     : {PipelineConfig.summary()}")
    print()
    print("  Full config:")
    for k, v in PipelineConfig.to_dict().items():
        val = str(v)
        if len(val) > 80:
            val = val[:77] + "..."
        print(f"    {k:<28} {val}")
    print("=" * 60)
