"""
RAG Evaluation Script — NIT Warangal Chatbot
=============================================
Runs 100 golden questions through the pipeline and scores them on:

  CUSTOM METRICS (always run, no extra install needed)
  ├── answer_correctness   — LLM-as-judge: does the answer match ground truth?
  ├── idk_accuracy         — did the bot correctly refuse unanswerable questions?
  ├── hallucination_rate   — 1 - idk_accuracy
  ├── latency_p50/p95/avg  — end-to-end response time
  ├── accuracy_easy/medium/hard — per-difficulty breakdown
  ├── multi_hop_accuracy   — questions requiring cross-document reasoning
  ├── per_category         — accuracy per question category
  └── error_rate           — pipeline errors / crashes

  RAGAS METRICS (requires: pip install ragas datasets)
  ├── faithfulness         — is the answer grounded in retrieved context?
  ├── answer_relevancy     — does the answer address the question asked?
  ├── context_recall       — did retrieval surface all needed information?
  └── factual_correctness  — is the answer factually right vs ground truth?

Usage
-----
    python benchmarks/evaluate.py
    python benchmarks/evaluate.py --skip-ragas      # skip RAGAS (faster)
    python benchmarks/evaluate.py --limit 10        # quick smoke-test on 10 Qs
    python benchmarks/evaluate.py --version v2_xxx  # override VERSION in config
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Pipeline"))

from dotenv import load_dotenv
load_dotenv()

from config import PipelineConfig as Config

# ── validate before doing anything ───────────────────────────────────────────
try:
    Config.validate()
except AssertionError as e:
    print(f"\nConfig validation failed:\n  {e}")
    sys.exit(1)

# ── pipeline imports (key-rotating versions) ──────────────────────────────────
from rag_chain import (
    make_embeddings, make_llm,
    next_embed_key, next_llm_key,
    embed_keys, llm_keys,
)
from retrieval import build_retriever, BM25Index

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pipeline setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_pipeline():
    embeddings = make_embeddings(next_embed_key())
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=Config.CHROMA_COLLECTION,
    )
    count = vectorstore._collection.count()
    print(f"  ChromaDB '{Config.CHROMA_COLLECTION}': {count} chunks")
    print(f"  Retrieval strategy : {Config.RETRIEVAL_TYPE}")
    print(f"  Reranker enabled   : {Config.USE_RERANKER}"
          + (f" ({Config.RERANKER_MODEL})" if Config.USE_RERANKER else ""))

    retriever = build_retriever(
        vectorstore,
        strategy=Config.RETRIEVAL_TYPE,
        top_k=Config.TOP_K,
        use_reranker=Config.USE_RERANKER,
        reranker_model=Config.RERANKER_MODEL,
    )
    return vectorstore, retriever


_CONN_ERRORS = ("connection", "timeout", "network", "ssl", "connect", "remotedisconnected", "eof")


def _is_conn_err(e: Exception) -> bool:
    return any(x in str(e).lower() for x in _CONN_ERRORS)


def run_single_query(vectorstore, question: str, retriever=None) -> dict:
    """Retrieve + generate for one question. Returns timing + answer + contexts."""

    # ── Retrieval ─────────────────────────────────────────────────────────────
    t0 = time.time()
    docs = None
    max_attempts = len(embed_keys) * 3
    for attempt in range(max_attempts):
        key = next_embed_key()
        try:
            embed_fn = make_embeddings(key).embed_query
            if retriever is not None:
                docs = retriever(question, embed_fn=embed_fn)
            else:
                vec = embed_fn(question)
                docs = vectorstore.similarity_search_by_vector(vec, k=Config.TOP_K)
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            if _is_conn_err(e):
                wait = min(2 ** (attempt % 4), 30)
                time.sleep(wait)
                continue
            raise
    if docs is None:
        raise RuntimeError("Retrieval failed: all embed keys exhausted (429/connection)")
    retrieval_ms = round((time.time() - t0) * 1000)

    contexts     = [d.page_content for d in docs]
    sources      = [d.metadata.get("source", "unknown") for d in docs]
    chunk_ids    = [d.metadata.get("chunk_id", "") for d in docs]
    chunk_indexes = [d.metadata.get("chunk_index", "") for d in docs]

    context_str = "\n\n".join(
        f"--- Document {i+1} (Source: {src}) ---\n{ctx}"
        for i, (ctx, src) in enumerate(zip(contexts, sources))
    )

    # ── Generation ────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", Config.SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    t1 = time.time()
    answer = None
    max_attempts = len(llm_keys) * 3
    for attempt in range(max_attempts):
        key = next_llm_key()
        chain = prompt | make_llm(key) | StrOutputParser()
        try:
            answer = chain.invoke({"context": context_str, "question": question})
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_attempts - 1:
                    continue
            elif _is_conn_err(e):
                wait = min(2 ** (attempt % 4), 30)
                if attempt < max_attempts - 1:
                    time.sleep(wait)
                    continue
            raise
    if answer is None:
        raise RuntimeError("Generation failed: all LLM keys exhausted (429/connection)")
    generation_ms = round((time.time() - t1) * 1000)

    return {
        "answer":         answer,
        "contexts":       contexts,
        "sources":        sources,
        "chunk_ids":      chunk_ids,
        "chunk_indexes":  chunk_indexes,
        "retrieval_ms":   retrieval_ms,
        "generation_ms":  generation_ms,
        "total_ms":       retrieval_ms + generation_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Run all questions
# ─────────────────────────────────────────────────────────────────────────────

def run_all_questions(vectorstore, questions: list, retriever=None) -> list:
    results = []
    total = len(questions)

    print(f"\n  Running {total} questions through the pipeline...")
    print(f"  {'ID':<6} {'Category':<22} {'Diff':<7} {'ms':>6}  Status")
    print(f"  {'-'*6} {'-'*22} {'-'*7} {'-'*6}  {'-'*10}")

    for i, item in enumerate(questions):
        qid  = item["id"]
        q    = item["question"]
        cat  = item["category"]
        diff = item.get("difficulty", "?")

        try:
            res = run_single_query(vectorstore, q, retriever=retriever)
            res.update({
                "question_id":      qid,
                "question":         q,
                "ground_truth":     item["ground_truth"],
                "category":         cat,
                "difficulty":       diff,
                "requires_multi_hop": item.get("requires_multi_hop", False),
                "status":           "success",
            })
            print(f"  {qid:<6} {cat:<22} {diff:<7} {res['total_ms']:>6}  ok")
        except Exception as e:
            res = {
                "question_id":  qid,
                "question":     q,
                "ground_truth": item["ground_truth"],
                "category":     cat,
                "difficulty":   diff,
                "requires_multi_hop": item.get("requires_multi_hop", False),
                "answer":       f"ERROR: {e}",
                "contexts":     [],
                "sources":      [],
                "retrieval_ms": 0,
                "generation_ms":0,
                "total_ms":     0,
                "status":       "error",
            }
            print(f"  {qid:<6} {cat:<22} {diff:<7} {'—':>6}  ERROR: {e}")

        results.append(res)
        time.sleep(Config.REQUEST_DELAY_SECS)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom metrics
# ─────────────────────────────────────────────────────────────────────────────

# Phrases the model should use when it can't answer
REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not available in",
    "cannot find",
    "no information",
    "not in the documents",
    "not mentioned",
    "not specified",
    "not provided",
    "outside the scope",
    "not present",
    "unanswerable",
    "not available in the provided",
]

JUDGE_PROMPT = """\
You are a strict evaluator for a RAG chatbot that answers questions about NIT Warangal academic regulations.

Question: {question}

Chatbot answer:
{answer}

Ground truth (the correct answer):
{ground_truth}

Evaluation rules — be STRICT:
1. Every specific number, percentage, credit count, year, or threshold in the ground truth MUST appear correctly in the chatbot answer.
2. Every named condition or requirement (e.g. "AND", "OR", eligibility clauses) must be present.
3. For list-based answers, all key items must be present — missing 2+ items = INCORRECT.
4. Vague or hedged answers that avoid committing to the specific facts = INCORRECT.
5. Minor rewording is acceptable ONLY if all facts are preserved exactly.

Reply on line 1 with ONLY "CORRECT" or "INCORRECT".
Reply on line 2 with one sentence: what key fact was correct or what specific detail was missing/wrong."""


def _make_primary_judge(key: str):
    """
    Primary judge: qwen/qwen3-32b on Groq.
    Alibaba model — completely different family from the LLaMA-3.3-70b answerer.
    Wraps reasoning in <think> tags; _parse() strips those before reading the verdict.
    """
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="qwen/qwen3-32b",
        groq_api_key=key,
        temperature=0.0,
        max_retries=0,
    )


def _make_fallback_judge(key: str):
    """
    Fallback judge: gemma2-9b-it on Groq.
    Google Gemma — different architecture from both LLaMA (answerer) and Qwen (primary judge).
    Used when all primary judge keys are rate-limited.
    """
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="gemma2-9b-it",
        groq_api_key=key,
        temperature=0.0,
        max_retries=0,
    )


def llm_judge(question: str, answer: str, ground_truth: str) -> tuple[str, str]:
    """
    Strict LLM-as-judge.
    Primary  : Groq qwen/qwen3-32b  (Alibaba — different family from LLaMA answerer)
    Fallback : Groq gemma2-9b-it    (Google Gemma — third independent family)
    Returns (verdict, reasoning). verdict is 'CORRECT' | 'INCORRECT' | 'ERROR'.
    """
    prompt_text = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
    )

    def _parse(response: str) -> tuple[str, str]:
        import re
        # Strip <think>...</think> reasoning blocks (qwen3, deepseek, etc.)
        clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        first_line = clean.split("\n")[0].strip().upper()
        verdict = "CORRECT" if first_line.startswith("CORRECT") else "INCORRECT"
        return verdict, clean

    _CONNECTION_ERRORS = ("connection", "timeout", "network", "ssl", "connect", "remotedisconnected", "eof")

    def _is_connection_err(e: Exception) -> bool:
        return any(x in str(e).lower() for x in _CONNECTION_ERRORS)

    def _try_judge(make_fn, keys_list, next_key_fn, label):
        max_attempts = len(keys_list) * 2
        for attempt in range(max_attempts):
            key = next_key_fn()
            try:
                response = make_fn(key).invoke(prompt_text).content.strip()
                return _parse(response)
            except Exception as e:
                err = str(e)
                if attempt == 0:
                    short = err[:120].replace("\n", " ")
                    print(f"    [judge:{label}] error (key …{key[-6:]}): {short}")
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt < max_attempts - 1:
                        time.sleep(5)
                        continue
                elif _is_connection_err(e):
                    wait = min(2 ** (attempt % 4), 30)
                    if attempt < max_attempts - 1:
                        time.sleep(wait)
                        continue
                # Non-retriable or last attempt
                return None, str(e)
        return None, f"{label} keys exhausted"

    # ── Phase 1: qwen3-32b (primary) ─────────────────────────────────────────
    verdict, reasoning = _try_judge(_make_primary_judge, llm_keys, next_llm_key, "qwen3-32b")
    if verdict is not None:
        return verdict, reasoning

    # ── Phase 2: gemma2-9b-it (fallback) ─────────────────────────────────────
    print("    [judge] qwen3-32b exhausted — falling back to gemma2-9b-it")
    verdict, reasoning = _try_judge(_make_fallback_judge, llm_keys, next_llm_key, "gemma2-9b-it")
    if verdict is not None:
        return verdict, reasoning

    return "ERROR", f"All judge keys exhausted. Last error: {reasoning}"


def compute_custom_metrics(results: list) -> dict:
    metrics = {}

    # ── Latency ───────────────────────────────────────────────────────────────
    times = sorted(r["total_ms"] for r in results if r["status"] == "success")
    if times:
        metrics["latency_p50_ms"]  = times[len(times) // 2]
        metrics["latency_p95_ms"]  = times[int(len(times) * 0.95)]
        metrics["latency_avg_ms"]  = round(sum(times) / len(times))
        metrics["latency_min_ms"]  = times[0]
        metrics["latency_max_ms"]  = times[-1]

    # ── Unanswerable / hallucination ──────────────────────────────────────────
    # Only count successful pipeline runs — errors are not hallucinations
    unanswerable = [
        r for r in results
        if r["ground_truth"] == "NOT_IN_DOCUMENTS" and r["status"] == "success"
    ]
    if unanswerable:
        refused = sum(
            1 for r in unanswerable
            if any(p in r["answer"].lower() for p in REFUSAL_PHRASES)
        )
        metrics["idk_accuracy"]       = refused / len(unanswerable)
        metrics["hallucination_rate"] = 1 - metrics["idk_accuracy"]
        metrics["unanswerable_total"] = len(unanswerable)
        metrics["correct_refusals"]   = refused

    # ── Answerable: LLM-as-judge ──────────────────────────────────────────────
    answerable = [
        r for r in results
        if r["ground_truth"] != "NOT_IN_DOCUMENTS" and r["status"] == "success"
    ]

    print(f"\n  Running LLM-as-judge on {len(answerable)} answerable questions...")

    correct = 0
    for idx, r in enumerate(answerable):
        verdict, reasoning = llm_judge(r["question"], r["answer"], r["ground_truth"])
        r["judge_verdict"]    = verdict
        r["judge_reasoning"]  = reasoning
        if verdict == "CORRECT":
            correct += 1
        mark = "+" if verdict == "CORRECT" else ("?" if verdict == "ERROR" else "-")
        print(f"    [{mark}] {r['question_id']} — {verdict}")
        time.sleep(4)   # ~15 RPM per key; 9 keys → safe at 4s intervals

    if answerable:
        evaluated = [r for r in answerable if r.get("judge_verdict") in ("CORRECT", "INCORRECT")]
        judge_errors = len(answerable) - len(evaluated)
        metrics["answerable_total"]      = len(answerable)
        metrics["evaluated_questions"]   = len(evaluated)
        metrics["judge_error_skipped"]   = judge_errors
        metrics["correct_answers"]       = correct
        # Accuracy over only evaluated questions (skips ERROR'd ones)
        metrics["answer_correctness"]    = correct / len(evaluated) if evaluated else 0.0

    # ── Per-difficulty ────────────────────────────────────────────────────────
    for diff in ("easy", "medium", "hard"):
        bucket = [
            r for r in answerable
            if r.get("difficulty") == diff and r.get("judge_verdict") in ("CORRECT", "INCORRECT")
        ]
        if bucket:
            n_correct = sum(1 for r in bucket if r["judge_verdict"] == "CORRECT")
            metrics[f"accuracy_{diff}"] = n_correct / len(bucket)

    # ── Multi-hop ─────────────────────────────────────────────────────────────
    multi_hop = [
        r for r in answerable
        if r.get("requires_multi_hop") and r.get("judge_verdict") in ("CORRECT", "INCORRECT")
    ]
    if multi_hop:
        mh_correct = sum(1 for r in multi_hop if r["judge_verdict"] == "CORRECT")
        metrics["multi_hop_accuracy"] = mh_correct / len(multi_hop)

    # ── Per-category ──────────────────────────────────────────────────────────
    categories = {r["category"] for r in results if r["category"] != "unanswerable"}
    cat_scores = {}
    for cat in categories:
        bucket = [
            r for r in answerable
            if r["category"] == cat and r.get("judge_verdict") in ("CORRECT", "INCORRECT")
        ]
        if bucket:
            n_correct = sum(1 for r in bucket if r["judge_verdict"] == "CORRECT")
            cat_scores[cat] = {
                "accuracy": n_correct / len(bucket),
                "correct":  n_correct,
                "total":    len(bucket),
            }
    metrics["per_category"] = cat_scores

    # ── Error rate ────────────────────────────────────────────────────────────
    metrics["error_rate"] = sum(1 for r in results if r["status"] == "error") / len(results)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. RAGAS metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas(results: list):
    """
    Evaluate with RAGAS using Google Gemini as the judge LLM.
    Returns a dict of scores, or None if RAGAS is not installed.

    Install: pip install ragas datasets
    """
    try:
        from ragas import evaluate as ragas_evaluate, RunConfig
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextRecall,
            FactualCorrectness,
        )
        from ragas.llms import llm_factory
        from ragas.embeddings import embedding_factory
        from datasets import Dataset
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError:
        print("\n  RAGAS not installed — skipping.")
        print("  Install with: pip install ragas datasets")
        return None

    # Filter: only answerable, successful results
    valid = [
        r for r in results
        if r["status"] == "success" and r["ground_truth"] != "NOT_IN_DOCUMENTS"
    ]
    if not valid:
        print("  No valid results for RAGAS.")
        return None

    print(f"\n  Running RAGAS on {len(valid)} questions (Gemini as judge, sequential)...")
    print(f"  This takes ~{len(valid) * 4 // 60 + 1} min at free-tier rate limits — please wait.\n")

    api_key = os.getenv("GOOGLE_API_KEY")

    # Use llm_factory (non-deprecated) with Gemini
    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=api_key)
        judge_llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
    except Exception:
        # fallback: wrap via langchain
        from ragas.llms import LangchainLLMWrapper
        judge_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.0)
        )

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        judge_embeddings = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        )
    except Exception:
        judge_embeddings = None

    dataset = Dataset.from_dict({
        "user_input":         [r["question"]     for r in valid],
        "response":           [r["answer"]       for r in valid],
        "retrieved_contexts": [r["contexts"]     for r in valid],
        "reference":          [r["ground_truth"] for r in valid],
    })

    metric_kwargs = {"llm": judge_llm}
    metrics = [
        Faithfulness(**metric_kwargs),
        AnswerRelevancy(**{**metric_kwargs, **({"embeddings": judge_embeddings} if judge_embeddings else {})}),
        LLMContextRecall(**metric_kwargs),
        FactualCorrectness(**metric_kwargs),
    ]

    # RunConfig: max_workers=1 forces sequential execution, preventing rate-limit timeouts
    run_cfg = RunConfig(
        max_workers=1,       # sequential — no parallel API bursts
        timeout=180,         # 3 min per individual call
        max_retries=3,
    )

    try:
        ragas_result = ragas_evaluate(dataset=dataset, metrics=metrics, run_config=run_cfg)
        scores = ragas_result.to_pandas().mean(numeric_only=True).to_dict()
        print("\n  RAGAS evaluation complete.")
        return scores
    except Exception as e:
        print(f"\n  RAGAS evaluation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Logging & reporting
# ─────────────────────────────────────────────────────────────────────────────

def log_csv(custom: dict, ragas: dict | None) -> None:
    os.makedirs(os.path.dirname(Config.EVAL_LOG), exist_ok=True)
    file_exists = os.path.isfile(Config.EVAL_LOG)

    row = {
        "timestamp":             datetime.now().isoformat(timespec="seconds"),
        "version":               Config.VERSION,
        "description":           Config.DESCRIPTION,
        # chunking
        "chunk_size":            Config.CHUNK_SIZE,
        "chunk_overlap":         Config.CHUNK_OVERLAP,
        "chunking_strategy":     Config.CHUNKING_STRATEGY,
        # embedding
        "embedding_model":       Config.EMBEDDING_MODEL,
        "embedding_dims":        Config.EMBEDDING_DIMENSIONS,
        "contextual_embed":      Config.USE_CONTEXTUAL_EMBEDDINGS,
        # retrieval
        "top_k":                 Config.TOP_K,
        "retrieval_type":        Config.RETRIEVAL_TYPE,
        "reranker":              Config.USE_RERANKER,
        "multi_query":           Config.USE_MULTI_QUERY,
        # llm
        "llm_model":             Config.active_llm_model(),
        "llm_temperature":       Config.LLM_TEMPERATURE,
        # custom metrics
        "answer_correctness":    f"{custom.get('answer_correctness', ''):.3f}" if custom.get('answer_correctness') != '' else '',
        "evaluated_questions":   custom.get("evaluated_questions", ""),
        "judge_error_skipped":   custom.get("judge_error_skipped", ""),
        "idk_accuracy":          f"{custom.get('idk_accuracy', ''):.3f}"       if custom.get('idk_accuracy') != '' else '',
        "hallucination_rate":    f"{custom.get('hallucination_rate', ''):.3f}" if custom.get('hallucination_rate') != '' else '',
        "accuracy_easy":         f"{custom.get('accuracy_easy', ''):.3f}"      if custom.get('accuracy_easy') != '' else '',
        "accuracy_medium":       f"{custom.get('accuracy_medium', ''):.3f}"    if custom.get('accuracy_medium') != '' else '',
        "accuracy_hard":         f"{custom.get('accuracy_hard', ''):.3f}"      if custom.get('accuracy_hard') != '' else '',
        "multi_hop_accuracy":    f"{custom.get('multi_hop_accuracy', ''):.3f}" if custom.get('multi_hop_accuracy') != '' else '',
        "latency_p50_ms":        custom.get("latency_p50_ms", ""),
        "latency_p95_ms":        custom.get("latency_p95_ms", ""),
        "latency_avg_ms":        custom.get("latency_avg_ms", ""),
        "error_rate":            f"{custom.get('error_rate', 0):.3f}",
    }

    # RAGAS columns
    ragas_keys = ["faithfulness", "answer_relevancy", "context_recall", "factual_correctness"]
    for k in ragas_keys:
        row[f"ragas_{k}"] = f"{ragas[k]:.3f}" if ragas and k in ragas else ""

    with open(Config.EVAL_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n  Appended to {Config.EVAL_LOG}")


def save_details(results: list, custom: dict, ragas: dict | None) -> None:
    os.makedirs(Config.EVAL_DETAILS_DIR, exist_ok=True)
    path = os.path.join(Config.EVAL_DETAILS_DIR, f"{Config.VERSION}.json")

    output = {
        "version":   Config.VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config":    Config.to_dict(),
        "summary": {
            "custom_metrics": {k: v for k, v in custom.items() if k != "per_category"},
            "per_category":   custom.get("per_category", {}),
            "ragas_metrics":  ragas,
        },
        "results": [
            {
                "question_id":     r["question_id"],
                "question":        r["question"],
                "ground_truth":    r["ground_truth"],
                "answer":          r.get("answer", ""),
                "category":        r["category"],
                "difficulty":      r.get("difficulty"),
                "requires_multi_hop": r.get("requires_multi_hop"),
                "sources_retrieved": r.get("sources", []),
                "judge_verdict":   r.get("judge_verdict", ""),
                "judge_reasoning": r.get("judge_reasoning", ""),
                "retrieval_ms":    r.get("retrieval_ms"),
                "generation_ms":   r.get("generation_ms"),
                "total_ms":        r.get("total_ms"),
                "status":          r.get("status"),
            }
            for r in results
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Detailed results saved to {path}")


def save_error_log(results: list) -> None:
    """
    Write errors_<version>.json with every question that could not be evaluated:
      - pipeline errors  (status == "error"): retrieval or generation failed
      - judge errors     (judge_verdict == "ERROR"): judge LLM failed after retries
    Each entry includes the bot answer (or error message), the ground truth,
    retrieved sources, and the failure reason.
    """
    error_entries = []

    for r in results:
        reason = None
        if r.get("status") == "error":
            reason = "pipeline_error"
        elif r.get("judge_verdict") == "ERROR":
            reason = "judge_error"

        if reason is None:
            continue

        error_entries.append({
            "question_id":      r["question_id"],
            "question":         r["question"],
            "failure_reason":   reason,
            "bot_answer":       r.get("answer", ""),
            "ground_truth":     r["ground_truth"],
            "sources_retrieved": r.get("sources", []),
            "judge_reasoning":  r.get("judge_reasoning", ""),
            "retrieval_ms":     r.get("retrieval_ms"),
            "generation_ms":    r.get("generation_ms"),
            "total_ms":         r.get("total_ms"),
            "category":         r.get("category"),
            "difficulty":       r.get("difficulty"),
        })

    if not error_entries:
        print("  No errors to log.")
        return

    os.makedirs(Config.EVAL_DETAILS_DIR, exist_ok=True)
    path = os.path.join(Config.EVAL_DETAILS_DIR, f"errors_{Config.VERSION}.json")

    output = {
        "version":    Config.VERSION,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "total_errors": len(error_entries),
        "pipeline_errors": sum(1 for e in error_entries if e["failure_reason"] == "pipeline_error"),
        "judge_errors":    sum(1 for e in error_entries if e["failure_reason"] == "judge_error"),
        "errors": error_entries,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Error log saved to {path}  ({len(error_entries)} entries: "
          f"{output['pipeline_errors']} pipeline, {output['judge_errors']} judge)")


def save_incorrect_analysis(results: list) -> None:
    """
    Save a focused report of every INCORRECT answer with:
      - the question and its metadata
      - what the bot answered
      - the correct ground truth
      - every retrieved chunk (text + source) that was fed to the LLM
      - the judge's reasoning for marking it incorrect

    Output: benchmarks/details/incorrect_<VERSION>.json
    """
    incorrect = [
        r for r in results
        if r.get("judge_verdict") == "INCORRECT"
    ]

    if not incorrect:
        print("  No incorrect answers to log.")
        return

    os.makedirs(Config.EVAL_DETAILS_DIR, exist_ok=True)
    path = os.path.join(Config.EVAL_DETAILS_DIR, f"incorrect_{Config.VERSION}.json")

    entries = []
    for r in incorrect:
        # Pair each retrieved chunk with its full metadata
        contexts      = r.get("contexts", [])
        sources       = r.get("sources", [])
        chunk_ids     = r.get("chunk_ids", [""] * len(contexts))
        chunk_indexes = r.get("chunk_indexes", [""] * len(contexts))
        retrieved_chunks = [
            {
                "chunk_id":    cid,
                "chunk_index": cidx,
                "source":      src,
                "text":        ctx,
            }
            for ctx, src, cid, cidx in zip(contexts, sources, chunk_ids, chunk_indexes)
        ]

        entries.append({
            "question_id":       r["question_id"],
            "category":          r.get("category", ""),
            "difficulty":        r.get("difficulty", ""),
            "requires_multi_hop": r.get("requires_multi_hop", False),
            "question":          r["question"],
            "bot_answer":        r.get("answer", ""),
            "ground_truth":      r["ground_truth"],
            "judge_reasoning":   r.get("judge_reasoning", ""),
            "retrieved_chunks":  retrieved_chunks,
            "latency_ms":        r.get("total_ms"),
        })

    output = {
        "version":         Config.VERSION,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "total_incorrect": len(entries),
        "incorrect":       entries,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Incorrect analysis saved to {path}  ({len(entries)} entries)")


def print_report(custom: dict, ragas: dict | None) -> None:
    W = 58

    print("\n" + "═" * W)
    print(f"  EVALUATION REPORT — {Config.VERSION}")
    print(f"  {Config.DESCRIPTION[:W-2]}")
    print("═" * W)

    def v(key):
        val = custom.get(key, None)
        if val is None:
            return "n/a"
        if isinstance(val, float):
            return f"{val:.1%}"
        return f"{val} ms"

    # reprint corrected table
    print(f"\n  {'METRIC':<30} {'SCORE':>10}")
    print(f"  {'-'*30} {'-'*10}")
    # Show evaluated vs skipped counts
    ev   = custom.get("evaluated_questions", custom.get("answerable_total", "?"))
    skip = custom.get("judge_error_skipped", 0)
    tot  = custom.get("answerable_total", "?")
    un   = custom.get("unanswerable_total", "?")
    print(f"\n  Answerable Qs     : {tot}")
    print(f"  Unanswerable Qs   : {un}  (evaluated via refusal phrase check)")
    print(f"  Evaluated by judge: {ev}  (skipped due to judge error: {skip})")

    rows = [
        ("Answer Correctness",   "answer_correctness"),
        ("IDK Accuracy",         "idk_accuracy"),
        ("Hallucination Rate",   "hallucination_rate"),
        ("Error Rate",           "error_rate"),
        None,
        ("Accuracy — Easy",      "accuracy_easy"),
        ("Accuracy — Medium",    "accuracy_medium"),
        ("Accuracy — Hard",      "accuracy_hard"),
        ("Multi-hop Accuracy",   "multi_hop_accuracy"),
        None,
        ("Latency P50",          "latency_p50_ms"),
        ("Latency P95",          "latency_p95_ms"),
        ("Latency Avg",          "latency_avg_ms"),
    ]
    for row in rows:
        if row is None:
            print(f"  {'-'*30} {'-'*10}")
            continue
        label, key = row
        val = custom.get(key, None)
        if val is None:
            display = "n/a"
        elif "ms" in key:
            display = f"{val} ms"
        else:
            display = f"{val:.1%}"
        print(f"  {label:<30} {display:>10}")

    # Per-category
    per_cat = custom.get("per_category", {})
    if per_cat:
        print(f"\n  {'CATEGORY':<24} {'ACC':>7}  {'SCORE'}")
        print(f"  {'-'*24} {'-'*7}  {'-'*10}")
        for cat, s in sorted(per_cat.items()):
            bar_len = round(s["accuracy"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {cat:<24} {s['accuracy']:>6.1%}  {bar}  ({s['correct']}/{s['total']})")

    # RAGAS
    if ragas:
        ragas_keys = ["faithfulness", "answer_relevancy", "context_recall", "factual_correctness"]
        print(f"\n  {'RAGAS METRIC':<30} {'SCORE':>10}")
        print(f"  {'-'*30} {'-'*10}")
        for k in ragas_keys:
            val = ragas.get(k, None)
            display = f"{val:.3f}" if val is not None else "n/a"
            print(f"  {k.replace('_',' ').title():<30} {display:>10}")

    print(f"\n  Config: {Config.summary()}")
    print("═" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RAG Evaluation")
    p.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS evaluation")
    p.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    p.add_argument("--version", type=str, default=None, help="Override Config.VERSION")
    return p.parse_args()


def _publish(results: list, ragas_scores, partial: bool = False) -> None:
    """Compute metrics and save all output files. Safe to call on partial results."""
    if not results:
        print("\n  No results to save.")
        return

    tag = f"  [PARTIAL — {len(results)} questions]" if partial else ""
    print(f"\n  Computing custom metrics...{tag}")
    custom = compute_custom_metrics(results)

    print_report(custom, ragas_scores)
    log_csv(custom, ragas_scores)
    save_details(results, custom, ragas_scores)
    save_error_log(results)
    save_incorrect_analysis(results)

    print(f"\n  {'Partial results' if partial else 'Done'}.")
    print(f"    View log : {Config.EVAL_LOG}")
    print(f"     Details : {Config.EVAL_DETAILS_DIR}/{Config.VERSION}.json")
    print(f"      Errors : {Config.EVAL_DETAILS_DIR}/errors_{Config.VERSION}.json")
    print(f"   Incorrect : {Config.EVAL_DETAILS_DIR}/incorrect_{Config.VERSION}.json\n")


def main():
    args = parse_args()

    if args.version:
        Config.VERSION = args.version

    print("=" * 58)
    print(f"  RAG BENCHMARK — {Config.VERSION}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 58)
    print(f"\n  {Config.summary()}")

    # Load dataset
    with open(Config.TEST_DATASET, encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    if args.limit:
        questions = questions[: args.limit]
        print(f"\n  [--limit] Running {len(questions)} of {dataset['total_questions']} questions")
    else:
        print(f"\n  Loaded {len(questions)} questions from golden set")

    # Setup
    print("\n  Setting up pipeline...")
    vectorstore, retriever = setup_pipeline()

    results = []
    ragas_scores = None

    try:
        # Run questions
        results = run_all_questions(vectorstore, questions, retriever=retriever)

        # RAGAS
        if not args.skip_ragas:
            ragas_scores = run_ragas(results)
        else:
            print("\n  Skipping RAGAS (--skip-ragas)")

        _publish(results, ragas_scores, partial=False)

    except KeyboardInterrupt:
        completed = len([r for r in results if "answer" in r])
        print(f"\n\n  Interrupted — saving {completed} completed results...")
        _publish(results, ragas_scores, partial=True)


if __name__ == "__main__":
    main()
