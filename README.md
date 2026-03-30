# Multimodal Hybrid RAG Assistant for College ERP

A Retrieval-Augmented Generation system that lets students and staff query college ERP data using natural language — across both text and image modalities.

Built as a personal project to solve a real problem: ERP systems are painful to navigate, and most information is buried across PDFs, tables, and image-based notices.

---

## What it does

- Accepts natural language queries about academic data (schedules, notices, results, forms)
- Retrieves relevant documents using a hybrid search pipeline
- Generates grounded, accurate answers using an LLM — with citations back to source documents
- Handles image-based documents (scanned notices, timetables) via multimodal embeddings

---

## Architecture

### Hybrid Retrieval Pipeline

The core insight is that neither keyword search nor semantic search alone works well for ERP queries:
- **BM25** (lexical) is great for exact matches — course codes, names, specific dates
- **Dense embeddings** (Gemini) are great for semantic meaning — "when is the next exam?" even if the document says "examination schedule"

Both are fused using **Reciprocal Rank Fusion (RRF)**, which combines ranked lists without needing to tune score weights.

### Reranking

After fusion, a **cross-encoder reranker** re-scores the top-k candidates. Unlike bi-encoders that embed query and document independently, cross-encoders process them jointly — giving much higher precision at the cost of speed. Used only on the shortlist to keep latency acceptable.

### Query Handling

- **Multi-query retrieval**: Each query is rephrased into multiple variants to improve recall on ambiguous questions
- **Automatic query rephrasing**: Handles poorly formed queries from non-technical users

### Evaluation Framework

Built a custom eval suite using:
- **IR metrics**: MRR (Mean Reciprocal Rank), nDCG (Normalized Discounted Cumulative Gain)
- **RAG-specific metrics**: Faithfulness (does the answer contradict the source?), Answer Relevance (does the answer actually address the question?)

This let me iterate on retrieval quality rather than just eyeballing outputs.

---

## Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| LLM + Embeddings | Gemini |
| Lexical search | BM25 (rank_bm25) |
| Vector store | Chroma |
| Reranking | Cross-encoder (sentence-transformers) |
| Language | Python |

---

## Running locally

```bash
git clone https://github.com/AkshayyVishnu/[Rag-Academic-Chatbot]

cd [repo-name]
pip install -r requirements.txt

# Add your Gemini API key
export GEMINI_API_KEY=your_key_here

python main.py
```

---

## Contact

Akshay Vishnu — akshayrkr22@gmail.com — [github.com/AkshayyVishnu](https://github.com/AkshayyVishnu)
