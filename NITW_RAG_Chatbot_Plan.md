# NIT Warangal ERP RAG Chatbot — Project Plan

## Project Overview

An AI-powered chatbot that answers student queries about NIT Warangal's academic regulations, policies, syllabi, and circulars by retrieving relevant information from institutional PDF documents using Retrieval-Augmented Generation (RAG).

**Target:** Resume-ready deployed project for SDE internship applications.

---

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| PDF Extraction | PyMuPDF (from scratch) | Free |
| Text Chunking | Recursive Character Splitting (from scratch) | Free |
| Embedding Model | Gemini `text-embedding-004` | Free |
| Vector Database | ChromaDB (local) | Free |
| LLM | Gemini 2.0 Flash | Free |
| RAG Orchestration | LangChain | Free |
| Web Interface | Streamlit | Free |
| Deployment | Streamlit Cloud | Free |
| **Total** | | **₹0 (Phase 1)** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE (Offline)              │
│                                                             │
│  PDF Documents ──► PyMuPDF ──► Raw Text                     │
│                                   │                         │
│                              Recursive                      │
│                              Chunking                       │
│                            (1000 chars,                     │
│                            200 overlap)                     │
│                                   │                         │
│                          Gemini Embeddings                  │
│                       (text-embedding-004)                  │
│                                   │                         │
│                             ChromaDB                        │
│                          (Vector Store)                     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    QUERY PIPELINE (Online)                   │
│                                                             │
│  User Question ──► Embed Query ──► Similarity Search        │
│                                       │                     │
│                                  Top-K Chunks               │
│                                       │                     │
│                              Prompt + Context               │
│                                       │                     │
│                              Gemini 2.0 Flash               │
│                                       │                     │
│                           Answer + Source Citations          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    USER INTERFACE                            │
│                                                             │
│              Streamlit Web App (Deployed on Cloud)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Documents Used

| # | Document | Description |
|---|---|---|
| 1 | UG Academic Regulations (2024-25) | Grading, attendance, makeup exams, minors |
| 2 | M.Tech Academic Regulations (2024-25) | PG-specific rules, dissertation |
| 3 | Ph.D. Regulations (2024-25) | Thesis submission, DSC, comprehensive exam |
| 4 | B.Tech Civil Engineering Curriculum | Course content, syllabi |

Source: `https://www.nitw.ac.in/`

---

## Phase 1: Ship It (Days 1-7) — MVP

**Goal:** Working chatbot deployed with a live URL.

### Day 1: PDF Extraction + Chunking

**Files:** `step1_extract.py`, `step2_chunk.py`

- [ ] Create project folder structure
  ```
  nitw-rag-chatbot/
  ├── data/              ← PDFs go here
  ├── extracted/         ← generated .txt files
  ├── chunks/            ← generated JSON chunks
  ├── step1_extract.py
  ├── step2_chunk.py
  └── requirements.txt
  ```
- [ ] Download 4 NIT Warangal PDFs into `data/`
- [ ] Install dependencies: `pip install pymupdf`
- [ ] Run `step1_extract.py` — verify extracted text is clean
- [ ] Run `step2_chunk.py` — inspect chunks in `all_chunks.json`
- [ ] Get Gemini API key from `https://aistudio.google.com/apikey`

**Verification:**
- Open `.txt` files — text should be readable, no garbled characters
- Open `all_chunks.json` — chunks should be coherent, ~1000 chars each
- Note total chunk count for reference

---

### Day 2: Embeddings + Vector Database

**Files:** `step3_embed_and_store.py`

- [ ] Install: `pip install langchain langchain-google-genai chromadb`
- [ ] Set Gemini API key as environment variable
- [ ] Load chunks from `all_chunks.json`
- [ ] Generate embeddings using Gemini `text-embedding-004`
- [ ] Store embeddings + metadata in ChromaDB (persistent local storage)
- [ ] Test: run a sample similarity search query and verify results

**Key Decisions:**
- Embedding model: Gemini `text-embedding-004` (768 dimensions, free)
- Vector DB: ChromaDB (local, no server needed, beginner-friendly)
- Distance metric: Cosine similarity (default, works well for text)

**Verification:**
- Query "minimum attendance requirement" → should return chunks about attendance
- Query "makeup examination eligibility" → should return chunks about makeup exams

---

### Day 3: LLM + RAG Chain

**Files:** `step4_rag_chain.py`

- [ ] Build LangChain RAG chain: retriever → prompt → Gemini LLM
- [ ] Write system prompt with instructions:
  - Answer ONLY from provided context
  - Cite the source document
  - Say "I don't have information about this" if context doesn't contain the answer
- [ ] Test with 10+ questions in terminal
- [ ] Tune `k` (number of retrieved chunks) — start with k=4

**Key Decisions:**
- LLM: Gemini 2.0 Flash (fast, free, good at instruction following)
- Top-K: 4 chunks retrieved per query (balance between context and noise)
- Temperature: 0.1 (low = more factual, less creative)

**Verification:**
- "What is the minimum attendance?" → correct answer with source
- "Who is the Prime Minister?" → "I don't have information about this"
- "How is CGPA calculated?" → correct formula with source

---

### Day 4: Streamlit Web Interface

**Files:** `app.py`

- [ ] Install: `pip install streamlit`
- [ ] Build chat interface with:
  - Text input for questions
  - Chat history (conversation style)
  - Source citations displayed below each answer
  - Sidebar with project info
- [ ] Style it to look clean and professional
- [ ] Test full flow: question → retrieval → answer → citation

**Verification:**
- App runs locally with `streamlit run app.py`
- Conversation history persists during session
- Sources are clearly displayed

---

### Day 5: Prompt Tuning + Edge Cases

**Files:** Modify `step4_rag_chain.py`, `app.py`

- [ ] Test with 20+ real student questions
- [ ] Fix cases where the chatbot:
  - Hallucinates (makes up information not in the documents)
  - Gives vague answers when specific info is available
  - Fails to say "I don't know" when it should
- [ ] Improve system prompt based on failure patterns
- [ ] Add error handling (API failures, empty responses)

**Test Questions Bank:**
```
1. What is the minimum attendance to appear for end semester exams?
2. How is SGPA calculated?
3. What is the eligibility for makeup examination?
4. What is the minimum CGPA for Minor program registration?
5. How many times can a student appear for makeup exam?
6. What happens if CGPA falls below 4.0?
7. What are the rules for SWAYAM-NPTEL credit transfer?
8. What is the maximum duration to complete B.Tech?
9. What is the grading system used? (relative vs absolute)
10. What are the rules for semester break?
11. What is the process for internship credit?
12. What courses are in B.Tech Civil 1st semester?
13. Can you explain the rules about academic dishonesty?
14. What is the fee for duplicate grade sheet?
15. What are the Ph.D. admission categories?
```

---

### Day 6: GitHub Repository + README

**Files:** `README.md`, architecture diagram, `.gitignore`

- [ ] Initialize git repo
- [ ] Create comprehensive README with:
  - Project description
  - Architecture diagram (use the ASCII diagram above or create one on draw.io)
  - Tech stack with justifications
  - Setup instructions (step by step)
  - Screenshots of the working chatbot
  - Sample Q&A demonstrations
  - Future improvements section (Phase 2 features)
- [ ] Add `.gitignore` (exclude API keys, __pycache__, chromadb data)
- [ ] Add `requirements.txt` with all dependencies
- [ ] Push to GitHub

**README Structure:**
```
# NIT Warangal ERP AI Chatbot

## Demo
[Link to live app] | [Screenshot]

## Architecture
[Diagram]

## Tech Stack & Design Decisions
- Why RAG over fine-tuning
- Why Gemini over OpenAI
- Why ChromaDB
- Chunking strategy

## Setup & Installation
## Usage
## Sample Queries
## Future Roadmap (Phase 2)
## License
```

---

### Day 7: Deploy on Streamlit Cloud

- [ ] Create `requirements.txt` with pinned versions
- [ ] Add secrets management for Gemini API key
- [ ] Push final code to GitHub
- [ ] Connect Streamlit Cloud to GitHub repo
- [ ] Deploy and get live URL
- [ ] Test the live deployment with 5-10 questions
- [ ] Update README with live demo link
- [ ] **UPDATE RESUME**

**Deployment Checklist:**
```
- [ ] All file paths are relative (not absolute)
- [ ] API key is in Streamlit secrets, not hardcoded
- [ ] ChromaDB is pre-built and included (or builds on first run)
- [ ] requirements.txt is complete and tested
- [ ] App loads within 30 seconds
```

---

## Phase 2: Make It Impressive (Days 8-14)

**Goal:** Add advanced features that differentiate this from basic RAG projects.

### Day 8-9: Hybrid Search (BM25 + Semantic + RRF)

**What:** Add keyword-based search (BM25) alongside semantic search, combine results using Reciprocal Rank Fusion.

**Why:** +20-30% retrieval accuracy for free. Handles cases where:
- Semantic search misses exact keyword matches (e.g., "DAC-UG")
- BM25 misses semantically similar but differently worded queries

**Implementation:**
- [ ] Install `rank-bm25` library
- [ ] Build BM25 index from all chunks
- [ ] For each query: run both BM25 and ChromaDB semantic search
- [ ] Fuse results using Reciprocal Rank Fusion (RRF)
- [ ] Return top-K fused results to the LLM

**RRF Formula:**
```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
where k = 60 (standard constant), rank_i = rank in each retrieval method
```

**Cost:** Free
**Complexity:** Medium (2-3 hours)

---

### Day 9-10: Multimodal — Tables & Images (Unstructured Library)

**What:** Replace basic PyMuPDF extraction with the `unstructured` library to properly handle tables and images in PDFs.

**Why:** NIT Warangal regulations contain important tables (grading scales, fee structures, course credits) that basic text extraction mangles.

**Implementation:**
- [ ] Install `unstructured[pdf]`
- [ ] Replace `step1_extract.py` with Unstructured-based extraction
- [ ] Tables extracted as structured elements (not garbled text)
- [ ] Images flagged with placeholders for future handling
- [ ] Re-run chunking and embedding pipeline

**Cost:** Free
**Complexity:** Medium (3-4 hours)

---

### Day 11: Contextual Embeddings

**What:** Before embedding each chunk, use Gemini to add context about what the chunk is about and which document/section it comes from.

**Why:** +15-20% retrieval accuracy. A chunk that says "the deadline is March 15" becomes "In the UG Regulations section on Makeup Examinations, the deadline for registration is March 15" — much easier for semantic search to match.

**Implementation:**
- [ ] For each chunk, call Gemini with:
  ```
  "Given this document: {doc_title}, and this section: {section},
   provide a brief 1-2 sentence context for this chunk: {chunk_text}"
  ```
- [ ] Prepend the context to the chunk before embedding
- [ ] Re-embed all chunks with contextual information
- [ ] Compare retrieval accuracy before vs after

**Cost:** ~$2-5 (one-time LLM calls during preprocessing)
**Complexity:** Medium (2-3 hours)

---

### Day 12: LLM Clarifying Questions

**What:** When a user's question is ambiguous, the LLM asks a clarifying question before answering.

**Why:** Makes the chatbot feel intelligent and reduces wrong answers.

**Examples:**
```
User: "What are the rules for registration?"
Bot:  "Could you clarify — are you asking about:
       1. Semester registration (course enrollment)
       2. Exam registration
       3. Ph.D. program registration?"

User: "What's the minimum CGPA?"
Bot:  "The minimum CGPA requirement depends on the context:
       - For Minor program: 7.0 CGPA
       - To avoid slow-paced learning: 4.0 CGPA
       Which one are you asking about?"
```

**Implementation:**
- [ ] Update system prompt to instruct LLM to identify ambiguous queries
- [ ] Add classification step: is the query clear or ambiguous?
- [ ] If ambiguous: generate clarifying options based on retrieved context
- [ ] Track conversation state to incorporate user's clarification

**Cost:** Free (prompt engineering only)
**Complexity:** Low (1-2 hours)

---

### Day 13: Enhanced Chunking (Structure-Aware)

**What:** Upgrade the chunking pipeline to detect and preserve document structure (headings, sections, headers/footers).

**Implementation:**
- [ ] Use `page.get_text("dict")` for structural extraction
- [ ] Detect headings via font size analysis
- [ ] Strip repeating headers/footers
- [ ] Insert section markers as custom chunk separators
- [ ] Attach section metadata to each chunk
- [ ] Re-run embedding pipeline

*Detailed plan saved separately — see enhanced chunking document.*

**Cost:** Free
**Complexity:** High (5-9 hours)

---

### Day 14: Final Polish + Resume Update

- [ ] Test all features end-to-end
- [ ] Update README with Phase 2 features
- [ ] Add architecture diagram v2 (showing hybrid search + multimodal)
- [ ] Add comparison: Phase 1 accuracy vs Phase 2 accuracy
- [ ] Re-deploy to Streamlit Cloud
- [ ] **Update resume with advanced features**

---

## Resume Bullet Points

### After Phase 1:
> **AI-Powered College ERP Chatbot** | Python, LangChain, Google Gemini API, ChromaDB, Streamlit
> - Developed an end-to-end RAG chatbot that processes institutional PDFs (regulations, syllabi, circulars) and answers student questions with cited sources
> - Implemented document ingestion pipeline with custom text extraction, recursive character chunking (1000-char chunks, 200-char overlap), and vector embedding storage in ChromaDB
> - Deployed as a web application using Streamlit with a conversational interface, handling queries about attendance policies, grading systems, exam regulations, and more

### After Phase 2 (Updated):
> **AI-Powered College ERP Chatbot** | Python, LangChain, Google Gemini, ChromaDB, Streamlit
> - Built a multimodal hybrid RAG chatbot that processes institutional PDFs with text, table, and image extraction using Unstructured, answering student queries with cited sources
> - Implemented hybrid search (BM25 + semantic + Reciprocal Rank Fusion) with contextual embeddings, achieving significantly higher retrieval accuracy over basic semantic search
> - Deployed as a conversational web app with LLM-driven clarifying questions, handling queries about attendance policies, grading systems, exam regulations, and more

---

## Interview Prep — Key Questions to Expect

| Question | Your Answer |
|---|---|
| Why RAG over fine-tuning? | Fine-tuning is expensive, data goes stale, hallucination risk is high. RAG retrieves fresh data per query, is cost-effective, and grounds answers in actual documents. |
| Walk me through the architecture | PDFs → extraction → chunking → embedding → ChromaDB. Query → embed → similarity search → top-K chunks → prompt + context → Gemini → answer with citations. |
| Why Gemini over OpenAI? | Free tier with generous limits, embedding model included for free, quality comparable to GPT-4o-mini for RAG Q&A. Can swap with 2 lines of code via LangChain. |
| Why ChromaDB? | Local (no cloud dependency), beginner-friendly, integrates natively with LangChain, sufficient for our document scale (~200-300 chunks). |
| What was the hardest part? | Chunking strategy — too small loses context, too large adds noise. Used recursive character splitting (1000 char, 200 overlap) as baseline, then added hybrid search for +20-30% accuracy. |
| How do you handle hallucination? | System prompt instructs LLM to answer ONLY from retrieved context. If context doesn't contain the answer, it says "I don't know." Low temperature (0.1) reduces creativity. |
| What's hybrid search / RRF? | BM25 handles exact keyword matches, semantic search handles meaning. RRF combines both ranked lists: score = Σ 1/(k + rank). Catches queries that either method alone would miss. |
| How would you scale this? | Swap ChromaDB for Pinecone/Weaviate (cloud vector DB), add authentication for student-specific data, containerize with Docker, add structured data layer (Text-to-SQL) for attendance/grades. |

---

## File Structure (Final)

```
nitw-rag-chatbot/
├── data/                          ← PDF documents
│   ├── UG_Regulations.pdf
│   ├── MTech_Regulations.pdf
│   ├── PhD_Regulations.pdf
│   └── BTech_Civil_Curriculum.pdf
│
├── extracted/                     ← extracted text (generated)
├── chunks/                        ← chunked JSON (generated)
├── chroma_db/                     ← vector database (generated)
│
├── step1_extract.py               ← PDF text extraction
├── step2_chunk.py                 ← recursive chunking
├── step3_embed_and_store.py       ← embeddings + ChromaDB
├── step4_rag_chain.py             ← LangChain RAG chain
├── app.py                         ← Streamlit web interface
│
├── requirements.txt
├── .env                           ← API keys (NOT committed)
├── .gitignore
└── README.md
```

---

## Key Links

- Gemini API Key: `https://aistudio.google.com/apikey`
- Streamlit Cloud: `https://streamlit.io/cloud`
- LangChain Docs: `https://python.langchain.com/`
- ChromaDB Docs: `https://docs.trychroma.com/`
- Project PDFs: `https://www.nitw.ac.in/`
