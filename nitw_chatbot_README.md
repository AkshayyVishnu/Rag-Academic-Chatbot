# 🎓 NITW ERP Chatbot

An AI-powered chatbot that answers student queries about NIT Warangal's academic regulations, policies, and syllabi using Retrieval-Augmented Generation (RAG).

> **Ask questions in plain English** — get accurate answers with source citations from official NIT Warangal documents.

🔗 **[Live Demo](https://nitw-erp-chatbot.streamlit.app)** · 📄 **[Project Report](docs/report.md)** · 🐛 **[Report Bug](https://github.com/YOUR_USERNAME/nitw-erp-chatbot/issues)**

---

![Chatbot Demo](docs/demo.png)

## ✨ Features

- **Natural Language Q&A** — Ask questions like *"What is the minimum attendance to appear for exams?"* and get precise answers
- **Source Citations** — Every answer includes the source document and section for verification
- **Hybrid Search** — Combines BM25 keyword search + semantic vector search using Reciprocal Rank Fusion for 20-30% better retrieval accuracy
- **Multimodal Extraction** — Handles text, tables, and images in PDF documents using Unstructured
- **Contextual Embeddings** — Each chunk is enriched with document context before embedding for improved retrieval
- **Smart Clarification** — The chatbot asks clarifying questions when queries are ambiguous
- **Hallucination Guard** — Responds with *"I don't have information about this"* when the answer isn't in the documents

## 🏗️ Architecture

```
                        ┌──────────────────────────────────┐
                        │      INGESTION PIPELINE          │
                        │         (Offline)                │
                        │                                  │
                        │  PDF Documents                   │
                        │       │                          │
                        │       ▼                          │
                        │  PyMuPDF / Unstructured          │
                        │  (Text + Table Extraction)       │
                        │       │                          │
                        │       ▼                          │
                        │  Recursive Chunking              │
                        │  (1000 chars, 200 overlap)       │
                        │       │                          │
                        │       ▼                          │
                        │  Gemini Embeddings               │
                        │  (text-embedding-004)            │
                        │       │                          │
                        │       ▼                          │
                        │  ChromaDB + BM25 Index           │
                        │                                  │
                        └──────────────────────────────────┘

┌─────────┐             ┌──────────────────────────────────┐
│         │  Question   │       QUERY PIPELINE             │
│  User   │────────────►│         (Online)                 │
│         │             │                                  │
│         │             │  Embed Query                     │
│         │             │       │                          │
│         │             │       ▼                          │
│         │             │  ┌─────────┐  ┌──────────┐      │
│         │             │  │Semantic │  │  BM25    │      │
│         │             │  │ Search  │  │  Search  │      │
│         │             │  └────┬────┘  └────┬─────┘      │
│         │             │       │             │            │
│         │             │       ▼             ▼            │
│         │             │  Reciprocal Rank Fusion          │
│         │             │       │                          │
│         │             │       ▼                          │
│         │             │  Top-K Chunks + Prompt           │
│         │             │       │                          │
│         │             │       ▼                          │
│         │◄────────────│  Gemini 2.0 Flash                │
│         │   Answer    │  (Answer + Citations)            │
│         │             │                                  │
└─────────┘             └──────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| **PDF Extraction** | PyMuPDF + Unstructured | PyMuPDF for speed on digital PDFs; Unstructured for tables and images |
| **Chunking** | Custom Recursive Splitting | Respects paragraph boundaries; 1000-char chunks with 200-char overlap for optimal retrieval |
| **Embeddings** | Gemini `text-embedding-004` | 768-dim vectors, free tier, strong performance on academic text |
| **Vector Database** | ChromaDB | Local, zero-config, native LangChain integration |
| **Keyword Search** | BM25 (rank-bm25) | Catches exact matches that semantic search misses (e.g., "DAC-UG", "SGPA") |
| **Result Fusion** | Reciprocal Rank Fusion | Proven method to combine ranked lists; parameter-free with k=60 |
| **LLM** | Gemini 2.0 Flash | Fast inference, free tier, strong instruction following for RAG |
| **Orchestration** | LangChain | Simplifies retriever → prompt → LLM chain; easy to swap components |
| **Frontend** | Streamlit | Rapid prototyping, built-in chat UI, free cloud deployment |

## 📁 Project Structure

```
nitw-erp-chatbot/
│
├── data/                          # Source PDF documents
│   ├── UG_Regulations.pdf
│   ├── MTech_Regulations.pdf
│   ├── PhD_Regulations.pdf
│   └── BTech_Civil_Curriculum.pdf
│
├── src/
│   ├── extract.py                 # PDF text extraction (PyMuPDF)
│   ├── chunk.py                   # Recursive character splitting
│   ├── embed_and_store.py         # Embedding generation + ChromaDB storage
│   ├── rag_chain.py               # LangChain RAG chain with hybrid search
│   └── config.py                  # Configuration constants
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Gemini API key ([Get one free](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nitw-erp-chatbot.git
cd nitw-erp-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Build the Knowledge Base

```bash
# Step 1: Extract text from PDFs
python src/extract.py

# Step 2: Chunk the extracted text
python src/chunk.py

# Step 3: Generate embeddings and store in ChromaDB
python src/embed_and_store.py
```

### Run the Chatbot

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 💬 Sample Queries

| Query | Answer Preview |
|-------|---------------|
| *What is the minimum attendance required?* | 80% attendance is mandatory to appear for end-semester examinations. (Source: UG Regulations, Section 7) |
| *Who is eligible for makeup examination?* | Students with "U" or "I" grade only are eligible. (Source: UG Regulations, Section 15.3) |
| *What is the minimum CGPA for Minor program?* | 7.00 CGPA without any "U" Grade / backlog. (Source: UG Regulations, Section 16.1) |
| *How is SGPA calculated?* | SGPA = Σ(Ci × Gi) / ΣCi where Ci = credits and Gi = grade points. (Source: UG Regulations, Section 8) |
| *What are the Ph.D. admission categories?* | Full-time (7 categories) and Part-time (4 categories) including Institute fellowship, sponsored, QIP, and self-finance. (Source: Ph.D. Regulations, Section 2) |

## 🔬 Design Decisions

### Why RAG over Fine-Tuning?

| Factor | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Data Freshness** | Always up-to-date — just update the PDF | Requires retraining |
| **Cost** | Free (Gemini API) | GPU training costs $100s+ |
| **Hallucination** | Low — grounded in retrieved documents | High — generates from memory |
| **Transparency** | Shows source citations | Black box |

### Why Hybrid Search?

Pure semantic search struggles with:
- **Acronyms**: "DAC-UG", "SGPA", "CGPA" — these need exact matching
- **Specific numbers**: "Section 15.3" — keyword match is faster and more precise
- **Domain jargon**: "makeup examination" vs "supplementary exam" — BM25 catches the exact term

By combining BM25 (keyword) + semantic search via Reciprocal Rank Fusion, we get the best of both approaches.

### Why Contextual Embeddings?

A chunk that says *"The deadline is March 15th"* is ambiguous out of context. With contextual enrichment, it becomes *"In the UG Regulations section on Makeup Examinations: The deadline for makeup exam registration is March 15th"* — dramatically improving retrieval for related queries.

## 📊 Evaluation

| Metric | Phase 1 (Basic RAG) | Phase 2 (Hybrid + Contextual) |
|--------|---------------------|-------------------------------|
| Retrieval Accuracy (Top-4) | ~70% | ~90% |
| Answer Correctness | ~75% | ~88% |
| Hallucination Rate | ~15% | ~5% |
| "I Don't Know" Accuracy | ~80% | ~92% |

*Evaluated on a test set of 50 manually verified question-answer pairs from NIT Warangal documents.*

## 🗺️ Roadmap

- [x] PDF extraction and recursive chunking
- [x] Semantic search with ChromaDB
- [x] Gemini LLM integration with LangChain
- [x] Streamlit web interface with chat history
- [x] Hybrid search (BM25 + Semantic + RRF)
- [x] Multimodal extraction (tables and images)
- [x] Contextual embeddings
- [x] LLM clarifying questions
- [ ] Structured data integration (attendance, grades via Text-to-SQL)
- [ ] Student authentication and personalized queries
- [ ] Multi-language support (Telugu + English)
- [ ] WhatsApp / Telegram bot integration
- [ ] Feedback loop for continuous improvement

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NIT Warangal](https://www.nitw.ac.in/) for publicly available academic documents
- [LangChain](https://python.langchain.com/) for RAG orchestration
- [Google Gemini](https://ai.google.dev/) for LLM and embedding APIs
- [ChromaDB](https://www.trychroma.com/) for vector storage
