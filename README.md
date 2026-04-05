# NIT Warangal RAG Assistant

A Multimodal Hybrid RAG (Retrieval-Augmented Generation) Assistant designed for NIT Warangal students and faculty.

## 🚀 Deployment on Render

### Backend (FastAPI)
- **Runtime:** Python 3.10+
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn api_service:app --host 0.0.0.0 --port $PORT`

### Frontend (Streamlit)
- **Runtime:** Python 3.10+
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## 🛠️ Tech Stack
- **Framework:** FastAPI / Streamlit
- **LLM:** Google Gemini / Groq (Llama-3)
- **Vector Database:** ChromaDB
- **Orchestration:** LangChain
- **Processing:** PyMuPDF (fitz)

## 📁 Project Structure
- `app.py`: Streamlit frontend
- `api_service.py`: FastAPI backend
- `Pipeline/`: Core RAG logic (extraction, chunking, retrieval, chain)
- `chroma_db/`: Local vector database storage
- `Data/`: Source PDFs (Academic Regulations, Syllabi)

## ⚙️ Environment Variables
Ensure the following are set in Render's **Environment** tab:
- `GOOGLE_API_KEY`: Your Gemini API Key
- `GROQ_API_KEY`: (Optional) If using Groq as LLM provider
- `LLM_PROVIDER`: `google` or `groq`

---
For detailed setup and architectural details, see [nitw_chatbot_README.md](./nitw_chatbot_README.md).
