import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Ensure the Pipeline directory is in the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Pipeline"))

from rag_chain import (
    load_vectorstore, create_rag_chain,
    make_embeddings, make_llm,
    next_embed_key, next_llm_key,
    TOP_K, llm_keys, embed_keys,
    LLM_PROVIDER, LLM_MODEL_GOOGLE, LLM_MODEL_GROQ,
)
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="NITW RAG Engine")

# --- Global State ---
vectorstore = None
rag_prompt = None

@app.on_event("startup")
def startup_event():
    global vectorstore, rag_prompt
    print("Loading vectorstore and RAG chain...")
    vectorstore = load_vectorstore()
    rag_prompt, _ = create_rag_chain(vectorstore)
    print("RAG Engine ready.")

# --- Models ---
class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    source: str
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

# --- Helpers ---
def retrieve_docs(question: str):
    for _ in range(len(embed_keys) * 3):
        key = next_embed_key()
        try:
            vec = make_embeddings(key).embed_query(question)
            return vectorstore.similarity_search_by_vector(vec, k=TOP_K)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            raise e
    return []

def get_answer(context: str, question: str):
    for _ in range(len(llm_keys) * 3):
        key = next_llm_key()
        chain = rag_prompt | make_llm(key) | StrOutputParser()
        try:
            return chain.invoke({"context": context, "question": question})
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            raise e
    return "I'm sorry, I'm currently hitting rate limits. Please try again in a moment."

# --- Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def query_assistant(request: QueryRequest):
    try:
        docs = retrieve_docs(request.question)
        context = "\n\n".join(
            f"--- Document {i} (Source: {doc.metadata.get('source', 'Unknown')}) ---\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
        )
        
        answer = get_answer(context, request.question)
        
        sources = [
            Source(
                source=doc.metadata.get("source", "Unknown").replace("\\", "/").split("/")[-1],
                excerpt=doc.page_content[:200].replace("\n", " ").strip() + "..."
            )
            for doc in docs
        ]
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "vectorstore_loaded": vectorstore is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
