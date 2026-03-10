
import os
import sys
import time
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not found!")
    exit(1)


try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "nitw_documents"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
LLM_MODEL = "gemini-2.0-flash"  # Free tier, fast, good for RAG
LLM_TEMPERATURE = 0.1           # Low = more factual, less creative
TOP_K = 4                       # Number of chunks to retrieve per query


SYSTEM_PROMPT = """You are an AI assistant for NIT Warangal students. Your job is to answer 
questions about academic regulations, policies, syllabi, and circulars based ONLY on the 
provided context from official NIT Warangal documents.

RULES YOU MUST FOLLOW:
1. Answer ONLY based on the provided context. Do NOT use any outside knowledge.
2. If the context does not contain enough information to answer the question, say:
   "I don't have enough information in the available documents to answer this question."
3. Always cite the source document at the end of your answer like:
   (Source: <document name>)
4. Keep answers clear, concise, and directly relevant to the question.
5. If the question is ambiguous, briefly explain what interpretations are possible
   and answer the most likely one.
6. Use bullet points for lists, but keep explanations in paragraph form.
7. If the context contains specific numbers, rules, or deadlines, include them exactly.

CONTEXT FROM NIT WARANGAL DOCUMENTS:
{context}
"""

HUMAN_PROMPT = "{question}"


def load_vectorstore():
    
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"ERROR: ChromaDB not found at '{CHROMA_DB_DIR}/'")
        exit(1)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    
    count = vectorstore._collection.count()
    print(f"Loaded ChromaDB with {count} embedded chunks")
    
    return vectorstore


def create_rag_chain(vectorstore):
 
    
    # 1. Create a retriever from the vector store
    #    k=4 means "return the 4 most similar chunks"
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    
    # 2. Create the LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_retries=2,
    )
    
    # 3. Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    
    # 4. Helper function: format retrieved documents into a single string
    def format_docs(docs):
   
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(
                f"--- Document {i} (Source: {source}) ---\n{doc.page_content}"
            )
        return "\n\n".join(formatted)
    
    # 5. Build the chain using LangChain Expression Language
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Also return retriever separately so we can show source documents
    return rag_chain, retriever


def ask_with_sources(rag_chain, retriever, question):
    """
    Ask a question and show both the answer and the source chunks.
    
    This runs the retriever and chain separately so we can display:
    1. The generated answer
    2. Which chunks were used (for transparency)
    """
    
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Get the answer
    answer = rag_chain.invoke(question)
    print(f"\nAnswer:\n{answer}")
    
    # Also show which chunks were retrieved (for debugging)
    retrieved_docs = retriever.invoke(question)
    print(f"\n--- Sources Used ({len(retrieved_docs)} chunks) ---")
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"  {i}. [{source}] {preview}...")
    
    return answer


def interactive_mode(rag_chain, retriever):
    """
    Run an interactive Q&A session in the terminal.
    Type your questions and get answers in real time.
    Type 'quit' or 'exit' to stop.
    """
    
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE — Ask anything about NIT Warangal!")
    print("=" * 60)
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        question = input("\nYou: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        try:
            ask_with_sources(rag_chain, retriever, question)
        except Exception as e:
            print(f"\nError: {e}")
            print("This might be a rate limit issue. Wait a few seconds and try again.")


def run_test_queries(rag_chain, retriever):
    """Run a set of pre-defined test queries to verify the RAG chain."""
    
    test_questions = [
        "What is the minimum attendance required to appear for end semester exams?",
        "Who is eligible for makeup examination?",
        "What is the minimum CGPA required for Minor program registration?",
        "How is SGPA calculated?",
        "What happens if a student's CGPA falls below 4.0?",
        # This question should trigger "I don't know" since it's not in the docs
        "What is the hostel fee for 2024-25?",
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING TEST QUERIES")
    print("=" * 60)
    
    for i, question in enumerate(test_questions):
        print("\n" + "=" * 60)
        ask_with_sources(rag_chain, retriever, question)
        print()
        if i < len(test_questions) - 1:
            print("Waiting 15 seconds before next query...")
            time.sleep(15)


def main():
    print("=" * 60)
    print("STEP 4: RAG CHAIN — NIT Warangal Q&A")
    print("=" * 60)
    
    # Load vector store
    vectorstore = load_vectorstore()
    
    # Create RAG chain
    print(f"\nCreating RAG chain...")
    print(f"  LLM: {LLM_MODEL} (temperature={LLM_TEMPERATURE})")
    print(f"  Retrieval: top-{TOP_K} chunks per query")
    rag_chain, retriever = create_rag_chain(vectorstore)
    print("  RAG chain ready!\n")
    
    # Ask user what to do
    print("Choose mode:")
    print("  1. Run test queries (automated)")
    print("  2. Interactive mode (ask your own questions)")

    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        run_test_queries(rag_chain, retriever)
    elif choice == "2":
        interactive_mode(rag_chain, retriever)
    else:
        print("Invalid choice. Running test queries by default.")
        run_test_queries(rag_chain, retriever)
    
if __name__ == "__main__":
    main()