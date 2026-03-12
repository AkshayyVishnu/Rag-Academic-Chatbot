
import os
import sys
import time
import itertools
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def load_api_keys(prefix):
    keys = []
    base = os.getenv(prefix)
    if base:
        keys.append(base)
    i = 2
    while True:
        k = os.getenv(f"{prefix}{i}")
        if not k:
            break
        keys.append(k)
        i += 1
    return keys


# --- LLM provider selection ---
# Set LLM_PROVIDER=groq in .env to use Groq instead of Gemini for the LLM.
# Embeddings always use Google (Gemini embedding model).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

embed_keys = load_api_keys("GOOGLE_API_KEY")
if not embed_keys:
    print("ERROR: GOOGLE_API_KEY not found!")
    exit(1)

if LLM_PROVIDER == "groq":
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        print("ERROR: langchain-groq not installed. Run: pip install langchain-groq")
        exit(1)
    llm_keys = load_api_keys("GROQ_API_KEY")
    if not llm_keys:
        print("ERROR: LLM_PROVIDER=groq but GROQ_API_KEY not found in .env")
        exit(1)
    print(f"LLM provider: Groq ({len(llm_keys)} key(s))")
else:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm_keys = embed_keys  # same keys for both
    print(f"LLM provider: Google Gemini ({len(llm_keys)} key(s))")

print(f"Embed keys: {len(embed_keys)}")

_embed_key_cycle = itertools.cycle(embed_keys)
_llm_key_cycle   = itertools.cycle(llm_keys)


def next_embed_key():
    return next(_embed_key_cycle)


def next_llm_key():
    return next(_llm_key_cycle)


try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
COLLECTION_NAME = "nitw_documents"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
LLM_MODEL_GOOGLE = "gemini-2.0-flash"
LLM_MODEL_GROQ   = "llama-3.3-70b-versatile"   # fast, free on Groq
LLM_TEMPERATURE  = 0.1
TOP_K            = 4
RPM_PER_KEY      = 30 if LLM_PROVIDER == "groq" else 15


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


def make_embeddings(key):
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=key,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )


def make_llm(key):
    if LLM_PROVIDER == "groq":
        return ChatGroq(
            model=LLM_MODEL_GROQ,
            groq_api_key=key,
            temperature=LLM_TEMPERATURE,
            max_retries=0,
        )
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_GOOGLE,
        google_api_key=key,
        temperature=LLM_TEMPERATURE,
        max_retries=0,
    )


def load_vectorstore():
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"ERROR: ChromaDB not found at '{CHROMA_DB_DIR}/'")
        exit(1)

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=make_embeddings(next_embed_key()),
        collection_name=COLLECTION_NAME,
    )

    count = vectorstore._collection.count()
    print(f"Loaded ChromaDB with {count} embedded chunks")
    return vectorstore


def create_rag_chain(vectorstore):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    # llm_chain has no LLM baked in — we inject a fresh one (with rotated key) per call
    return prompt, vectorstore


def ask_with_sources(prompt, retrieved_docs, question, max_retries=None, min_delay=None):
    """Ask a question — rotates to next LLM key on every 429 instead of waiting."""
    max_retries = max_retries or len(llm_keys) * 3
    # Minimum seconds between requests: safe rate across all keys combined
    if min_delay is None:
        min_delay = 60 / (RPM_PER_KEY * len(llm_keys))  # try each key a few times

    print(f"\nQuestion: {question}")
    print("-" * 50)

    formatted_context = "\n\n".join(
        f"--- Document {i} (Source: {doc.metadata.get('source', 'Unknown')}) ---\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs, 1)
    )

    t_start = time.time()

    for attempt in range(max_retries):
        key = next_llm_key()
        llm_chain = prompt | make_llm(key) | StrOutputParser()
        try:
            answer = llm_chain.invoke({"context": formatted_context, "question": question})
            # Throttle: sleep any remaining time to stay under combined RPM budget
            elapsed = time.time() - t_start
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
            print(f"\nAnswer:\n{answer}")
            print(f"\n--- Sources Used ({len(retrieved_docs)} chunks) ---")
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                preview = doc.page_content[:150].replace("\n", " ")
                print(f"  {i}. [{source}] {preview}...")
            return answer
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if len(llm_keys) > 1:
                    print(f"  Rate limit on key ...{key[-6:]} → rotating to next key (attempt {attempt+1})")
                else:
                    wait = 60 * (2 ** min(attempt, 4))
                    print(f"  Rate limit hit (only 1 LLM key). Waiting {wait}s (attempt {attempt+1})...")
                    time.sleep(wait)
            else:
                print(f"\nError: {e}")
                return None

    print("All keys exhausted / max retries reached. Skipping this question.")
    return None


def batch_retrieve(questions, vectorstore, max_retries=None):
    """Embed all questions in ONE batch call — rotates embed key on 429."""
    max_retries = max_retries or len(embed_keys) * 3

    print(f"\nBatch-embedding {len(questions)} queries in a single API call...")

    for attempt in range(max_retries):
        key = next_embed_key()
        try:
            query_vectors = make_embeddings(key).embed_documents(questions)
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if len(embed_keys) > 1:
                    print(f"  Rate limit on embed key ...{key[-6:]} → rotating (attempt {attempt+1})")
                else:
                    wait = 60 * (2 ** min(attempt, 4))
                    print(f"  Rate limit hit (only 1 embed key). Waiting {wait}s (attempt {attempt+1})...")
                    time.sleep(wait)
            else:
                print(f"  Embedding error: {e}")
                return [[] for _ in questions]
    else:
        print("  All embed keys exhausted. Returning empty results.")
        return [[] for _ in questions]

    all_docs = []
    for vec in query_vectors:
        docs = vectorstore.similarity_search_by_vector(vec, k=TOP_K)
        all_docs.append(docs)

    print(f"  Retrieved {TOP_K} chunks for each of {len(questions)} queries.")
    return all_docs


def interactive_mode(prompt, vectorstore):
    """Interactive Q&A — rotates keys on rate limits."""
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
            docs_list = batch_retrieve([question], vectorstore)
            ask_with_sources(prompt, docs_list[0], question)
        except Exception as e:
            print(f"\nError: {e}")
            print("This might be a rate limit issue. Wait a few seconds and try again.")


def run_test_queries(prompt, vectorstore):
    """Run pre-defined test queries — all queries embedded in ONE batch API call."""

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

    # Single batch embed call for all questions
    all_docs = batch_retrieve(test_questions, vectorstore)

    for question, retrieved_docs in zip(test_questions, all_docs):
        print("\n" + "=" * 60)
        ask_with_sources(prompt, retrieved_docs, question)
        print()


def main():
    print("=" * 60)
    print("STEP 4: RAG CHAIN — NIT Warangal Q&A")
    print("=" * 60)
    
    # Load vector store
    vectorstore = load_vectorstore()
    
    # Create RAG chain
    print(f"\nCreating RAG chain...")
    llm_model = LLM_MODEL_GROQ if LLM_PROVIDER == "groq" else LLM_MODEL_GOOGLE
    print(f"  LLM: {llm_model} via {LLM_PROVIDER} (temperature={LLM_TEMPERATURE})")
    print(f"  Retrieval: top-{TOP_K} chunks per query")
    prompt, vectorstore = create_rag_chain(vectorstore)
    print("  RAG chain ready!\n")

    # Ask user what to do
    print("Choose mode:")
    print("  1. Run test queries (automated)")
    print("  2. Interactive mode (ask your own questions)")

    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        run_test_queries(prompt, vectorstore)
    elif choice == "2":
        interactive_mode(prompt, vectorstore)
    else:
        print("Invalid choice. Running test queries by default.")
        run_test_queries(prompt, vectorstore)
    
if __name__ == "__main__":
    main()