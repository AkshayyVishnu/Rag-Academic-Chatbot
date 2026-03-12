
import os
import sys
import json
import time
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not found!")
    exit(1)

CHUNKS_FILE = os.path.join("chunks", "all_chunks.json")
CHROMA_DB_DIR = "chroma_db"          # where ChromaDB stores its data
COLLECTION_NAME = "nitw_documents"   # name of the collection in ChromaDB
EMBEDDING_MODEL = "models/gemini-embedding-001"  # Gemini's latest embedding model
EMBEDDING_DIMENSIONS = 768  # Default is 3072, but 768 for
BATCH_SIZE = 20  # Process chunks in batches to avoid rate limits


def load_chunks():
    
    if not os.path.exists(CHUNKS_FILE):
        print(f"ERROR: '{CHUNKS_FILE}' not found!")
        print("Run step2_chunk.py first.")
        exit(1)
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks from '{CHUNKS_FILE}'")
    return chunks


def chunks_to_langchain_documents(chunks):
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
            }
        )
        documents.append(doc)
    
    print(f"Converted {len(documents)} chunks to LangChain Documents")
    return documents


def create_embeddings_and_store(documents):
    

    
    print(f"\nInitializing Gemini embedding model: {EMBEDDING_MODEL}")
    print(f"Output dimensions: {EMBEDDING_DIMENSIONS} (default is 3072, we use 768 for efficiency)")
 
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )
    
    # Test the embedding model with a simple text
    print("\nTesting embedding model...")
    test_vector = embeddings.embed_query("test query")
    print(f"   Test embedding dimensions: {len(test_vector)}")
    print(f"   First 5 values: {test_vector[:5]}")
    
    # CRITICAL CHECK: Verify we got the right dimensions
    if len(test_vector) != EMBEDDING_DIMENSIONS:
        print(f"\n   WARNING: Expected {EMBEDDING_DIMENSIONS} dimensions but got {len(test_vector)}!")
        print(f"   This means output_dimensionality in constructor was ignored.")
        print(f"   Your langchain-google-genai version may be outdated.")
        print(f"   Try: pip install --upgrade langchain-google-genai")
        print(f"\n   Continuing with {len(test_vector)} dimensions (this still works, just uses more storage).")
    else:
        print(f"   Dimensions match expected {EMBEDDING_DIMENSIONS}!")
    
    print("   Embedding model is working!\n")
    
    # Process in batches to avoid rate limits
    print(f"Embedding {len(documents)} chunks in batches of {BATCH_SIZE}...")
    print(f"Storing in ChromaDB at '{CHROMA_DB_DIR}/'")
    print("This may take a few minutes on first run.\n")
    
    # For the first batch, use from_documents (creates the collection)
    # For subsequent batches, add to the existing collection
    vectorstore = None
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        if vectorstore is None:
            # First batch — create the vector store
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR,
                collection_name=COLLECTION_NAME,
            )
        else:
            # Subsequent batches — add to existing store
            vectorstore.add_documents(documents=batch)
        
        if i + BATCH_SIZE < len(documents):
            print(f"   Waiting 15 seconds to respect API rate limits...")
            time.sleep(15)
    
    print(f"\n   All {len(documents)} chunks embedded and stored!")
    print(f"   ChromaDB data saved to '{CHROMA_DB_DIR}/'")
    
    return vectorstore


def test_retrieval(vectorstore):

    print("\n" + "=" * 60)
    print("TESTING RETRIEVAL")
    print("=" * 60)
    
    test_queries = [
        "What is the minimum attendance required for end semester exams?",
        "How is CGPA calculated?",
        "Who is eligible for makeup examination?",
        "What is the minimum CGPA for Minor program?",
        "What are the rules for Ph.D. thesis submission?",
    ]
    
    for query in test_queries:
        print(f"\n{'-' * 50}")
        print(f"QUERY: {query}")
        print(f"{'-' * 50}")
        
        # Retrieve top 3 most similar chunks
        results = vectorstore.similarity_search_with_score(query, k=3)
        
        for rank, (doc, score) in enumerate(results, 1):
            print(f"\n  Result #{rank} (similarity: {score:.4f})")
            print(f"  Source: {doc.metadata['source']}")
            # Show first 200 chars of the chunk
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  Text: {preview}...")
    
    print("\n" + "=" * 60)
    print("RETRIEVAL TEST COMPLETE")
    print("=" * 60)
    print("\nLook at the results above and check:")
    print("  1. Are the retrieved chunks RELEVANT to each query?")
    print("  2. Are they from the CORRECT source document?")
    print("  3. Do the similarity scores make sense? (lower = more similar in ChromaDB)")


def main():
    print("=" * 60)
    print("STEP 3: EMBEDDINGS + CHROMADB")
    print("=" * 60)
    
    chunks = load_chunks()
    
    documents = chunks_to_langchain_documents(chunks)
    
    # Check if ChromaDB already has data (avoid re-embedding)
    if os.path.exists(CHROMA_DB_DIR):
        print(f"\nFound existing ChromaDB at '{CHROMA_DB_DIR}/'")
        print("Loading existing embeddings (delete the folder to re-embed)")
        
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
        
        # Check how many documents are stored
        count = vectorstore._collection.count()
        print(f"ChromaDB contains {count} embedded chunks")
        
        if count < len(documents):
            print(f"ChromaDB is incomplete ({count}/{len(documents)})! Re-embedding all chunks...")
            import shutil
            shutil.rmtree(CHROMA_DB_DIR)
            vectorstore = create_embeddings_and_store(documents)
    else:
        # First time — create embeddings and store
        vectorstore = create_embeddings_and_store(documents)
    
    # Test retrieval
    test_retrieval(vectorstore)
    
    # Summary
    print("\n" + "=" * 60)
    print("DAY 2 COMPLETE!")
    print("=" * 60)
    print(f"  Chunks embedded: {len(documents)}")
    print(f"  Vector dimensions: {EMBEDDING_DIMENSIONS} (from gemini-embedding-001)")
    print(f"  Stored at: {CHROMA_DB_DIR}/")


if __name__ == "__main__":
    main()