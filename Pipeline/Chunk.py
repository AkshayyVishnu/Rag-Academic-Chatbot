"""
STEP 2: Chunk the extracted text using Recursive Character Splitting
====================================================================
RUN THIS:  python step2_chunk.py
"""

import os
import json

EXTRACTED_FOLDER = "extracted"
CHUNKS_FOLDER = "chunks"
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # characters of overlap between chunks


def recursive_character_split(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into chunks using a hierarchy of separators.
    
    Algorithm:
    1. Try to split by double newline (paragraph breaks)
    2. If a piece is still too big, split by single newline
    3. If still too big, split by sentence (period + space)
    4. If still too big, split by space (word boundary)
    5. Last resort: split by character
    """

    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(text, separators):
        """Recursively split text using the separator hierarchy."""

        if len(text) <= chunk_size:
            return [text]

        # Find the best separator (first one that exists in the text)
        separator = separators[-1]
        for sep in separators:
            if sep in text:
                separator = sep
                break

        # Split the text using this separator
        if separator:
            pieces = text.split(separator)
        else:
            pieces = list(text)

        # Merge pieces into chunks of appropriate size
        chunks = []
        current_chunk = ""

        for piece in pieces:
            test_chunk = current_chunk + (separator if current_chunk else "") + piece

            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(piece) > chunk_size:
                    remaining_seps = separators[separators.index(separator) + 1 :]
                    if remaining_seps:
                        sub_chunks = split_text(piece, remaining_seps)
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1] if sub_chunks else ""
                    else:
                        chunks.append(piece[:chunk_size])
                        current_chunk = piece[chunk_size:]
                else:
                    current_chunk = piece

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # Get the initial chunks
    raw_chunks = split_text(text, separators)

    # Add overlap between consecutive chunks
    overlapped_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
        else:
            prev_chunk = raw_chunks[i - 1]
            overlap_text = (
                prev_chunk[-chunk_overlap:]
                if len(prev_chunk) > chunk_overlap
                else prev_chunk
            )
            overlapped_chunk = overlap_text + " " + chunk
            overlapped_chunks.append(overlapped_chunk)

    return overlapped_chunks


def create_chunks_with_metadata(filename, text, chunk_size=1000, chunk_overlap=200):
    """
    Create chunks AND attach metadata to each chunk.
    
    Each chunk gets:
    - source: which document it came from
    - chunk_id: unique identifier
    - chunk_index: position in the document
    """

    chunks = recursive_character_split(text, chunk_size, chunk_overlap)

    chunks_with_metadata = []

    for i, chunk_text in enumerate(chunks):
        chunk_text = chunk_text.strip()

        # Skip empty or very short chunks (likely noise)
        if len(chunk_text) < 50:
            continue

        chunk_data = {
            "chunk_id": f"{filename}_chunk_{i:04d}",
            "source": filename,
            "chunk_index": i,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "token_count_approx": len(chunk_text) // 4,
        }
        chunks_with_metadata.append(chunk_data)

    return chunks_with_metadata


def main():
    os.makedirs(CHUNKS_FOLDER, exist_ok=True)

    txt_files = [f for f in os.listdir(EXTRACTED_FOLDER) if f.endswith(".txt")]

    if not txt_files:
        print(f"No .txt files found in '{EXTRACTED_FOLDER}/'!")
        print(f"Run step1_extract.py first.")
        return

    print(f"Found {len(txt_files)} extracted text file(s)")
    print(f"Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    print("=" * 60)

    all_chunks = []

    for txt_file in txt_files:
        txt_path = os.path.join(EXTRACTED_FOLDER, txt_file)

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"\nChunking: {txt_file}")
        print(f"   Input: {len(text):,} characters")

        chunks = create_chunks_with_metadata(
            filename=txt_file.replace(".txt", ""),
            text=text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        all_chunks.extend(chunks)

        print(f"   Created {len(chunks)} chunks")

        sizes = [c["char_count"] for c in chunks]
        if sizes:
            print(
                f"   Chunk sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}"
            )

    # Save all chunks to a JSON file
    output_path = os.path.join(CHUNKS_FOLDER, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"   Total chunks created: {len(all_chunks)}")
    print(f"   Saved to: {output_path}")

    # Show sample chunks
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS (first 3)")
    print("=" * 60)
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"\n--- Chunk {i} [{chunk['source']}] ({chunk['char_count']} chars) ---")
        preview = chunk["text"][:300]
        print(preview)
        print("...")

    print(f"\nChunking complete!")
    print(f"Open '{output_path}' in VS Code to inspect all chunks")
    print(f"\nNEXT STEP: We'll create embeddings from these chunks")


if __name__ == "__main__":
    main()
