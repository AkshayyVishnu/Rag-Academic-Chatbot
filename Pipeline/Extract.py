"""
STEP 1: Extract text from NIT Warangal PDF documents
=====================================================
RUN THIS:  python step1_extract.py
"""

import fitz  # PyMuPDF
import os

DATA_FOLDER = "data"
OUTPUT_FOLDER = "extracted"


def extract_text_from_pdf(pdf_path):
    """Extract all text from a single PDF file."""
    doc = fitz.open(pdf_path)

    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        all_text.append(text)

    doc.close()

    full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)

    return {
        "filename": os.path.basename(pdf_path),
        "text": full_text,
        "num_pages": len(all_text),
        "chars": len(full_text),
    }


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{DATA_FOLDER}/' folder!")
        print(f"Please download the NIT Warangal PDFs and place them there.")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{DATA_FOLDER}/'")
    print("=" * 60)

    all_documents = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_FOLDER, pdf_file)
        print(f"\nProcessing: {pdf_file}")

        result = extract_text_from_pdf(pdf_path)
        all_documents.append(result)

        # Save extracted text to a .txt file for inspection
        txt_filename = pdf_file.replace(".pdf", ".txt")
        txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"   Pages: {result['num_pages']}")
        print(f"   Characters extracted: {result['chars']:,}")
        print(f"   Saved to: {txt_path}")

        # Show a small preview
        preview = result["text"][:300].replace("\n", " ")
        print(f"   Preview: {preview}...")

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    total_chars = sum(d["chars"] for d in all_documents)
    total_pages = sum(d["num_pages"] for d in all_documents)
    print(f"   Total documents: {len(all_documents)}")
    print(f"   Total pages:     {total_pages}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Approx tokens:   ~{total_chars // 4:,}")

    print(f"\nAll extracted text saved to '{OUTPUT_FOLDER}/' folder.")
    print("Open the .txt files to inspect the quality of extraction!")
    print("\nNEXT STEP: Run step2_chunk.py to chunk the extracted text")


if __name__ == "__main__":
    main()
