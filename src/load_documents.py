from pathlib import Path
from pypdf import PdfReader


def load_pdfs(folder_path="data/documents"):
    documents = []

    folder = Path(folder_path)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return documents

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return documents

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()

            if text and text.strip():
                documents.append({
                    "text": text.strip(),
                    "source": pdf_file.name,
                    "page": page_number + 1
                })

    print(f"Loaded {len(documents)} pages.")

    return documents


if __name__ == "__main__":
    docs = load_pdfs()

    for doc in docs[:3]:
        print("\n---")
        print("Source:", doc["source"])
        print("Page:", doc["page"])
        print(doc["text"][:500])