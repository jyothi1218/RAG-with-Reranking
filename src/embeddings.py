from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_chunks(documents, chunk_size=500, overlap=100):
    """
    Split extracted documents into smaller overlapping chunks.
    """

    chunks = []

    for document in documents:
        text = document["text"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "source": document["source"],
                    "page": document["page"]
                })

            start += chunk_size - overlap

    return chunks


def create_embeddings(chunks):
    """
    Convert text chunks into numerical vectors.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":
    from load_documents import load_pdfs

    documents = load_pdfs()

    chunks = create_chunks(documents)

    print(f"\nNumber of chunks: {len(chunks)}")

    embeddings = create_embeddings(chunks)

    print(f"Embedding shape: {embeddings.shape}")

    print("\nFirst chunk:")
    print(chunks[0]["text"][:300])