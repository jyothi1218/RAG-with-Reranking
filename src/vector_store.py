import faiss
import numpy as np

from load_documents import load_pdfs
from embeddings import create_chunks, create_embeddings


def build_vector_store():
    """
    Load documents, create chunks, generate embeddings,
    and build a FAISS vector index.
    """

    # Step 1: Load PDF documents
    documents = load_pdfs()

    if not documents:
        raise ValueError("No documents were found.")

    # Step 2: Create text chunks
    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError("No text chunks were created.")

    print(f"Created {len(chunks)} chunks.")

    # Step 3: Create embeddings
    embeddings = create_embeddings(chunks)

    # Convert embeddings to float32 for FAISS
    embeddings = np.asarray(embeddings).astype("float32")

    # Step 4: Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Step 5: Add embeddings to FAISS
    index.add(embeddings)

    print(f"FAISS index created.")
    print(f"Number of vectors: {index.ntotal}")
    print(f"Vector dimension: {dimension}")

    return index, chunks


if __name__ == "__main__":
    index, chunks = build_vector_store()

    print("\nVector store is ready!")

    print("\nExample stored chunk:")
    print(chunks[0]["text"][:300])