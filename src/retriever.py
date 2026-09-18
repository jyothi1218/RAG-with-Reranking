import numpy as np

from embeddings import model
from vector_store import build_vector_store


def retrieve(query, index, chunks, top_k=5):
    """
    Retrieve the top-k most relevant chunks for a user query.
    """

    # Convert the user's question into an embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(query_embedding).astype("float32")

    # Search FAISS
    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, index_position in zip(scores[0], indices[0]):

        if index_position == -1:
            continue

        results.append({
            "text": chunks[index_position]["text"],
            "source": chunks[index_position]["source"],
            "page": chunks[index_position]["page"],
            "score": float(score)
        })

    return results


if __name__ == "__main__":

    # Build the vector store
    index, chunks = build_vector_store()

    # Example question
    query = "What is the main purpose of the project?"

    print("\nUser Query:")
    print(query)

    # Retrieve top 5 chunks
    results = retrieve(
        query,
        index,
        chunks,
        top_k=5
    )

    print("\nTop Retrieved Results:")

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 60)

        print(f"Rank: {i}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")

        print("\nText:")
        print(result["text"][:500])