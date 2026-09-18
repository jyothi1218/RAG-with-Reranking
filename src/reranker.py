from sentence_transformers import CrossEncoder

from retriever import retrieve
from vector_store import build_vector_store


# Load the cross-encoder reranking model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, results, top_n=3):
    """
    Rerank retrieved documents using a cross-encoder.
    """

    # Create query-document pairs
    pairs = []

    for result in results:
        pairs.append([
            query,
            result["text"]
        ])

    # Calculate relevance scores
    scores = reranker.predict(pairs)

    # Add reranking score to each result
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    # Sort by reranking score
    reranked_results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Return only the top N results
    return reranked_results[:top_n]


if __name__ == "__main__":

    # Build the vector store
    index, chunks = build_vector_store()

    # Example question
    query = "What is the main purpose of the project?"

    print("\nUser Query:")
    print(query)

    # Stage 1: Retrieve top 5 candidates
    results = retrieve(
        query,
        index,
        chunks,
        top_k=5
    )

    print("\nBefore Reranking:")

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 60)
        print(f"Rank: {i}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Page: {result['page']}")
        print(f"Text: {result['text'][:300]}")

    # Stage 2: Rerank the candidates
    reranked_results = rerank(
        query,
        results,
        top_n=3
    )

    print("\n\nAfter Reranking:")

    for i, result in enumerate(reranked_results, start=1):

        print("\n" + "=" * 60)
        print(f"New Rank: {i}")
        print(f"Original Similarity Score: {result['score']:.4f}")
        print(f"Reranker Score: {result['rerank_score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")

        print("\nText:")
        print(result["text"][:500])