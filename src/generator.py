from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from reranker import rerank
from retriever import retrieve
from vector_store import build_vector_store


# Model name
MODEL_NAME = "google/flan-t5-small"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_answer(query, reranked_results):
    """
    Generate an answer using the top reranked document chunks.
    """

    # Combine the retrieved chunks into context
    context = "\n\n".join(
        result["text"]
        for result in reranked_results
    )

    # Create the prompt
    prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{query}

Answer:
"""

    # Convert prompt into model input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    # Convert model output back to text
    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()


if __name__ == "__main__":

    # ==========================================
    # STEP 1: Build vector store
    # ==========================================

    index, chunks = build_vector_store()

    # User question
    query = "What is the main purpose of the project?"

    print("\nUser Query:")
    print(query)

    # ==========================================
    # STEP 2: FAISS RETRIEVAL
    # ==========================================

    retrieved_results = retrieve(
        query,
        index,
        chunks,
        top_k=10
    )

    # ==========================================
    # STEP 3: CROSS-ENCODER RERANKING
    # ==========================================

    reranked_results = rerank(
        query,
        retrieved_results,
        top_n=3
    )

    print("\nTop Reranked Context:")

    for i, result in enumerate(reranked_results, start=1):

        print("\n" + "=" * 60)
        print(f"Rank: {i}")
        print(f"Page: {result['page']}")
        print(f"Reranker Score: {result['rerank_score']:.4f}")

        print("\nText:")
        print(result["text"][:400])

    # ==========================================
    # STEP 4: GENERATE FINAL ANSWER
    # ==========================================

    answer = generate_answer(
        query,
        reranked_results
    )

    print("\n\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)