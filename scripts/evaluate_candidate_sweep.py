from local_rag.config import (
    RETRIEVAL_CANDIDATES,
    TOP_K,
)

from local_rag.embeddings.embedder import Embedder
from local_rag.retrieval.retriever import Retriever
from local_rag.retrieval.reranker import Reranker
from local_rag.evaluation.metrics import (
    calculate_recall,
    calculate_mrr,
)
from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)

EVALUATION_DATASET = [
    {
        "id": "q1",
        "question": "What are metamaterials?",
        "relevant_pages": [2, 4, 7],
    },
    {
        "id": "q2",
        "question": "What are the aerospace applications of metamaterials?",
        "relevant_pages": [2, 8, 10],
    },
    {
        "id": "q3",
        "question": "How does additive manufacturing contribute to metamaterial development?",
        "relevant_pages": [2, 10, 17, 21],
    },
    {
        "id": "q4",
        "question": "What is inverse design in metamaterials?",
        "relevant_pages": [7, 12, 16],
    },
    {
        "id": "q5",
        "question": "What challenges limit practical metamaterial deployment?",
        "relevant_pages": [5, 16, 17, 21],
    },
]


CANDIDATE_VALUES = [
    4,
    8,
    12,
    16,
    20,
    30,
]


def evaluate_candidate_count(
    retriever,
    reranker,
    candidate_count,
):

    before_scores = []
    after_scores = []

    for item in EVALUATION_DATASET:

        question = item["question"]
        relevant_pages = item["relevant_pages"]

        print(
            f"\nQuestion: {question}"
        )

        results = retriever.retrieve(
            query=question,
            top_k=TOP_K,
            candidate_limit=candidate_count,
        )

        retrieved_pages = [
            r.page_number
            for r in results
        ]

        before_recall = calculate_recall(
            retrieved_pages,
            relevant_pages,
        )

        before_mrr = calculate_mrr(
            retrieved_pages,
            relevant_pages,
        )


        reranked = reranker.rerank(
            question,
            results,
        )


        reranked_pages = [
            r.page_number
            for r in reranked
        ]

        after_recall = calculate_recall(
            reranked_pages,
            relevant_pages,
        )

        after_mrr = calculate_mrr(
            reranked_pages,
            relevant_pages,
        )


        before_scores.append(
            {
                "recall": before_recall,
                "mrr": before_mrr,
            }
        )

        after_scores.append(
            {
                "recall": after_recall,
                "mrr": after_mrr,
            }
        )


    return {
        "candidate_count": candidate_count,

        "before_recall": average(
            x["recall"]
            for x in before_scores
        ),

        "after_recall": average(
            x["recall"]
            for x in after_scores
        ),

        "before_mrr": average(
            x["mrr"]
            for x in before_scores
        ),

        "after_mrr": average(
            x["mrr"]
            for x in after_scores
        ),
    }


def average(values):

    values = list(values)

    if not values:
        return 0

    return sum(values) / len(values)



def main():

    print(
        "Loading embedding model..."
    )

    embedder = Embedder()

    print(
    "Loading vector store..."
    )

    vector_store = QdrantVectorStore(
        vector_size=embedder.dimension,
    )


    print(
        "Loading retriever..."
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    print(
        "Loading reranker..."
    )

    reranker = Reranker()


    results = []

    for candidate_count in CANDIDATE_VALUES:

        print(
            "\n"
            "=" * 60
        )

        print(
            f"Evaluating candidates={candidate_count}"
        )

        result = evaluate_candidate_count(
            retriever,
            reranker,
            candidate_count,
        )

        results.append(result)


    print(
        "\n"
        "=" * 80
    )

    print(
        "CANDIDATE SWEEP RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Candidates':<12}"
        f"{'Recall Before':<18}"
        f"{'Recall After':<18}"
        f"{'MRR Before':<15}"
        f"{'MRR After':<15}"
    )

    print(
        "-" * 80
    )

    for r in results:

        print(
            f"{r['candidate_count']:<12}"
            f"{r['before_recall']:<18.3f}"
            f"{r['after_recall']:<18.3f}"
            f"{r['before_mrr']:<15.3f}"
            f"{r['after_mrr']:<15.3f}"
        )


if __name__ == "__main__":
    main()