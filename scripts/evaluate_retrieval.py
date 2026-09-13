import json
from pathlib import Path

from local_rag.config import (
    RETRIEVAL_CANDIDATES,
    TOP_K,
)

from local_rag.embeddings.embedder import (
    Embedder,
)

from local_rag.evaluation.models import (
    RetrievalTestCase,
)

from local_rag.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)

from local_rag.retrieval.reranker import (
    Reranker,
)

from local_rag.retrieval.retriever import (
    Retriever,
)

from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)


DATASET_PATH = Path(
    "data/evaluation/retrieval_dataset.json"
)


def load_dataset() -> list[RetrievalTestCase]:

    data = json.loads(
        DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )

    return [
        RetrievalTestCase(
            id=item["id"],
            question=item["question"],
            source=item["source"],
            relevant_pages=item[
                "relevant_pages"
            ],
        )
        for item in data
    ]


def average(values: list[float]) -> float:

    if not values:
        return 0.0

    return sum(values) / len(values)


def main() -> None:

    print(
        "Loading embedding model..."
    )

    embedder = Embedder()

    vector_store = (
        QdrantVectorStore(
            vector_size=embedder.dimension
        )
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    reranker = Reranker()

    evaluator = RetrievalEvaluator()

    dataset = load_dataset()

    vector_metrics = []
    reranker_metrics = []

    for test_case in dataset:

        print(
            f"\n[{test_case.id}] "
            f"{test_case.question}"
        )

        candidates = retriever.retrieve(
            query=test_case.question,
            top_k=RETRIEVAL_CANDIDATES,
        )

        vector_results = (
            candidates[:TOP_K]
        )

        reranked_results = (
            reranker.rerank(
                query=test_case.question,
                chunks=candidates,
                top_k=TOP_K,
            )
        )

        vector_result = evaluator.evaluate(
            test_case=test_case,
            retrieved_chunks=vector_results,
        )

        reranker_result = evaluator.evaluate(
            test_case=test_case,
            retrieved_chunks=reranked_results,
        )

        vector_metrics.append(
            vector_result
        )

        reranker_metrics.append(
            reranker_result
        )

        print(
            "\nVector retrieval:"
        )

        for chunk in vector_results:
            print(
                f"  page {chunk.page_number} "
                f"| {chunk.score:.4f}"
            )

        print(
            f"Recall@{TOP_K}: "
            f"{vector_result.recall_at_k:.3f}"
        )

        print(
            f"MRR: "
            f"{vector_result.reciprocal_rank:.3f}"
        )

        print(
            "\nAfter reranking:"
        )

        for chunk in reranked_results:
            print(
                f"  page {chunk.page_number} "
                f"| vector={chunk.score:.4f} "
                f"| rerank="
                f"{chunk.rerank_score:.4f}"
            )

        print(
            f"Recall@{TOP_K}: "
            f"{reranker_result.recall_at_k:.3f}"
        )

        print(
            f"MRR: "
            f"{reranker_result.reciprocal_rank:.3f}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nVector Retrieval:"
    )

    print(
        "Hit@K:",
        average([
            metric.hit_at_k
            for metric in vector_metrics
        ])
    )

    print(
        "Precision@K:",
        average([
            metric.precision_at_k
            for metric in vector_metrics
        ])
    )

    print(
        "Recall@K:",
        average([
            metric.recall_at_k
            for metric in vector_metrics
        ])
    )

    print(
        "MRR:",
        average([
            metric.reciprocal_rank
            for metric in vector_metrics
        ])
    )

    print(
        "\nVector + Reranker:"
    )

    print(
        "Hit@K:",
        average([
            metric.hit_at_k
            for metric in reranker_metrics
        ])
    )

    print(
        "Precision@K:",
        average([
            metric.precision_at_k
            for metric in reranker_metrics
        ])
    )

    print(
        "Recall@K:",
        average([
            metric.recall_at_k
            for metric in reranker_metrics
        ])
    )

    print(
        "MRR:",
        average([
            metric.reciprocal_rank
            for metric in reranker_metrics
        ])
    )


if __name__ == "__main__":
    main()