import json
from pathlib import Path

from local_rag.config import (
    RETRIEVAL_CANDIDATES,
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

K_VALUES = [
    2,
    3,
    4,
    5,
    6,
]


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


def average(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return sum(values) / len(values)


def evaluate_for_k(
    k: int,
    dataset: list[RetrievalTestCase],
    retriever: Retriever,
    reranker: Reranker,
    evaluator: RetrievalEvaluator,
):

    metrics = []

    for test_case in dataset:

        candidates = retriever.retrieve(
            query=test_case.question,
            top_k=RETRIEVAL_CANDIDATES,
        )

        reranked = reranker.rerank(
            query=test_case.question,
            chunks=candidates,
            top_k=k,
        )

        result = evaluator.evaluate(
            test_case=test_case,
            retrieved_chunks=reranked,
        )

        metrics.append(
            result
        )

    return {
        "k": k,

        "hit": average([
            metric.hit_at_k
            for metric in metrics
        ]),

        "precision": average([
            metric.precision_at_k
            for metric in metrics
        ]),

        "recall": average([
            metric.recall_at_k
            for metric in metrics
        ]),

        "mrr": average([
            metric.reciprocal_rank
            for metric in metrics
        ]),
    }


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

    results = []

    for k in K_VALUES:

        print(
            f"\nEvaluating K={k}..."
        )

        result = evaluate_for_k(
            k=k,
            dataset=dataset,
            retriever=retriever,
            reranker=reranker,
            evaluator=evaluator,
        )

        results.append(
            result
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "K SWEEP RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"{'K':<6}"
        f"{'Hit@K':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<12}"
        f"{'MRR':<12}"
    )

    print(
        "-" * 80
    )

    for result in results:

        print(
            f"{result['k']:<6}"
            f"{result['hit']:<12.3f}"
            f"{result['precision']:<14.3f}"
            f"{result['recall']:<12.3f}"
            f"{result['mrr']:<12.3f}"
        )


if __name__ == "__main__":
    main()