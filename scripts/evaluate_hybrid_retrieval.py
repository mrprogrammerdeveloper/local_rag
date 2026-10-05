import json
from pathlib import Path

from local_rag.embeddings.embedder import (
    Embedder,
)

from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)

from local_rag.retrieval.retriever import (
    Retriever,
)

from local_rag.retrieval.sparse_retriever import (
    BM25SparseRetriever,
)

from local_rag.retrieval.fusion import (
    ReciprocalRankFusion,
)

from local_rag.retrieval.hybrid_retriever import (
    HybridRetriever,
)

from local_rag.retrieval.reranker import (
    Reranker,
)


DATASET_PATH = Path(
    "data/evaluation/retrieval_dataset.json"
)

TOP_K = 5

CANDIDATE_COUNT = 8

HYBRID_FUSED_COUNT = 12


DocumentPageKey = tuple[str, int]


def build_relevant_pages(
    documents: list[dict],
) -> set[DocumentPageKey]:

    relevant: set[
        DocumentPageKey
    ] = set()

    for document in documents:

        source = document[
            "source"
        ]

        for page in document[
            "pages"
        ]:

            relevant.add(
                (
                    source,
                    int(page),
                )
            )

    return relevant


def recall_at_k(
    retrieved: list[DocumentPageKey],
    relevant: set[DocumentPageKey],
) -> float:

    if not relevant:
        return 0.0

    retrieved_set = set(
        retrieved
    )

    return (
        len(
            retrieved_set
            & relevant
        )
        / len(relevant)
    )


def precision_at_k(
    retrieved: list[DocumentPageKey],
    relevant: set[DocumentPageKey],
) -> float:

    retrieved_set = set(
        retrieved
    )

    if not retrieved_set:
        return 0.0

    return (
        len(
            retrieved_set
            & relevant
        )
        / len(retrieved_set)
    )


def hit_at_k(
    retrieved: list[DocumentPageKey],
    relevant: set[DocumentPageKey],
) -> float:

    retrieved_set = set(
        retrieved
    )

    return (
        1.0
        if retrieved_set & relevant
        else 0.0
    )


def reciprocal_rank(
    retrieved: list[DocumentPageKey],
    relevant: set[DocumentPageKey],
) -> float:

    for rank, item in enumerate(
        retrieved,
        start=1,
    ):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def average(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def evaluate_results(
    results,
    relevant: set[DocumentPageKey],
):

    retrieved = [
        (
            item.source,
            item.page_number,
        )
        for item in results
    ]

    return {
        "hit": hit_at_k(
            retrieved,
            relevant,
        ),
        "precision": precision_at_k(
            retrieved,
            relevant,
        ),
        "recall": recall_at_k(
            retrieved,
            relevant,
        ),
        "mrr": reciprocal_rank(
            retrieved,
            relevant,
        ),
        "pages": retrieved,
    }


def main():

    dataset = json.loads(
        DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        "Loading embedding model..."
    )

    embedder = Embedder()

    vector_store = (
        QdrantVectorStore(
            vector_size=1024,
        )
    )

    dense_retriever = (
        Retriever(
            embedder=embedder,
            vector_store=vector_store,
        )
    )

    sparse_retriever = (
        BM25SparseRetriever(
            vector_store=vector_store,
        )
    )

    fusion = (
        ReciprocalRankFusion()
    )

    hybrid_retriever = (
        HybridRetriever(
            dense_retriever=(
                dense_retriever
            ),
            sparse_retriever=(
                sparse_retriever
            ),
            fusion=fusion,
        )
    )

    reranker = Reranker()

    dense_metrics = []

    dense_rerank_metrics = []

    hybrid_metrics = []

    for item in dataset:

        question = item[
            "question"
        ]

        relevant_documents = item[
            "relevant_documents"
        ]

        relevant = (
            build_relevant_pages(
                relevant_documents
            )
        )

        print()
        print(
            "=" * 80
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Relevant: "
            f"{sorted(relevant)}"
        )

        # --------------------------------
        # Dense
        # --------------------------------

        dense_candidates = (
            dense_retriever.retrieve(
                query=question,
                top_k=CANDIDATE_COUNT,
            )
        )

        dense_top = (
            dense_candidates[
                :TOP_K
            ]
        )

        dense_result = (
            evaluate_results(
                dense_top,
                relevant,
            )
        )

        dense_metrics.append(
            dense_result
        )

        # --------------------------------
        # Dense + Reranker
        # --------------------------------

        dense_reranked = (
            reranker.rerank(
                query=question,
                chunks=dense_candidates,
                top_k=TOP_K,
            )
        )

        dense_rerank_result = (
            evaluate_results(
                dense_reranked,
                relevant,
            )
        )

        dense_rerank_metrics.append(
            dense_rerank_result
        )

        # --------------------------------
        # Hybrid + RRF + Reranker
        # --------------------------------

        hybrid_candidates = (
            hybrid_retriever.retrieve(
                query=question,
                candidate_count=(
                    CANDIDATE_COUNT
                ),
                fused_top_k=(
                    HYBRID_FUSED_COUNT
                ),
            )
        )

        hybrid_reranked = (
            reranker.rerank(
                query=question,
                chunks=hybrid_candidates,
                top_k=TOP_K,
            )
        )

        hybrid_result = (
            evaluate_results(
                hybrid_reranked,
                relevant,
            )
        )

        hybrid_metrics.append(
            hybrid_result
        )

        print(
            f"Dense pages: "
            f"{dense_result['pages']}"
        )

        print(
            f"Dense + rerank pages: "
            f"{dense_rerank_result['pages']}"
        )

        print(
            f"Hybrid pages: "
            f"{hybrid_result['pages']}"
        )

        print()

        print(
            "Dense metrics:"
        )

        print(
            f"Hit={dense_result['hit']:.4f} | "
            f"Precision={dense_result['precision']:.4f} | "
            f"Recall={dense_result['recall']:.4f} | "
            f"MRR={dense_result['mrr']:.4f}"
        )

        print(
            "Dense + Reranker metrics:"
        )

        print(
            f"Hit={dense_rerank_result['hit']:.4f} | "
            f"Precision={dense_rerank_result['precision']:.4f} | "
            f"Recall={dense_rerank_result['recall']:.4f} | "
            f"MRR={dense_rerank_result['mrr']:.4f}"
        )

        print(
            "Hybrid + Reranker metrics:"
        )

        print(
            f"Hit={hybrid_result['hit']:.4f} | "
            f"Precision={hybrid_result['precision']:.4f} | "
            f"Recall={hybrid_result['recall']:.4f} | "
            f"MRR={hybrid_result['mrr']:.4f}"
        )

    systems = {
        "Dense":
            dense_metrics,

        "Dense + Reranker":
            dense_rerank_metrics,

        "Hybrid + Reranker":
            hybrid_metrics,
    }

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL RETRIEVAL COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        f"{'System':<25}"
        f"{'Hit@K':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'MRR':<12}"
    )

    print(
        "-" * 100
    )

    for name, metrics in (
        systems.items()
    ):

        hit = average(
            [
                x["hit"]
                for x in metrics
            ]
        )

        precision = average(
            [
                x["precision"]
                for x in metrics
            ]
        )

        recall = average(
            [
                x["recall"]
                for x in metrics
            ]
        )

        mrr = average(
            [
                x["mrr"]
                for x in metrics
            ]
        )

        print(
            f"{name:<25}"
            f"{hit:<12.4f}"
            f"{precision:<14.4f}"
            f"{recall:<14.4f}"
            f"{mrr:<12.4f}"
        )


if __name__ == "__main__":
    main()