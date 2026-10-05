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


def print_results(
    title,
    results,
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        score = (
            result.rerank_score
            if result.rerank_score
            is not None
            else result.score
        )

        print(
            f"{rank}. "
            f"{result.source} | "
            f"page={result.page_number} | "
            f"chunk={result.chunk_index} | "
            f"score={score:.6f}"
        )


def main():

    print(
        "Loading embedding model..."
    )

    embedder = Embedder()

    vector_store = (
        QdrantVectorStore(
            vector_size=1024,
        )
    )

    dense_retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
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

    question = (
        "What are the aerospace "
        "applications of metamaterials?"
    )

    hybrid_results = (
        hybrid_retriever.retrieve(
            query=question,
            candidate_count=8,
            fused_top_k=12,
        )
    )

    reranked_results = (
        reranker.rerank(
            query=question,
            chunks=hybrid_results,
            top_k=5,
        )
    )

    print_results(
        "HYBRID RRF",
        hybrid_results,
    )

    print_results(
        "HYBRID + RERANKER",
        reranked_results,
    )


if __name__ == "__main__":
    main()