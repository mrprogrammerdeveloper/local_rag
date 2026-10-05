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

        print(
            f"{rank}. "
            f"{result.source} | "
            f"page={result.page_number} | "
            f"chunk={result.chunk_index} | "
            f"score={result.score:.6f}"
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

    question = (
        "What are the aerospace "
        "applications of metamaterials?"
    )

    dense_results = (
        dense_retriever.retrieve(
            query=question,
            top_k=8,
        )
    )

    sparse_results = (
        sparse_retriever.retrieve(
            query=question,
            top_k=8,
        )
    )

    fused_results = fusion.fuse(
        result_lists=[
            dense_results,
            sparse_results,
        ],
        top_k=12,
    )

    print_results(
        "DENSE",
        dense_results,
    )

    print_results(
        "BM25",
        sparse_results,
    )

    print_results(
        "RRF",
        fused_results,
    )


if __name__ == "__main__":
    main()