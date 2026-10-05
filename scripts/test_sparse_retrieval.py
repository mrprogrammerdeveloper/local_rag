from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)

from local_rag.retrieval.sparse_retriever import (
    BM25SparseRetriever,
)


def main():

    vector_store = (
        QdrantVectorStore(
            vector_size=1024,
        )
    )

    retriever = (
        BM25SparseRetriever(
            vector_store=vector_store,
        )
    )

    question = (
        "What are the aerospace "
        "applications of metamaterials?"
    )

    results = retriever.retrieve(
        query=question,
        top_k=5,
    )

    print()
    print(
        f"Question: {question}"
    )

    print(
        "=" * 80
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nRank: {rank}"
        )

        print(
            f"Source: "
            f"{result.source}"
        )

        print(
            f"Page: "
            f"{result.page_number}"
        )

        print(
            f"Chunk: "
            f"{result.chunk_index}"
        )

        print(
            f"BM25 score: "
            f"{result.score:.4f}"
        )

        print()

        print(
            result.content[:500]
        )

        print(
            "-" * 80
        )


if __name__ == "__main__":
    main()