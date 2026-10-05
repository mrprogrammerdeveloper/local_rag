from local_rag.retrieval.retriever import (
    Retriever,
    RetrievedChunk,
)

from local_rag.retrieval.sparse_retriever import (
    BM25SparseRetriever,
)

from local_rag.retrieval.fusion import (
    ReciprocalRankFusion,
)


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: BM25SparseRetriever,
        fusion: ReciprocalRankFusion,
    ):
        self.dense_retriever = (
            dense_retriever
        )

        self.sparse_retriever = (
            sparse_retriever
        )

        self.fusion = fusion

    def retrieve(
        self,
        query: str,
        candidate_count: int,
        fused_top_k: int,
    ) -> list[RetrievedChunk]:

        dense_results = (
            self.dense_retriever.retrieve(
                query=query,
                top_k=candidate_count,
            )
        )

        sparse_results = (
            self.sparse_retriever.retrieve(
                query=query,
                top_k=candidate_count,
            )
        )

        fused_results = (
            self.fusion.fuse(
                result_lists=[
                    dense_results,
                    sparse_results,
                ],
                top_k=fused_top_k,
            )
        )

        return fused_results