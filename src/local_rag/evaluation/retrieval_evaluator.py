from local_rag.evaluation.models import (
    RetrievalMetrics,
    RetrievalTestCase,
)

from local_rag.retrieval.retriever import (
    RetrievedChunk,
)


class RetrievalEvaluator:

    def evaluate(
        self,
        test_case: RetrievalTestCase,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RetrievalMetrics:

        if not retrieved_chunks:
            return RetrievalMetrics(
                hit_at_k=0.0,
                precision_at_k=0.0,
                recall_at_k=0.0,
                reciprocal_rank=0.0,
            )

        relevant_pages = set(
            test_case.relevant_pages
        )

        retrieved_pages = [
            chunk.page_number
            for chunk in retrieved_chunks
            if chunk.source == test_case.source
        ]

        unique_retrieved_pages = set(
            retrieved_pages
        )

        matched_pages = (
            relevant_pages
            & unique_retrieved_pages
        )

        hit_at_k = (
            1.0
            if matched_pages
            else 0.0
        )

        precision_at_k = (
            len(matched_pages)
            / len(unique_retrieved_pages)
            if unique_retrieved_pages
            else 0.0
        )

        recall_at_k = (
            len(matched_pages)
            / len(relevant_pages)
            if relevant_pages
            else 0.0
        )

        reciprocal_rank = 0.0

        for rank, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            if (
                chunk.source
                == test_case.source
                and chunk.page_number
                in relevant_pages
            ):
                reciprocal_rank = (
                    1.0 / rank
                )

                break

        return RetrievalMetrics(
            hit_at_k=hit_at_k,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            reciprocal_rank=reciprocal_rank,
        )