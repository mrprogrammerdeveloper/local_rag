from dataclasses import replace

from local_rag.retrieval.retriever import (
    RetrievedChunk,
)


class ReciprocalRankFusion:
    def __init__(
        self,
        k: int = 60,
    ):
        self.k = k

    def fuse(
        self,
        result_lists: list[
            list[RetrievedChunk]
        ],
        top_k: int,
    ) -> list[RetrievedChunk]:

        scores: dict[
            tuple[str, int, int],
            float,
        ] = {}

        chunks: dict[
            tuple[str, int, int],
            RetrievedChunk,
        ] = {}

        for results in result_lists:

            for rank, chunk in enumerate(
                results,
                start=1,
            ):

                key = self._chunk_key(
                    chunk
                )

                rrf_score = (
                    1.0
                    / (
                        self.k
                        + rank
                    )
                )

                scores[key] = (
                    scores.get(
                        key,
                        0.0,
                    )
                    + rrf_score
                )

                if key not in chunks:
                    chunks[key] = chunk

        ranked_keys = sorted(
            scores,
            key=lambda key: scores[key],
            reverse=True,
        )

        fused: list[
            RetrievedChunk
        ] = []

        for key in ranked_keys[:top_k]:

            original = chunks[key]

            fused.append(
                replace(
                    original,
                    score=scores[key],
                    rerank_score=None,
                )
            )

        return fused

    @staticmethod
    def _chunk_key(
        chunk: RetrievedChunk,
    ) -> tuple[str, int, int]:

        return (
            chunk.source,
            chunk.page_number,
            chunk.chunk_index,
        )