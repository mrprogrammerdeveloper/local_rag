from dataclasses import dataclass

from local_rag.config import TOP_K
from local_rag.embeddings.embedder import Embedder
from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    page_number: int
    chunk_index: int
    score: float
    rerank_score: float | None = None


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[RetrievedChunk]:

        query = query.strip()

        if not query:
            return []

        query_vector = (
            self.embedder.embed_text(
                query
            )
        )

        candidate_count = max(
            top_k * 3,
            top_k,
        )

        results = (
            self.vector_store.search(
                query_vector=query_vector,
                limit=candidate_count,
            )
        )

        candidates: list[
            RetrievedChunk
        ] = []

        for result in results:
            payload = (
                result.payload
                or {}
            )

            candidates.append(
                RetrievedChunk(
                    content=str(
                        payload.get(
                            "content",
                            "",
                        )
                    ),
                    source=str(
                        payload.get(
                            "source",
                            "unknown",
                        )
                    ),
                    page_number=int(
                        payload.get(
                            "page_number",
                            0,
                        )
                    ),
                    chunk_index=int(
                        payload.get(
                            "chunk_index",
                            0,
                        )
                    ),
                    score=float(
                        result.score
                    ),
                )
            )

        return self._deduplicate(
            candidates=candidates,
            limit=top_k,
        )

    def _deduplicate(
        self,
        candidates: list[
            RetrievedChunk
        ],
        limit: int,
    ) -> list[RetrievedChunk]:

        selected: list[
            RetrievedChunk
        ] = []

        for candidate in candidates:

            if not candidate.content:
                continue

            duplicate = any(
                self._content_similarity(
                    candidate.content,
                    existing.content,
                )
                >= 0.80
                for existing in selected
            )

            if duplicate:
                continue

            selected.append(
                candidate
            )

            if len(selected) >= limit:
                break

        return selected

    @staticmethod
    def _content_similarity(
        first: str,
        second: str,
    ) -> float:

        first_words = set(
            first.lower().split()
        )

        second_words = set(
            second.lower().split()
        )

        if (
            not first_words
            or not second_words
        ):
            return 0.0

        intersection = (
            first_words
            & second_words
        )

        union = (
            first_words
            | second_words
        )

        return (
            len(intersection)
            / len(union)
        )