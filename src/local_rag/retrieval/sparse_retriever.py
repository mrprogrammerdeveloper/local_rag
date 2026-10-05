import re

from rank_bm25 import BM25Okapi

from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)

from local_rag.retrieval.retriever import (
    RetrievedChunk,
)


class BM25SparseRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
    ):
        self.vector_store = vector_store

        self.chunks: list[
            RetrievedChunk
        ] = []

        self.tokenized_corpus: list[
            list[str]
        ] = []

        self.bm25: BM25Okapi | None = None

        self._load_index()

    def _load_index(
        self,
    ) -> None:

        payloads = (
            self.vector_store
            .get_all_payloads()
        )

        chunks: list[
            RetrievedChunk
        ] = []

        tokenized_corpus: list[
            list[str]
        ] = []

        for payload in payloads:

            content = str(
                payload.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:
                continue

            chunk = RetrievedChunk(
                content=content,
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
                score=0.0,
            )

            chunks.append(
                chunk
            )

            tokenized_corpus.append(
                self._tokenize(
                    content
                )
            )

        self.chunks = chunks

        self.tokenized_corpus = (
            tokenized_corpus
        )

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(
                self.tokenized_corpus
            )

        print(
            f"BM25 index loaded: "
            f"{len(self.chunks)} chunks"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:

        query = query.strip()

        if (
            not query
            or self.bm25 is None
        ):
            return []

        query_tokens = (
            self._tokenize(
                query
            )
        )

        if not query_tokens:
            return []

        scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (
                scores[index]
            ),
            reverse=True,
        )

        results: list[
            RetrievedChunk
        ] = []

        for index in ranked_indices:

            if len(results) >= top_k:
                break

            score = float(
                scores[index]
            )

            chunk = (
                self.chunks[index]
            )

            results.append(
                RetrievedChunk(
                    content=chunk.content,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=score,
                )
            )

        return results

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"\b\w+\b",
            text.lower(),
            flags=re.UNICODE,
        )