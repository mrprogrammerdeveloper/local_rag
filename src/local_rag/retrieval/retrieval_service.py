from __future__ import annotations

import importlib
import inspect
from typing import Any

from local_rag.config import (
    RETRIEVAL_CANDIDATES,
    TOP_K,
)

from local_rag.embeddings.embedder import (
    Embedder,
)

from local_rag.retrieval.reranker import (
    Reranker,
)

from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)


class RetrievalService:
    """
    Thin application service around the project's existing
    dense retriever + BGE reranker.

    The retriever implementation is discovered dynamically so
    the MCP layer does not depend on a hard-coded retriever
    class name.
    """

    def __init__(
        self,
        candidates: int = RETRIEVAL_CANDIDATES,
        default_top_k: int = TOP_K,
    ) -> None:

        self.candidates = candidates
        self.default_top_k = default_top_k

        self._embedder: Embedder | None = None
        self._vector_store: QdrantVectorStore | None = None
        self._retriever: Any | None = None
        self._reranker: Reranker | None = None

    @property
    def embedder(
        self,
    ) -> Embedder:

        if self._embedder is None:

            self._embedder = (
                Embedder()
            )

        return self._embedder

    @property
    def vector_store(
        self,
    ) -> QdrantVectorStore:

        if self._vector_store is None:

            self._vector_store = (
                QdrantVectorStore(
                    vector_size=(
                        self.embedder.dimension
                    )
                )
            )

        return self._vector_store

    @property
    def reranker(
        self,
    ) -> Reranker:

        if self._reranker is None:

            self._reranker = (
                Reranker()
            )

        return self._reranker

    @property
    def retriever(
        self,
    ) -> Any:

        if self._retriever is None:

            self._retriever = (
                self._build_retriever()
            )

        return self._retriever

    def search(
        self,
        query: str,
        top_k: int | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:

        query = query.strip()

        if not query:

            raise ValueError(
                "query cannot be empty"
            )

        if top_k is None:

            top_k = (
                self.default_top_k
            )

        top_k = max(
            1,
            min(
                int(top_k),
                20,
            ),
        )

        candidates = (
            self._retrieve_candidates(
                query
            )
        )

        if source:

            source_lower = (
                source.strip()
                .lower()
            )

            candidates = [
                chunk
                for chunk
                in candidates
                if str(
                    getattr(
                        chunk,
                        "source",
                        "",
                    )
                ).lower()
                == source_lower
            ]

        if not candidates:

            return {
                "query": query,
                "count": 0,
                "results": [],
            }

        reranked = (
            self.reranker.rerank(
                query=query,
                chunks=candidates,
                top_k=top_k,
            )
        )

        results = []

        for rank, chunk in enumerate(
            reranked,
            start=1,
        ):

            result = {
                "rank": rank,
                "content": getattr(
                    chunk,
                    "content",
                    "",
                ),
                "source": getattr(
                    chunk,
                    "source",
                    None,
                ),
                "page": getattr(
                    chunk,
                    "page_number",
                    None,
                ),
                "chunk_index": getattr(
                    chunk,
                    "chunk_index",
                    None,
                ),
                "vector_score": getattr(
                    chunk,
                    "score",
                    None,
                ),
                "rerank_score": getattr(
                    chunk,
                    "rerank_score",
                    None,
                ),
            }

            section_title = getattr(
                chunk,
                "section_title",
                None,
            )

            if section_title:

                result[
                    "section"
                ] = section_title

            page_end = getattr(
                chunk,
                "page_end",
                None,
            )

            if page_end is not None:

                result[
                    "page_end"
                ] = page_end

            results.append(
                result
            )

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }

    def _retrieve_candidates(
        self,
        query: str,
    ) -> list[Any]:

        method = getattr(
            self.retriever,
            "retrieve",
            None,
        )

        if method is None:

            raise RuntimeError(
                "Discovered retriever has no "
                "retrieve() method."
            )

        signature = (
            inspect.signature(
                method
            )
        )

        kwargs: dict[str, Any] = {}

        parameters = (
            signature.parameters
        )

        candidate_names = {
            "top_k",
            "k",
            "limit",
            "candidate_count",
            "candidates",
            "n_results",
        }

        for name in candidate_names:

            if name in parameters:

                kwargs[
                    name
                ] = self.candidates

                break

        try:

            result = method(
                query,
                **kwargs,
            )

        except TypeError:

            result = method(
                query
            )

        return list(
            result
        )

    def _build_retriever(
        self,
    ) -> Any:

        module = (
            importlib.import_module(
                "local_rag.retrieval.retriever"
            )
        )

        preferred_names = [
            "DenseRetriever",
            "Retriever",
            "VectorRetriever",
        ]

        candidates = []

        for name in preferred_names:

            candidate = getattr(
                module,
                name,
                None,
            )

            if (
                inspect.isclass(
                    candidate
                )
                and hasattr(
                    candidate,
                    "retrieve",
                )
            ):

                candidates.append(
                    candidate
                )

        if not candidates:

            for _, candidate in (
                inspect.getmembers(
                    module,
                    inspect.isclass,
                )
            ):

                if candidate.__module__ != (
                    module.__name__
                ):
                    continue

                if candidate.__name__ == (
                    "RetrievedChunk"
                ):
                    continue

                if hasattr(
                    candidate,
                    "retrieve",
                ):

                    candidates.append(
                        candidate
                    )

        if not candidates:

            raise RuntimeError(
                "Could not discover a retriever "
                "class in retrieval/retriever.py."
            )

        errors: list[str] = []

        for retriever_class in (
            candidates
        ):

            try:

                return (
                    self._construct_retriever(
                        retriever_class
                    )
                )

            except Exception as exc:

                errors.append(
                    f"{retriever_class.__name__}: "
                    f"{exc}"
                )

        raise RuntimeError(
            "Could not initialize project "
            "retriever. Attempts: "
            + " | ".join(
                errors
            )
        )

    def _construct_retriever(
        self,
        retriever_class: type,
    ) -> Any:

        init_signature = (
            inspect.signature(
                retriever_class.__init__
            )
        )

        kwargs: dict[str, Any] = {}

        for name, parameter in (
            init_signature
            .parameters
            .items()
        ):

            if name == "self":
                continue

            lowered = (
                name.lower()
            )

            if lowered in {
                "embedder",
                "embedding",
                "embedding_model",
            }:

                kwargs[
                    name
                ] = self.embedder

                continue

            if lowered in {
                "vector_store",
                "store",
                "qdrant",
                "qdrant_store",
            }:

                kwargs[
                    name
                ] = self.vector_store

                continue

            if lowered in {
                "candidate_count",
                "candidates",
                "top_k",
                "limit",
            }:

                if (
                    parameter.default
                    is inspect.Parameter.empty
                ):

                    kwargs[
                        name
                    ] = self.candidates

                continue

            if (
                parameter.default
                is inspect.Parameter.empty
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            ):

                raise TypeError(
                    "Unsupported required "
                    f"constructor parameter: {name}"
                )

        try:

            return retriever_class(
                **kwargs
            )

        except TypeError:

            attempts = [
                (
                    self.embedder,
                    self.vector_store,
                ),
                (
                    self.vector_store,
                    self.embedder,
                ),
                (
                    self.embedder,
                ),
                (
                    self.vector_store,
                ),
                tuple(),
            ]

            last_error = None

            for args in attempts:

                try:

                    return retriever_class(
                        *args
                    )

                except Exception as exc:

                    last_error = exc

            raise TypeError(
                str(
                    last_error
                )
            )
