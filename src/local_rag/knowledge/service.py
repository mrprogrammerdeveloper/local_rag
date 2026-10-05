from __future__ import annotations

import shutil

from pathlib import Path
from typing import Any

from local_rag.config import (
    PDF_PATH,
    REFERENCES_PATH,
)

from local_rag.embeddings.embedder import (
    Embedder,
)

from local_rag.ingestion.pipelines import (
    register_ingestion_pipelines,
)

from local_rag.ingestion.registry import (
    IngestionRegistry,
)

from local_rag.references.store import (
    ReferenceStore,
)

from local_rag.retrieval.retrieval_service import (
    RetrievalService,
)

from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)


class KnowledgeService:
    """
    Application-facing academic knowledge service.

    Claude performs reasoning and generation.
    This service only exposes ingestion + retrieval.
    """

    def __init__(
        self,
    ) -> None:

        self.project_root = (
            Path.cwd()
        )

        self.pdf_directory = Path(
            PDF_PATH
        )

        self.excel_directory = (
            Path("data/excel")
        )

        self.pdf_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.excel_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._retrieval_service: (
            RetrievalService | None
        ) = None

        self._embedder: (
            Embedder | None
        ) = None

        self._vector_store: (
            QdrantVectorStore | None
        ) = None

        self._pipelines_registered = (
            False
        )

    @property
    def retrieval_service(
        self,
    ) -> RetrievalService:

        if (
            self._retrieval_service
            is None
        ):

            self._retrieval_service = (
                RetrievalService()
            )

        return (
            self._retrieval_service
        )

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

    def list_sources(
        self,
    ) -> dict[str, Any]:

        pdfs = [
            {
                "name": path.name,
                "path": str(
                    path.resolve()
                ),
                "type": "pdf",
            }
            for path in sorted(
                self.pdf_directory.glob(
                    "*.pdf"
                )
            )
        ]

        excels = []

        for suffix in (
            "*.xlsx",
            "*.xlsm",
        ):

            for path in sorted(
                self.excel_directory.glob(
                    suffix
                )
            ):

                excels.append(
                    {
                        "name": (
                            path.name
                        ),
                        "path": str(
                            path.resolve()
                        ),
                        "type": (
                            "excel"
                        ),
                    }
                )

        return {
            "pdfs": pdfs,
            "excel": excels,
            "count": (
                len(pdfs)
                + len(excels)
            ),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
    ) -> dict[str, Any]:

        return (
            self.retrieval_service.search(
                query=query,
                top_k=top_k,
                source=source,
            )
        )

    def ingest_documents(
        self,
        paths: list[str],
        pipeline: str = "v2-basic",
    ) -> dict[str, Any]:

        if not paths:

            raise ValueError(
                "paths cannot be empty"
            )

        self._ensure_pipelines()

        available = (
            IngestionRegistry.names()
        )

        selected_pipeline = (
            self._resolve_pipeline_name(
                requested=pipeline,
                available=available,
            )
        )

        ingestion = (
            IngestionRegistry.create(
                selected_pipeline
            )
        )

        reference_store = (
            ReferenceStore(
                REFERENCES_PATH
            )
        )

        results = []

        for raw_path in paths:

            path = (
                Path(raw_path)
                .expanduser()
                .resolve()
            )

            if not path.exists():

                results.append(
                    {
                        "path": raw_path,
                        "status": "error",
                        "error": (
                            "File not found"
                        ),
                    }
                )

                continue

            suffix = (
                path.suffix.lower()
            )

            if suffix in {
                ".xlsx",
                ".xlsm",
            }:

                destination = (
                    self.excel_directory
                    / path.name
                )

                if (
                    destination.resolve()
                    != path
                ):

                    shutil.copy2(
                        path,
                        destination,
                    )

                results.append(
                    {
                        "path": str(path),
                        "status": "stored",
                        "type": "excel",
                        "destination": str(
                            destination.resolve()
                        ),
                    }
                )

                continue

            if suffix != ".pdf":

                results.append(
                    {
                        "path": str(path),
                        "status": "error",
                        "error": (
                            "Supported formats: "
                            "PDF, XLSX, XLSM"
                        ),
                    }
                )

                continue

            result = ingestion.ingest(
                path
            )

            reference_store.save(
                source=result.source,
                references=(
                    result.references
                ),
            )

            if not result.chunks:

                results.append(
                    {
                        "path": str(path),
                        "status": "skipped",
                        "type": "pdf",
                        "reason": (
                            "No chunks created"
                        ),
                    }
                )

                continue

            texts = [
                (
                    getattr(
                        chunk,
                        "embedding_text",
                        None,
                    )
                    or chunk.content
                )
                for chunk
                in result.chunks
            ]

            vectors = (
                self.embedder.embed_texts(
                    texts
                )
            )

            self.vector_store.upsert(
                chunks=result.chunks,
                vectors=vectors,
            )

            destination = (
                self.pdf_directory
                / path.name
            )

            if (
                destination.resolve()
                != path
            ):

                shutil.copy2(
                    path,
                    destination,
                )

            results.append(
                {
                    "path": str(path),
                    "status": "indexed",
                    "type": "pdf",
                    "pipeline": (
                        selected_pipeline
                    ),
                    "pages": len(
                        result.pages
                    ),
                    "main_pages": len(
                        result.main_pages
                    ),
                    "references": len(
                        result.references
                    ),
                    "chunks": len(
                        result.chunks
                    ),
                    "source": (
                        result.source
                    ),
                }
            )

        # Ensure future searches build a fresh retriever
        # after the corpus changes.
        self._retrieval_service = None

        return {
            "pipeline": (
                selected_pipeline
            ),
            "available_pipelines": (
                available
            ),
            "results": results,
        }

    def _ensure_pipelines(
        self,
    ) -> None:

        if self._pipelines_registered:

            return

        register_ingestion_pipelines()

        self._pipelines_registered = (
            True
        )

    @staticmethod
    def _resolve_pipeline_name(
        requested: str,
        available: list[str],
    ) -> str:

        if requested in available:

            return requested

        aliases = {
            "v2-basic": [
                "v2-basic",
                "v2",
            ],
            "v2": [
                "v2",
                "v2-basic",
            ],
            "v1": [
                "v1",
            ],
        }

        for candidate in (
            aliases.get(
                requested,
                []
            )
        ):

            if candidate in available:

                return candidate

        raise ValueError(
            f"Unknown ingestion pipeline: "
            f"{requested}. Available: "
            f"{', '.join(available)}"
        )
