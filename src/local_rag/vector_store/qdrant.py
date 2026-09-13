from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from local_rag.config import (
    QDRANT_COLLECTION,
    VECTOR_STORE_PATH,
)
from local_rag.ingestion.models import DocumentChunk


class QdrantVectorStore:
    def __init__(
        self,
        vector_size: int,
        path: str | Path = VECTOR_STORE_PATH,
        collection_name: str = QDRANT_COLLECTION,
    ):
        self.path = Path(path)
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = QdrantClient(
            path=str(self.path),
        )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(
            collection_name=self.collection_name
        ):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:

        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks and vectors must be equal."
            )

        points: list[PointStruct] = []

        for chunk, vector in zip(chunks, vectors):
            point_id = self._create_point_id(
                chunk
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": chunk.content,
                        "source": chunk.source,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ):
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return result.points

    @staticmethod
    def _create_point_id(
        chunk: DocumentChunk,
    ) -> str:

        unique_value = (
            f"{chunk.source}:"
            f"{chunk.page_number}:"
            f"{chunk.chunk_index}"
        )

        return str(
            uuid5(
                NAMESPACE_URL,
                unique_value,
            )
        )