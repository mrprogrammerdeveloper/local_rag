from sentence_transformers import SentenceTransformer

from local_rag.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    @property
    def dimension(self) -> int:
        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "Could not determine embedding dimension."
            )

        return dimension

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()