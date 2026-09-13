from dataclasses import replace

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from local_rag.config import (
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_MODEL,
    TOP_K,
)

from local_rag.retrieval.retriever import (
    RetrievedChunk,
)


class Reranker:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: str = RERANKER_DEVICE,
        batch_size: int = RERANKER_BATCH_SIZE,
    ):
        self.device = torch.device(
            device
        )

        self.batch_size = batch_size

        print(
            f"Loading reranker: "
            f"{model_name} "
            f"on {self.device}..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = TOP_K,
    ) -> list[RetrievedChunk]:

        if not chunks:
            return []

        query = query.strip()

        if not query:
            return chunks[:top_k]

        scored_chunks: list[
            RetrievedChunk
        ] = []

        for start in range(
            0,
            len(chunks),
            self.batch_size,
        ):
            batch = chunks[
                start:
                start + self.batch_size
            ]

            pairs = [
                [
                    query,
                    chunk.content,
                ]
                for chunk in batch
            ]

            inputs = (
                self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
            )

            inputs = {
                key: value.to(
                    self.device
                )
                for key, value
                in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    return_dict=True,
                )

                scores = (
                    outputs.logits
                    .view(-1)
                    .float()
                    .cpu()
                    .tolist()
                )

            for chunk, score in zip(
                batch,
                scores,
            ):
                scored_chunks.append(
                    replace(
                        chunk,
                        rerank_score=float(
                            score
                        ),
                    )
                )

        scored_chunks.sort(
            key=lambda chunk: (
                chunk.rerank_score
                if chunk.rerank_score
                is not None
                else float("-inf")
            ),
            reverse=True,
        )

        return scored_chunks[:top_k]