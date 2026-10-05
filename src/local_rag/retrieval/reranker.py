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
    ) -> None:

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

    def score_pairs(
        self,
        pairs: list[
            tuple[str, str]
        ],
    ) -> list[float]:

        if not pairs:
            return []

        scores: list[float] = []

        for start in range(
            0,
            len(pairs),
            self.batch_size,
        ):

            batch = pairs[
                start:
                start + self.batch_size
            ]

            model_pairs = [
                [
                    first,
                    second,
                ]
                for first, second
                in batch
            ]

            inputs = (
                self.tokenizer(
                    model_pairs,
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

                batch_scores = (
                    outputs.logits
                    .view(-1)
                    .float()
                    .cpu()
                    .tolist()
                )

            scores.extend(
                float(score)
                for score
                in batch_scores
            )

        return scores

    def score_pair(
        self,
        first: str,
        second: str,
    ) -> float:

        scores = self.score_pairs(
            [
                (
                    first,
                    second,
                )
            ]
        )

        if not scores:
            return float("-inf")

        return scores[0]

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

        pairs = [
            (
                query,
                chunk.content,
            )
            for chunk in chunks
        ]

        scores = self.score_pairs(
            pairs
        )

        scored_chunks: list[
            RetrievedChunk
        ] = []

        for chunk, score in zip(
            chunks,
            scores,
        ):

            scored_chunks.append(
                replace(
                    chunk,
                    rerank_score=score,
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

        return scored_chunks[
            :top_k
        ]