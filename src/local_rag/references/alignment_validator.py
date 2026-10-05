from local_rag.references.alignment import (
    ReferenceAlignmentResult,
)

from local_rag.references.store import (
    ReferenceStore,
)

from local_rag.retrieval.reranker import (
    Reranker,
)


class ReferenceAlignmentValidator:

    def __init__(
        self,
        reference_store: ReferenceStore,
        reranker: Reranker,
    ) -> None:

        self.reference_store = (
            reference_store
        )

        self.reranker = reranker

    def score(
        self,
        evidence_text: str,
        source: str,
        reference_number: int,
    ) -> ReferenceAlignmentResult | None:

        reference_text = (
            self.reference_store.get(
                source=source,
                number=reference_number,
            )
        )

        if not reference_text:
            return None

        score = (
            self.reranker.score_pair(
                first=evidence_text,
                second=reference_text,
            )
        )

        return (
            ReferenceAlignmentResult(
                source=source,
                reference_number=(
                    reference_number
                ),
                evidence_text=(
                    evidence_text
                ),
                reference_text=(
                    reference_text
                ),
                score=score,
            )
        )

    def score_many(
        self,
        evidence_text: str,
        source: str,
        reference_numbers: list[int],
    ) -> list[
        ReferenceAlignmentResult
    ]:

        available: list[
            tuple[
                int,
                str,
            ]
        ] = []

        for number in reference_numbers:

            reference_text = (
                self.reference_store.get(
                    source=source,
                    number=number,
                )
            )

            if not reference_text:
                continue

            available.append(
                (
                    number,
                    reference_text,
                )
            )

        if not available:
            return []

        pairs = [
            (
                evidence_text,
                reference_text,
            )
            for _, reference_text
            in available
        ]

        scores = (
            self.reranker.score_pairs(
                pairs
            )
        )

        results = []

        for (
            number,
            reference_text,
        ), score in zip(
            available,
            scores,
        ):

            results.append(
                ReferenceAlignmentResult(
                    source=source,
                    reference_number=number,
                    evidence_text=evidence_text,
                    reference_text=reference_text,
                    score=score,
                )
            )

        return results