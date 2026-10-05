import csv
import statistics
from pathlib import Path

from local_rag.config import (
    REFERENCES_PATH,
)

from local_rag.references.citation_normalizer import (
    CitationNormalizer,
)

from local_rag.references.store import (
    ReferenceStore,
)

from local_rag.retrieval.evidence_builder import (
    EvidenceBuilder,
)

from local_rag.retrieval.reranker import (
    Reranker,
)

from local_rag.retrieval.retriever import (
    RetrievedChunk,
)

from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)


OUTPUT_PATH = Path(
    "data/evaluation/"
    "reference_alignment_scores.csv"
)

VECTOR_SIZE = 1024

NEGATIVES_PER_EVIDENCE = 2


def percentile(
    values: list[float],
    percent: float,
) -> float:

    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    if len(ordered) == 1:
        return ordered[0]

    position = (
        percent
        / 100.0
        * (len(ordered) - 1)
    )

    lower = int(
        position
    )

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = (
        position - lower
    )

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


def payloads_to_chunks(
    payloads: list[dict],
) -> list[RetrievedChunk]:

    chunks: list[
        RetrievedChunk
    ] = []

    for payload in payloads:

        content = payload.get(
            "content"
        )

        source = payload.get(
            "source"
        )

        page_number = payload.get(
            "page_number"
        )

        chunk_index = payload.get(
            "chunk_index"
        )

        if not content:
            continue

        if not source:
            continue

        if page_number is None:
            continue

        if chunk_index is None:
            continue

        chunks.append(
            RetrievedChunk(
                content=str(
                    content
                ),
                source=str(
                    source
                ),
                page_number=int(
                    page_number
                ),
                chunk_index=int(
                    chunk_index
                ),
                score=0.0,
                rerank_score=None,
            )
        )

    return chunks


def deduplicate_evidence(
    evidence,
):

    unique = []

    seen = set()

    for item in evidence:

        key = (
            item.source,
            item.page_number,
            item.content,
            tuple(
                item.reference_numbers
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            item
        )

    return unique


def nearest_negative_references(
    all_numbers: list[int],
    positive_numbers: list[int],
    count: int,
) -> list[int]:

    if not positive_numbers:
        return []

    positive_set = set(
        positive_numbers
    )

    candidates = [
        number
        for number in all_numbers
        if number not in positive_set
    ]

    if not candidates:
        return []

    anchor = positive_numbers[0]

    candidates.sort(
        key=lambda number: (
            abs(
                number - anchor
            ),
            number,
        )
    )

    return candidates[
        :count
    ]


def print_summary(
    title: str,
    scores: list[float],
) -> None:

    print()
    print(
        title
    )

    print(
        "-" * len(title)
    )

    if not scores:

        print(
            "No scores."
        )

        return

    print(
        f"Count:  {len(scores)}"
    )

    print(
        f"Min:    "
        f"{min(scores):.6f}"
    )

    print(
        f"P10:    "
        f"{percentile(scores, 10):.6f}"
    )

    print(
        f"P25:    "
        f"{percentile(scores, 25):.6f}"
    )

    print(
        f"Median: "
        f"{statistics.median(scores):.6f}"
    )

    print(
        f"Mean:   "
        f"{statistics.mean(scores):.6f}"
    )

    print(
        f"P75:    "
        f"{percentile(scores, 75):.6f}"
    )

    print(
        f"P90:    "
        f"{percentile(scores, 90):.6f}"
    )

    print(
        f"Max:    "
        f"{max(scores):.6f}"
    )


def print_record(
    record: dict,
) -> None:

    print(
        "-" * 80
    )

    print(
        f"Type: "
        f"{record['type']}"
    )

    print(
        f"Score: "
        f"{record['score']:.6f}"
    )

    print(
        f"Source: "
        f"{record['source']}"
    )

    print(
        f"Page: "
        f"{record['page_number']}"
    )

    print(
        f"Reference: "
        f"{record['reference_number']}"
    )

    print()

    print(
        "Evidence:"
    )

    print(
        record[
            "evidence_text"
        ]
    )

    print()

    print(
        "Reference:"
    )

    print(
        record[
            "reference_text"
        ]
    )

    print()


def main() -> None:

    reference_store = (
        ReferenceStore(
            REFERENCES_PATH
        )
    )

    citation_normalizer = (
        CitationNormalizer(
            reference_store
        )
    )

    evidence_builder = (
        EvidenceBuilder(
            citation_normalizer
        )
    )

    print(
        "Loading indexed chunks..."
    )

    vector_store = (
        QdrantVectorStore(
            vector_size=VECTOR_SIZE,
        )
    )

    payloads = (
        vector_store
        .get_all_payloads()
    )

    chunks = (
        payloads_to_chunks(
            payloads
        )
    )

    print(
        f"Loaded "
        f"{len(chunks)} chunks."
    )

    print(
        "Building evidence..."
    )

    evidence = (
        evidence_builder.build(
            chunks
        )
    )

    evidence = (
        deduplicate_evidence(
            evidence
        )
    )

    cited_evidence = [
        item
        for item in evidence
        if item.reference_numbers
    ]

    print(
        f"Evidence items: "
        f"{len(evidence)}"
    )

    print(
        f"Evidence with citations: "
        f"{len(cited_evidence)}"
    )

    print(
        "Loading reranker..."
    )

    reranker = Reranker()

    records: list[
        dict
    ] = []

    pairs: list[
        tuple[str, str]
    ] = []

    pair_metadata: list[
        dict
    ] = []

    # ----------------------------------------
    # Cited pairs
    # ----------------------------------------

    for item in cited_evidence:

        for number in (
            item.reference_numbers
        ):

            reference_text = (
                reference_store.get(
                    source=(
                        item.source
                    ),
                    number=number,
                )
            )

            if not reference_text:
                continue

            pairs.append(
                (
                    item.content,
                    reference_text,
                )
            )

            pair_metadata.append(
                {
                    "type": (
                        "cited"
                    ),
                    "source": (
                        item.source
                    ),
                    "page_number": (
                        item.page_number
                    ),
                    "reference_number": (
                        number
                    ),
                    "evidence_text": (
                        item.content
                    ),
                    "reference_text": (
                        reference_text
                    ),
                }
            )

    # ----------------------------------------
    # Hard negatives
    # ----------------------------------------

    for item in cited_evidence:

        references = (
            reference_store.load(
                item.source
            )
        )

        if not references:
            continue

        negative_numbers = (
            nearest_negative_references(
                all_numbers=list(
                    references.keys()
                ),
                positive_numbers=(
                    item.reference_numbers
                ),
                count=(
                    NEGATIVES_PER_EVIDENCE
                ),
            )
        )

        for number in negative_numbers:

            reference_text = (
                references.get(
                    number
                )
            )

            if not reference_text:
                continue

            pairs.append(
                (
                    item.content,
                    reference_text,
                )
            )

            pair_metadata.append(
                {
                    "type": (
                        "hard_negative"
                    ),
                    "source": (
                        item.source
                    ),
                    "page_number": (
                        item.page_number
                    ),
                    "reference_number": (
                        number
                    ),
                    "evidence_text": (
                        item.content
                    ),
                    "reference_text": (
                        reference_text
                    ),
                }
            )

    print(
        f"Scoring "
        f"{len(pairs)} pairs..."
    )

    scores = (
        reranker.score_pairs(
            pairs
        )
    )

    for metadata, score in zip(
        pair_metadata,
        scores,
    ):

        record = dict(
            metadata
        )

        record[
            "score"
        ] = float(
            score
        )

        records.append(
            record
        )

    cited_records = [
        record
        for record in records
        if record["type"]
        == "cited"
    ]

    negative_records = [
        record
        for record in records
        if record["type"]
        == "hard_negative"
    ]

    cited_scores = [
        record["score"]
        for record
        in cited_records
    ]

    negative_scores = [
        record["score"]
        for record
        in negative_records
    ]

    print()
    print(
        "=" * 80
    )

    print(
        "REFERENCE ALIGNMENT "
        "CALIBRATION"
    )

    print(
        "=" * 80
    )

    print_summary(
        "CITED PAIRS",
        cited_scores,
    )

    print_summary(
        "HARD NEGATIVE PAIRS",
        negative_scores,
    )

    # ----------------------------------------
    # Threshold diagnostics
    # ----------------------------------------

    thresholds = [
        -3.0,
        -2.0,
        -1.0,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ]

    print()
    print(
        "THRESHOLD DIAGNOSTICS"
    )

    print(
        "-" * 80
    )

    print(
        f"{'Threshold':<12}"
        f"{'Cited >= T':<18}"
        f"{'Negative < T':<18}"
    )

    for threshold in thresholds:

        if cited_scores:

            cited_above = (
                sum(
                    score
                    >= threshold
                    for score
                    in cited_scores
                )
                / len(
                    cited_scores
                )
            )

        else:
            cited_above = 0.0

        if negative_scores:

            negative_below = (
                sum(
                    score
                    < threshold
                    for score
                    in negative_scores
                )
                / len(
                    negative_scores
                )
            )

        else:
            negative_below = 0.0

        print(
            f"{threshold:<12.2f}"
            f"{cited_above:<18.3f}"
            f"{negative_below:<18.3f}"
        )

    # ----------------------------------------
    # Lowest cited pairs
    # ----------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "LOWEST-SCORING CITED PAIRS"
    )

    print(
        "=" * 80
    )

    cited_sorted = sorted(
        cited_records,
        key=lambda record: (
            record["score"]
        ),
    )

    for record in cited_sorted[
        :15
    ]:

        print_record(
            record
        )

    # ----------------------------------------
    # Highest hard negatives
    # ----------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "HIGHEST-SCORING "
        "HARD NEGATIVES"
    )

    print(
        "=" * 80
    )

    negative_sorted = sorted(
        negative_records,
        key=lambda record: (
            record["score"]
        ),
        reverse=True,
    )

    for record in negative_sorted[
        :15
    ]:

        print_record(
            record
        )

    # ----------------------------------------
    # CSV
    # ----------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "type",
                "source",
                "page_number",
                "reference_number",
                "score",
                "evidence_text",
                "reference_text",
            ],
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    print()
    print(
        f"Saved calibration data to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()