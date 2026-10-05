import re

from local_rag.references.citation_normalizer import (
    CitationNormalizer,
)

from local_rag.retrieval.citation_extractor import (
    CitationExtractor,
)

from local_rag.retrieval.evidence import (
    Evidence,
)

from local_rag.retrieval.retriever import (
    RetrievedChunk,
)


class EvidenceBuilder:

    SENTENCE_SPLIT_PATTERN = re.compile(
        r"(?<=[.!?])\s+"
        r"(?=[A-Z0-9])"
    )

    NORMALIZED_CITATION_PATTERN = re.compile(
        r"\[[^]]+?\.pdf,\s*"
        r"ref\s+\d+\]",
        re.IGNORECASE,
    )

    # Handles cases such as:
    #
    # Sentence ends here. [paper.pdf, ref 24] Next sentence
    #
    # and rewrites them internally to:
    #
    # Sentence ends here [paper.pdf, ref 24]. Next sentence
    #
    # before sentence splitting.

    POST_PUNCTUATION_CITATION_PATTERN = re.compile(
        r"([.!?])"
        r"\s+"
        r"("
        r"(?:"
        r"\[[^]]+?\.pdf,\s*"
        r"ref\s+\d+\]"
        r"\s*"
        r")+"
        r")"
        r"(?=[A-Z0-9])",
        re.IGNORECASE,
    )

    def __init__(
        self,
        citation_normalizer: CitationNormalizer,
    ):

        self.citation_normalizer = (
            citation_normalizer
        )

        self.citation_extractor = (
            CitationExtractor()
        )

    def build(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[Evidence]:

        evidence_items: list[
            Evidence
        ] = []

        evidence_index = 1

        for chunk in chunks:

            normalized = (
                self.citation_normalizer
                .normalize(
                    text=chunk.content,
                    source=chunk.source,
                )
            )

            normalized = (
                self._normalize_citation_position(
                    normalized
                )
            )

            sentences = (
                self.SENTENCE_SPLIT_PATTERN
                .split(
                    normalized
                )
            )

            for sentence in sentences:

                sentence = (
                    sentence.strip()
                )

                if not sentence:
                    continue

                used_references = (
                    self.citation_extractor
                    .extract(
                        sentence
                    )
                )

                reference_numbers = [
                    reference.number
                    for reference
                    in used_references
                    if (
                        reference.source
                        == chunk.source
                    )
                ]

                # Preserve order while
                # removing duplicates.

                reference_numbers = list(
                    dict.fromkeys(
                        reference_numbers
                    )
                )

                clean_sentence = (
                    self.NORMALIZED_CITATION_PATTERN
                    .sub(
                        "",
                        sentence,
                    )
                )

                clean_sentence = re.sub(
                    r"\s+",
                    " ",
                    clean_sentence,
                ).strip()

                if not clean_sentence:
                    continue

                evidence_items.append(
                    Evidence(
                        id=(
                            f"E{evidence_index}"
                        ),
                        content=(
                            clean_sentence
                        ),
                        source=(
                            chunk.source
                        ),
                        page_number=(
                            chunk.page_number
                        ),
                        reference_numbers=(
                            reference_numbers
                        ),
                    )
                )

                evidence_index += 1

        return evidence_items

    def _normalize_citation_position(
        self,
        text: str,
    ) -> str:

        def replace(
            match: re.Match,
        ) -> str:

            punctuation = (
                match.group(1)
            )

            citations = (
                match.group(2)
                .strip()
            )

            return (
                f" {citations}"
                f"{punctuation} "
            )

        previous = None

        current = text

        while current != previous:

            previous = current

            current = (
                self.POST_PUNCTUATION_CITATION_PATTERN
                .sub(
                    replace,
                    current,
                )
            )

        return current