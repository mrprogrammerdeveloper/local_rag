import re

from local_rag.ingestion.models import (
    DocumentChunk,
    DocumentPage,
)


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 75,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        pages: list[DocumentPage],
    ) -> list[DocumentChunk]:

        chunks: list[DocumentChunk] = []

        chunk_index = 0

        for page in pages:
            page_chunks = self._split_page(
                page.content
            )

            for content in page_chunks:
                chunks.append(
                    DocumentChunk(
                        content=content,
                        page_number=page.page_number,
                        source=page.source,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

        return chunks

    def _split_page(
        self,
        text: str,
    ) -> list[str]:

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                text,
            )
            if paragraph.strip()
        ]

        units: list[str] = []

        for paragraph in paragraphs:
            paragraph_words = (
                paragraph.split()
            )

            if (
                len(paragraph_words)
                <= self.chunk_size
            ):
                units.append(
                    paragraph
                )
                continue

            units.extend(
                self._split_long_paragraph(
                    paragraph
                )
            )

        return self._pack_units(
            units
        )

    def _split_long_paragraph(
        self,
        paragraph: str,
    ) -> list[str]:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph,
        )

        units: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            words = sentence.split()

            if len(words) <= self.chunk_size:
                units.append(
                    sentence
                )
                continue

            start = 0

            while start < len(words):
                end = (
                    start
                    + self.chunk_size
                )

                units.append(
                    " ".join(
                        words[start:end]
                    )
                )

                start = end

        return units

    def _pack_units(
        self,
        units: list[str],
    ) -> list[str]:

        chunks: list[str] = []

        current_words: list[str] = []

        for unit in units:
            unit_words = unit.split()

            if not unit_words:
                continue

            if (
                current_words
                and len(current_words)
                + len(unit_words)
                > self.chunk_size
            ):
                chunks.append(
                    " ".join(
                        current_words
                    )
                )

                overlap_words = (
                    current_words[
                        -self.overlap:
                    ]
                    if self.overlap
                    else []
                )

                current_words = (
                    overlap_words.copy()
                )

            current_words.extend(
                unit_words
            )

            while (
                len(current_words)
                > self.chunk_size
            ):
                chunks.append(
                    " ".join(
                        current_words[
                            :self.chunk_size
                        ]
                    )
                )

                if self.overlap:
                    current_words = (
                        current_words[
                            self.chunk_size
                            - self.overlap:
                        ]
                    )
                else:
                    current_words = (
                        current_words[
                            self.chunk_size:
                        ]
                    )

        if current_words:
            chunks.append(
                " ".join(
                    current_words
                )
            )

        return chunks