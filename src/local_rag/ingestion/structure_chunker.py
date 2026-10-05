import re
from dataclasses import dataclass

from transformers import AutoTokenizer

from local_rag.config import (
    EMBEDDING_MODEL,
)

from local_rag.ingestion.models import (
    DocumentChunk,
    DocumentPage,
)


@dataclass
class TextUnit:

    content: str

    source: str

    page_start: int

    page_end: int

    section_title: str | None

    section_path: list[str]


class StructureAwareChunker:

    HEADING_PATTERNS = [

        # 1. Introduction
        # 2.3 Applications
        # 4.3.1 Example

        re.compile(
            r"^(?P<number>"
            r"\d+(?:\.\d+){0,3}"
            r")"
            r"[.)]?\s+"
            r"(?P<title>.+)$"
        ),

        # Abstract / Introduction /
        # Conclusion / Discussion etc.

        re.compile(
            r"^(?P<title>"
            r"Abstract|Introduction|"
            r"Background|Methods?|"
            r"Methodology|Results?|"
            r"Discussion|Conclusion|"
            r"Conclusions|"
            r"Future Work|"
            r"Future Directions|"
            r"Limitations"
            r")$",
            re.IGNORECASE,
        ),
    ]

    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+"
    )

    def __init__(
        self,
        target_tokens: int = 220,
        min_tokens: int = 100,
        max_tokens: int = 320,
        overlap_sentences: int = 1,
        max_overlap_tokens: int = 40,
        tokenizer_name: str = EMBEDDING_MODEL,
    ) -> None:

        if target_tokens <= 0:
            raise ValueError(
                "target_tokens must be > 0"
            )

        if min_tokens <= 0:
            raise ValueError(
                "min_tokens must be > 0"
            )

        if max_tokens <= target_tokens:
            raise ValueError(
                "max_tokens must be greater "
                "than target_tokens"
            )

        if min_tokens > target_tokens:
            raise ValueError(
                "min_tokens must be <= "
                "target_tokens"
            )

        self.target_tokens = (
            target_tokens
        )

        self.min_tokens = (
            min_tokens
        )

        self.max_tokens = (
            max_tokens
        )

        self.overlap_sentences = (
            overlap_sentences
        )

        self.max_overlap_tokens = (
            max_overlap_tokens
        )

        print(
            "Loading chunk tokenizer: "
            f"{tokenizer_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                tokenizer_name
            )
        )

    def split(
        self,
        pages: list[DocumentPage],
    ) -> list[DocumentChunk]:

        if not pages:
            return []

        units = (
            self._build_units(
                pages
            )
        )

        chunks = (
            self._pack_units(
                units
            )
        )

        return chunks

    def _build_units(
        self,
        pages: list[DocumentPage],
    ) -> list[TextUnit]:

        units: list[TextUnit] = []

        current_section: (
            str | None
        ) = None

        current_section_path: list[
            str
        ] = []

        for page in pages:

            paragraphs = [
                paragraph.strip()
                for paragraph
                in re.split(
                    r"\n\s*\n",
                    page.content,
                )
                if paragraph.strip()
            ]

            for paragraph in paragraphs:

                heading = (
                    self._detect_heading(
                        paragraph
                    )
                )

                if heading is not None:

                    current_section = (
                        heading
                    )

                    current_section_path = (
                        self._update_section_path(
                            heading=heading,
                            current_path=(
                                current_section_path
                            ),
                        )
                    )

                    continue

                sentence_units = (
                    self._split_paragraph(
                        paragraph
                    )
                )

                for sentence in (
                    sentence_units
                ):

                    units.append(
                        TextUnit(
                            content=sentence,
                            source=page.source,
                            page_start=(
                                page.page_number
                            ),
                            page_end=(
                                page.page_number
                            ),
                            section_title=(
                                current_section
                            ),
                            section_path=list(
                                current_section_path
                            ),
                        )
                    )

        return units

    def _detect_heading(
        self,
        text: str,
    ) -> str | None:

        text = text.strip()

        if not text:
            return None

        # Long paragraphs are almost
        # certainly not headings.

        if len(text.split()) > 18:
            return None

        for pattern in (
            self.HEADING_PATTERNS
        ):

            match = pattern.match(
                text
            )

            if not match:
                continue

            number = (
                match.groupdict()
                .get(
                    "number"
                )
            )

            title = (
                match.groupdict()
                .get(
                    "title"
                )
            )

            if title is None:
                continue

            title = title.strip()

            if number:

                return (
                    f"{number} {title}"
                )

            return title

        return None

    def _update_section_path(
        self,
        heading: str,
        current_path: list[str],
    ) -> list[str]:

        match = re.match(
            r"^(\d+(?:\.\d+)*)\s+",
            heading,
        )

        if match is None:

            return [
                heading
            ]

        number = match.group(1)

        level = (
            number.count(".")
            + 1
        )

        path = list(
            current_path[
                :level - 1
            ]
        )

        path.append(
            heading
        )

        return path

    def _split_paragraph(
        self,
        paragraph: str,
    ) -> list[str]:

        paragraph = paragraph.strip()

        if not paragraph:
            return []

        if (
            self._token_count(
                paragraph
            )
            <= self.max_tokens
        ):

            return [
                paragraph
            ]

        sentences = [
            sentence.strip()
            for sentence
            in self.SENTENCE_PATTERN.split(
                paragraph
            )
            if sentence.strip()
        ]

        units: list[str] = []

        for sentence in sentences:

            if (
                self._token_count(
                    sentence
                )
                <= self.max_tokens
            ):

                units.append(
                    sentence
                )

                continue

            units.extend(
                self._split_long_text(
                    sentence
                )
            )

        return units

    def _split_long_text(
        self,
        text: str,
    ) -> list[str]:

        token_ids = (
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

        pieces: list[str] = []

        start = 0

        while start < len(
            token_ids
        ):

            end = min(
                start
                + self.max_tokens,
                len(token_ids),
            )

            piece_ids = (
                token_ids[start:end]
            )

            piece = (
                self.tokenizer.decode(
                    piece_ids,
                    skip_special_tokens=True,
                )
                .strip()
            )

            if piece:

                pieces.append(
                    piece
                )

            start = end

        return pieces

    def _pack_units(
        self,
        units: list[TextUnit],
    ) -> list[DocumentChunk]:

        chunks: list[
            DocumentChunk
        ] = []

        current_units: list[
            TextUnit
        ] = []

        chunk_index = 0

        for unit in units:

            if not current_units:

                current_units.append(
                    unit
                )

                continue

            current_section = (
                current_units[-1]
                .section_title
            )

            section_changed = (
                unit.section_title
                != current_section
            )

            candidate_text = (
                self._join_units(
                    current_units
                    + [unit]
                )
            )

            candidate_tokens = (
                self._token_count(
                    candidate_text
                )
            )

            should_flush = (
                section_changed
                or candidate_tokens
                > self.max_tokens
            )

            if should_flush:

                chunk = (
                    self._create_chunk(
                        current_units,
                        chunk_index,
                    )
                )

                chunks.append(
                    chunk
                )

                chunk_index += 1

                overlap = (
                    self._build_overlap(
                        current_units
                    )
                )

                if section_changed:

                    current_units = []

                else:

                    overlap_with_unit = (
                        overlap
                        + [unit]
                    )

                    overlap_tokens = (
                        self._token_count(
                            self._join_units(
                                overlap_with_unit
                            )
                        )
                    )

                    if (
                        overlap_tokens
                        <= self.max_tokens
                    ):

                        current_units = (
                            overlap
                        )

                    else:

                        current_units = []

            current_units.append(
                unit
            )

            current_text = (
                self._join_units(
                    current_units
                )
            )

            if (
                self._token_count(
                    current_text
                )
                >= self.target_tokens
            ):

                chunk = (
                    self._create_chunk(
                        current_units,
                        chunk_index,
                    )
                )

                chunks.append(
                    chunk
                )

                chunk_index += 1

                current_units = (
                    self._build_overlap(
                        current_units
                    )
                )

        if current_units:

            chunks.append(
                self._create_chunk(
                    current_units,
                    chunk_index,
                )
            )

        return self._merge_small_chunks(
            chunks
        )

    def _build_overlap(
        self,
        units: list[TextUnit],
    ) -> list[TextUnit]:

        if (
            not units
            or self.overlap_sentences
            <= 0
        ):

            return []

        overlap = units[
            -self.overlap_sentences:
        ]

        overlap_text = (
            self._join_units(
                overlap
            )
        )

        if (
            self._token_count(
                overlap_text
            )
            <= self.max_overlap_tokens
        ):

            return list(
                overlap
            )

        return []

    def _create_chunk(
        self,
        units: list[TextUnit],
        chunk_index: int,
    ) -> DocumentChunk:

        content = (
            self._join_units(
                units
            )
        )

        page_start = min(
            unit.page_start
            for unit in units
        )

        page_end = max(
            unit.page_end
            for unit in units
        )

        section_title = (
            units[0]
            .section_title
        )

        section_path = list(
            units[0]
            .section_path
        )

        embedding_text = (
            self._build_embedding_text(
                content=content,
                section_path=(
                    section_path
                ),
            )
        )

        return DocumentChunk(
            content=content,
            page_number=page_start,
            page_end=page_end,
            source=(
                units[0].source
            ),
            chunk_index=chunk_index,
            section_title=(
                section_title
            ),
            section_path=(
                section_path
            ),
            content_type="paragraph",
            parent_id=None,
            token_count=(
                self._token_count(
                    content
                )
            ),
            embedding_text=(
                embedding_text
            ),
        )
    def _merge_small_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:

        if not chunks:
            return []

        result: list[
            DocumentChunk
        ] = []

        for chunk in chunks:

            if not result:

                result.append(
                    chunk
                )

                continue

            if (
                chunk.token_count
                is not None
                and chunk.token_count
                < self.min_tokens
            ):

                previous = (
                    result[-1]
                )

                same_section = (
                    previous.section_title
                    == chunk.section_title
                )

                combined = (
                    f"{previous.content} "
                    f"{chunk.content}"
                ).strip()

                combined_tokens = (
                    self._token_count(
                        combined
                    )
                )

                if (
                    same_section
                    and combined_tokens
                    <= self.max_tokens
                ):

                    previous.content = (
                        combined
                    )

                    previous.page_end = max(
                        previous.page_end
                        or previous.page_number,
                        chunk.page_end
                        or chunk.page_number,
                    )

                    previous.token_count = (
                        combined_tokens
                    )

                    previous.embedding_text = (
                        self._build_embedding_text(
                            content=combined,
                            section_path=(
                                previous
                                .section_path
                            ),
                        )
                    )

                    continue

            result.append(
                chunk
            )

        for index, chunk in enumerate(
            result
        ):

            chunk.chunk_index = (
                index
            )

        return result

    def _build_embedding_text(
        self,
        content: str,
        section_path: list[str],
    ) -> str:

        if not section_path:

            return content

        path = " > ".join(
            section_path
        )

        return (
            f"Section: {path}\n\n"
            f"{content}"
        )

    def _token_count(
        self,
        text: str,
    ) -> int:

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    @staticmethod
    def _join_units(
        units: list[TextUnit],
    ) -> str:

        return " ".join(
            unit.content.strip()
            for unit in units
            if unit.content.strip()
        )