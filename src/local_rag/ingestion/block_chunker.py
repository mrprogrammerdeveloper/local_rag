import hashlib
import re

from dataclasses import (
    dataclass,
)

from transformers import (
    AutoTokenizer,
)

from local_rag.config import (
    EMBEDDING_MODEL,
)

from local_rag.ingestion.models import (
    DocumentBlock,
    DocumentChunk,
)


@dataclass
class BlockUnit:

    content: str

    source: str

    page_number: int

    section_title: str | None

    section_path: list[str]


class BlockAwareChunker:

    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+"
        r"(?=[A-Z0-9])"
    )

    def __init__(
        self,
        target_tokens: int = 220,
        min_tokens: int = 100,
        max_tokens: int = 320,
        overlap_sentences: int = 1,
        max_overlap_tokens: int = 40,
        tokenizer_name: str = (
            EMBEDDING_MODEL
        ),
    ) -> None:

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
            "Loading block chunk "
            f"tokenizer: "
            f"{tokenizer_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                tokenizer_name
            )
        )

    def split(
        self,
        blocks: list[DocumentBlock],
    ) -> list[DocumentChunk]:

        searchable_blocks = [
            block
            for block in blocks
            if (
                block.content_type
                == "prose"
            )
        ]

        units = (
            self._build_units(
                searchable_blocks
            )
        )

        chunks = (
            self._pack_units(
                units
            )
        )

        chunks = (
            self._merge_small_chunks(
                chunks
            )
        )

        for index, chunk in enumerate(
            chunks
        ):

            chunk.chunk_index = (
                index
            )

        return chunks

    def _build_units(
        self,
        blocks: list[DocumentBlock],
    ) -> list[BlockUnit]:

        units: list[
            BlockUnit
        ] = []

        for block in blocks:

            sentences = [
                sentence.strip()
                for sentence
                in self.SENTENCE_PATTERN.split(
                    block.content
                )
                if sentence.strip()
            ]

            for sentence in sentences:

                if (
                    self._token_count(
                        sentence
                    )
                    <= self.max_tokens
                ):

                    units.append(
                        BlockUnit(
                            content=sentence,
                            source=block.source,
                            page_number=(
                                block.page_number
                            ),
                            section_title=(
                                block.section_title
                            ),
                            section_path=list(
                                block.section_path
                            ),
                        )
                    )

                    continue

                pieces = (
                    self._split_long_text(
                        sentence
                    )
                )

                for piece in pieces:

                    units.append(
                        BlockUnit(
                            content=piece,
                            source=block.source,
                            page_number=(
                                block.page_number
                            ),
                            section_title=(
                                block.section_title
                            ),
                            section_path=list(
                                block.section_path
                            ),
                        )
                    )

        return units

    def _pack_units(
        self,
        units: list[BlockUnit],
    ) -> list[DocumentChunk]:

        chunks: list[
            DocumentChunk
        ] = []

        current: list[
            BlockUnit
        ] = []

        chunk_index = 0

        for unit in units:

            if current:

                section_changed = (
                    current[-1].section_path
                    != unit.section_path
                )

                if section_changed:

                    chunks.append(
                        self._create_chunk(
                            units=current,
                            chunk_index=(
                                chunk_index
                            ),
                        )
                    )

                    chunk_index += 1

                    current = []

            if current:

                candidate = (
                    current
                    + [unit]
                )

                candidate_tokens = (
                    self._token_count(
                        self._join_units(
                            candidate
                        )
                    )
                )

                if (
                    candidate_tokens
                    > self.max_tokens
                ):

                    chunks.append(
                        self._create_chunk(
                            units=current,
                            chunk_index=(
                                chunk_index
                            ),
                        )
                    )

                    chunk_index += 1

                    overlap = (
                        self._build_overlap(
                            current
                        )
                    )

                    combined = (
                        overlap
                        + [unit]
                    )

                    if (
                        self._token_count(
                            self._join_units(
                                combined
                            )
                        )
                        <= self.max_tokens
                    ):

                        current = (
                            overlap
                        )

                    else:

                        current = []

            current.append(
                unit
            )

            current_tokens = (
                self._token_count(
                    self._join_units(
                        current
                    )
                )
            )

            if (
                current_tokens
                >= self.target_tokens
            ):

                chunks.append(
                    self._create_chunk(
                        units=current,
                        chunk_index=(
                            chunk_index
                        ),
                    )
                )

                chunk_index += 1

                current = (
                    self._build_overlap(
                        current
                    )
                )

        if current:

            chunks.append(
                self._create_chunk(
                    units=current,
                    chunk_index=(
                        chunk_index
                    ),
                )
            )

        return chunks

    def _build_overlap(
        self,
        units: list[BlockUnit],
    ) -> list[BlockUnit]:

        if (
            not units
            or self.overlap_sentences
            <= 0
        ):

            return []

        overlap = units[
            -self.overlap_sentences:
        ]

        tokens = (
            self._token_count(
                self._join_units(
                    overlap
                )
            )
        )

        if (
            tokens
            > self.max_overlap_tokens
        ):

            return []

        return list(
            overlap
        )

    def _create_chunk(
        self,
        units: list[BlockUnit],
        chunk_index: int,
    ) -> DocumentChunk:

        content = (
            self._join_units(
                units
            )
        )

        source = (
            units[0].source
        )

        page_start = min(
            unit.page_number
            for unit in units
        )

        page_end = max(
            unit.page_number
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

        token_count = (
            self._token_count(
                content
            )
        )

        parent_id = (
            self._build_parent_id(
                source=source,
                section_path=(
                    section_path
                ),
            )
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
            page_number=(
                page_start
            ),
            page_end=(
                page_end
            ),
            source=source,
            chunk_index=(
                chunk_index
            ),
            section_title=(
                section_title
            ),
            section_path=(
                section_path
            ),
            content_type=(
                "paragraph"
            ),
            parent_id=(
                parent_id
            ),
            token_count=(
                token_count
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
                is None
                or chunk.token_count
                >= self.min_tokens
            ):

                result.append(
                    chunk
                )

                continue

            previous = (
                result[-1]
            )

            same_section = (
                previous.section_path
                == chunk.section_path
            )

            if not same_section:

                result.append(
                    chunk
                )

                continue

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
                combined_tokens
                > self.max_tokens
            ):

                result.append(
                    chunk
                )

                continue

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

        return result

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

            piece = (
                self.tokenizer.decode(
                    token_ids[
                        start:end
                    ],
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

    def _build_parent_id(
        self,
        source: str,
        section_path: list[str],
    ) -> str | None:

        if not section_path:
            return None

        raw = (
            source
            + "::"
            + " > ".join(
                section_path
            )
        )

        digest = hashlib.sha1(
            raw.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            f"P-{digest[:12]}"
        )

    @staticmethod
    def _build_embedding_text(
        content: str,
        section_path: list[str],
    ) -> str:

        if not section_path:
            return content

        section = " > ".join(
            section_path
        )

        return (
            f"Section: {section}"
            f"\n\n"
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
        units: list[BlockUnit],
    ) -> str:

        return " ".join(
            unit.content.strip()
            for unit in units
            if unit.content.strip()
        )