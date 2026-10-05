import math
import re

from collections import Counter

from pathlib import Path

import fitz

from local_rag.ingestion.cleaner import (
    TextCleaner,
)

from local_rag.ingestion.models import (
    DocumentBlock,
    DocumentPage,
)


class BlockPDFLoader:

    TOP_MARGIN_RATIO = 0.12

    BOTTOM_MARGIN_RATIO = 0.88

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
    ) -> None:

        self.cleaner = (
            cleaner
            or TextCleaner()
        )

    def load(
        self,
        file_path: str | Path,
    ) -> list[DocumentPage]:

        file_path = Path(
            file_path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"PDF not found: "
                f"{file_path}"
            )

        if (
            file_path.suffix.lower()
            != ".pdf"
        ):

            raise ValueError(
                f"File is not a PDF: "
                f"{file_path}"
            )

        raw_pages: list[dict] = []

        with fitz.open(
            file_path
        ) as document:

            for page_index, page in enumerate(
                document
            ):

                page_dict = page.get_text(
                    "dict",
                    sort=True,
                )

                page_height = (
                    page.rect.height
                )

                blocks: list[
                    DocumentBlock
                ] = []

                block_index = 0

                for raw_block in (
                    page_dict.get(
                        "blocks",
                        []
                    )
                ):

                    # PyMuPDF:
                    # type=0 → text block

                    if (
                        raw_block.get(
                            "type"
                        )
                        != 0
                    ):
                        continue

                    lines: list[str] = []

                    font_sizes: list[
                        float
                    ] = []

                    bold_chars = 0

                    total_chars = 0

                    for line in (
                        raw_block.get(
                            "lines",
                            []
                        )
                    ):

                        line_parts: list[
                            str
                        ] = []

                        for span in (
                            line.get(
                                "spans",
                                []
                            )
                        ):

                            span_text = (
                                span.get(
                                    "text",
                                    ""
                                )
                            )

                            if not span_text:
                                continue

                            line_parts.append(
                                span_text
                            )

                            size = span.get(
                                "size"
                            )

                            if size is not None:

                                font_sizes.append(
                                    float(size)
                                )

                            characters = len(
                                span_text
                            )

                            total_chars += (
                                characters
                            )

                            font_name = (
                                span.get(
                                    "font",
                                    ""
                                )
                                .lower()
                            )

                            flags = int(
                                span.get(
                                    "flags",
                                    0
                                )
                            )

                            is_bold = (
                                "bold"
                                in font_name
                                or "semibold"
                                in font_name
                                or "demi"
                                in font_name
                                or bool(
                                    flags
                                    & 16
                                )
                            )

                            if is_bold:

                                bold_chars += (
                                    characters
                                )

                        raw_line = (
                            " ".join(
                                line_parts
                            )
                            .strip()
                        )

                        if not raw_line:
                            continue

                        cleaned_line = (
                            self.cleaner.clean(
                                raw_line
                            )
                        )

                        if cleaned_line:

                            lines.append(
                                cleaned_line
                            )

                    if not lines:
                        continue

                    content = (
                        "\n".join(
                            lines
                        )
                        .strip()
                    )

                    if not content:
                        continue

                    bbox = raw_block.get(
                        "bbox",
                        (
                            None,
                            None,
                            None,
                            None,
                        ),
                    )

                    font_size = None

                    if font_sizes:

                        font_size = max(
                            font_sizes
                        )

                    is_bold = False

                    if total_chars:

                        is_bold = (
                            bold_chars
                            / total_chars
                            >= 0.50
                        )

                    blocks.append(
                        DocumentBlock(
                            content=content,
                            source=(
                                file_path.name
                            ),
                            page_number=(
                                page_index + 1
                            ),
                            block_index=(
                                block_index
                            ),
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3],
                            font_size=(
                                font_size
                            ),
                            is_bold=(
                                is_bold
                            ),
                        )
                    )

                    block_index += 1

                raw_pages.append(
                    {
                        "page_number": (
                            page_index + 1
                        ),
                        "height": (
                            page_height
                        ),
                        "blocks": blocks,
                    }
                )

        repeated_margin_blocks = (
            self._find_repeated_margin_blocks(
                raw_pages
            )
        )

        pages: list[
            DocumentPage
        ] = []

        for raw_page in raw_pages:

            page_height = (
                raw_page["height"]
            )

            filtered_blocks: list[
                DocumentBlock
            ] = []

            for block in (
                raw_page["blocks"]
            ):

                if self._is_repeated_margin_block(
                    block=block,
                    page_height=(
                        page_height
                    ),
                    repeated_signatures=(
                        repeated_margin_blocks
                    ),
                ):
                    continue

                filtered_blocks.append(
                    block
                )

            for index, block in enumerate(
                filtered_blocks
            ):

                block.block_index = index

            content = "\n\n".join(
                block.content
                for block
                in filtered_blocks
            ).strip()

            if not content:
                continue

            pages.append(
                DocumentPage(
                    content=content,
                    page_number=(
                        raw_page[
                            "page_number"
                        ]
                    ),
                    source=(
                        file_path.name
                    ),
                    blocks=(
                        filtered_blocks
                    ),
                )
            )

        return pages

    def _find_repeated_margin_blocks(
        self,
        pages: list[dict],
    ) -> set[str]:

        counter: Counter = (
            Counter()
        )

        for page in pages:

            height = page[
                "height"
            ]

            signatures = set()

            for block in page[
                "blocks"
            ]:

                if not self._is_margin_block(
                    block,
                    height,
                ):
                    continue

                signature = (
                    self._signature(
                        block.content
                    )
                )

                if signature:

                    signatures.add(
                        signature
                    )

            for signature in signatures:

                counter[
                    signature
                ] += 1

        if not pages:
            return set()

        minimum_occurrences = max(
            3,
            math.ceil(
                len(pages)
                * 0.30
            ),
        )

        return {
            signature
            for signature, count
            in counter.items()
            if (
                count
                >= minimum_occurrences
            )
        }

    def _is_repeated_margin_block(
        self,
        block: DocumentBlock,
        page_height: float,
        repeated_signatures: set[str],
    ) -> bool:

        if not self._is_margin_block(
            block,
            page_height,
        ):
            return False

        signature = self._signature(
            block.content
        )

        return (
            signature
            in repeated_signatures
        )

    def _is_margin_block(
        self,
        block: DocumentBlock,
        page_height: float,
    ) -> bool:

        if (
            block.y0 is None
            or block.y1 is None
        ):
            return False

        top_margin = (
            page_height
            * self.TOP_MARGIN_RATIO
        )

        bottom_margin = (
            page_height
            * self.BOTTOM_MARGIN_RATIO
        )

        return (
            block.y1
            <= top_margin
            or block.y0
            >= bottom_margin
        )

    @staticmethod
    def _signature(
        text: str,
    ) -> str:

        text = (
            text.lower()
        )

        text = re.sub(
            r"\d+",
            "<n>",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()