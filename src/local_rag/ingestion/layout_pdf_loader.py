import math
import re
from collections import Counter
from pathlib import Path

import fitz

from local_rag.ingestion.cleaner import (
    TextCleaner,
)

from local_rag.ingestion.models import (
    DocumentPage,
)


class LayoutPDFLoader:

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

        raw_pages = []

        with fitz.open(
            file_path
        ) as document:

            for index, page in enumerate(
                document
            ):

                page_height = (
                    page.rect.height
                )

                blocks = []

                for block in (
                    page.get_text(
                        "blocks",
                        sort=True,
                    )
                ):

                    if len(block) < 7:
                        continue

                    x0 = block[0]
                    y0 = block[1]
                    x1 = block[2]
                    y1 = block[3]

                    raw_text = block[4]

                    block_type = block[6]

                    # type 0 = text block
                    if block_type != 0:
                        continue

                    text = (
                        self.cleaner.clean(
                            raw_text
                        )
                    )

                    if not text:
                        continue

                    blocks.append(
                        {
                            "text": text,
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                        }
                    )

                raw_pages.append(
                    {
                        "page_number": (
                            index + 1
                        ),
                        "height": (
                            page_height
                        ),
                        "blocks": blocks,
                    }
                )

        repeated_margin_signatures = (
            self._find_repeated_margin_blocks(
                raw_pages
            )
        )

        pages: list[
            DocumentPage
        ] = []

        for page in raw_pages:

            height = page[
                "height"
            ]

            clean_blocks: list[
                str
            ] = []

            for block in page[
                "blocks"
            ]:

                text = block[
                    "text"
                ]

                if self._is_noise_block(
                    text
                ):
                    continue

                in_margin = (
                    block["y1"]
                    <= (
                        height
                        * self.TOP_MARGIN_RATIO
                    )
                    or block["y0"]
                    >= (
                        height
                        * self.BOTTOM_MARGIN_RATIO
                    )
                )

                if in_margin:

                    signature = (
                        self._margin_signature(
                            text
                        )
                    )

                    if (
                        signature
                        in
                        repeated_margin_signatures
                    ):
                        continue

                clean_blocks.append(
                    text
                )

            if not clean_blocks:
                continue

            # IMPORTANT:
            # preserve PDF block boundaries.
            content = (
                "\n\n".join(
                    clean_blocks
                )
                .strip()
            )

            if not content:
                continue

            pages.append(
                DocumentPage(
                    content=content,
                    page_number=page[
                        "page_number"
                    ],
                    source=(
                        file_path.name
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

            page_signatures = set()

            for block in page[
                "blocks"
            ]:

                in_margin = (
                    block["y1"]
                    <= (
                        height
                        * self.TOP_MARGIN_RATIO
                    )
                    or block["y0"]
                    >= (
                        height
                        * self.BOTTOM_MARGIN_RATIO
                    )
                )

                if not in_margin:
                    continue

                signature = (
                    self._margin_signature(
                        block["text"]
                    )
                )

                if not signature:
                    continue

                page_signatures.add(
                    signature
                )

            for signature in (
                page_signatures
            ):

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

    @staticmethod
    def _margin_signature(
        text: str,
    ) -> str:

        text = text.lower()

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

    @staticmethod
    def _is_noise_block(
        text: str,
    ) -> bool:

        normalized = (
            " ".join(
                text.split()
            )
            .strip()
        )

        if not normalized:
            return True

        patterns = [
            r"^\d+$",

            r"^page\s+\d+"
            r"(?:\s+of\s+\d+)?$",

            r"^\d+\s+of\s+\d+$",
        ]

        for pattern in patterns:

            if re.fullmatch(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                return True

        return False