import re
import statistics

from dataclasses import replace

from local_rag.ingestion.models import (
    DocumentBlock,
)


class BlockClassifier:

    NUMBERED_HEADING_PATTERN = re.compile(
        r"^\s*"
        r"\d+(?:\.\d+){0,4}"
        r"\s*"
        r"(?:\|\s*)?"
        r"[A-Z]"
    )

    TABLE_PATTERN = re.compile(
        r"^\s*Table\s+\d+",
        re.IGNORECASE,
    )

    FIGURE_PATTERN = re.compile(
        r"^\s*"
        r"(?:Figure|Fig\.)"
        r"\s*\d+",
        re.IGNORECASE,
    )

    KNOWN_HEADING_PATTERN = re.compile(
        r"^\s*"
        r"(?:"
        r"Abstract|"
        r"Introduction|"
        r"Background|"
        r"Methodology|"
        r"Methods?|"
        r"Results?|"
        r"Discussion|"
        r"Conclusion|"
        r"Conclusions|"
        r"Future Work|"
        r"Future Directions|"
        r"Challenges|"
        r"Outlook|"
        r"Outlooks"
        r")"
        r"\s*$",
        re.IGNORECASE,
    )

    BACK_MATTER_PATTERN = re.compile(
        r"^\s*"
        r"(?:"
        r"Acknowledg(?:e)?ments|"
        r"Data Availability|"
        r"Conflict of Interest|"
        r"Disclosure of Conflict of Interest|"
        r"Compliance with Ethical Standards|"
        r"Author Contributions"
        r")"
        r"\s*$",
        re.IGNORECASE,
    )

    FRONT_MATTER_PATTERN = re.compile(
        r"\b(?:"
        r"Correspondence|"
        r"Received|"
        r"Revised|"
        r"Accepted|"
        r"Funding|"
        r"Grant/Award|"
        r"Department of|"
        r"University|"
        r"Institute|"
        r"Laboratory|"
        r"Email"
        r")\b",
        re.IGNORECASE,
    )

    def classify(
        self,
        blocks: list[DocumentBlock],
    ) -> list[DocumentBlock]:

        body_font_size = (
            self._estimate_body_font_size(
                blocks
            )
        )

        result: list[
            DocumentBlock
        ] = []

        for block in blocks:

            content_type = (
                self._classify_block(
                    block=block,
                    body_font_size=(
                        body_font_size
                    ),
                )
            )

            result.append(
                replace(
                    block,
                    content_type=(
                        content_type
                    ),
                )
            )

        return result

    def _classify_block(
        self,
        block: DocumentBlock,
        body_font_size: float,
    ) -> str:

        text = block.content.strip()

        word_count = len(
            text.split()
        )

        if (
            self.TABLE_PATTERN.match(
                text
            )
        ):
            return "table"

        if (
            self.FIGURE_PATTERN.match(
                text
            )
        ):
            return "figure_caption"

        if (
            self.BACK_MATTER_PATTERN.match(
                text
            )
        ):
            return "back_matter"

        # First-page administrative metadata.

        if (
            block.page_number == 1
            and self.FRONT_MATTER_PATTERN.search(
                text
            )
        ):
            return "metadata"

        # Likely paper title.

        if (
            block.page_number == 1
            and block.font_size
            is not None
            and block.font_size
            >= body_font_size + 2.0
            and word_count <= 40
        ):
            return "title"

        if (
            self.NUMBERED_HEADING_PATTERN.match(
                text
            )
        ):
            return "heading"

        if (
            self.KNOWN_HEADING_PATTERN.match(
                text
            )
        ):
            return "heading"

        # Typography-based heading fallback.

        if (
            word_count <= 20
            and block.font_size
            is not None
            and (
                block.font_size
                >= body_font_size + 1.0
            )
        ):
            return "heading"

        if (
            word_count <= 20
            and block.is_bold
            and block.font_size
            is not None
            and block.font_size
            >= body_font_size
        ):
            return "heading"

        return "prose"

    @staticmethod
    def _estimate_body_font_size(
        blocks: list[DocumentBlock],
    ) -> float:

        sizes = [
            block.font_size
            for block in blocks
            if (
                block.font_size
                is not None
                and len(
                    block.content.split()
                )
                >= 5
            )
        ]

        if not sizes:

            return 10.0

        return float(
            statistics.median(
                sizes
            )
        )