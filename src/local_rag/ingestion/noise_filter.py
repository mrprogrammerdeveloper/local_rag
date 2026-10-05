import re

from dataclasses import replace

from local_rag.ingestion.models import (
    DocumentBlock,
)


class NoiseFilter:

    WILEY_DOWNLOAD_PATTERN = re.compile(
        r"^\s*"
        r"\d{6,}"
        r"\s*,\s*\d+"
        r"\s*,\s*Downloaded from\s+"
        r"https?://.*?"
        r"Creative Commons License"
        r"\s*",
        re.IGNORECASE
        | re.DOTALL,
    )

    GLOBAL_JOURNAL_PATTERN = re.compile(
        r"Global Journal of Engineering "
        r"and Technology Advances"
        r"\s*,?\s*"
        r"\d{4}"
        r"\s*,?\s*"
        r"\d+\s*\(\d+\)"
        r"\s*,?\s*"
        r"\d+\s*[–-]\s*\d+"
        r"\s*\d*",
        re.IGNORECASE,
    )

    STANDALONE_PAGE_PATTERN = re.compile(
        r"^\s*"
        r"(?:page\s+)?"
        r"\d+"
        r"(?:\s+of\s+\d+)?"
        r"\s*$",
        re.IGNORECASE,
    )

    def clean_blocks(
        self,
        blocks: list[DocumentBlock],
    ) -> list[DocumentBlock]:

        cleaned: list[
            DocumentBlock
        ] = []

        for block in blocks:

            content = (
                self.clean_text(
                    block.content
                )
            )

            if not content:
                continue

            cleaned.append(
                replace(
                    block,
                    content=content,
                )
            )

        return cleaned

    def clean_text(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        text = (
            self.WILEY_DOWNLOAD_PATTERN.sub(
                "",
                text,
            )
        )

        text = (
            self.GLOBAL_JOURNAL_PATTERN.sub(
                "",
                text,
            )
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        text = text.strip()

        if (
            self.STANDALONE_PAGE_PATTERN
            .fullmatch(
                text
            )
        ):
            return ""

        return text