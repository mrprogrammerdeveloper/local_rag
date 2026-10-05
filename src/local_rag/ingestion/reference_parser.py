import re

from local_rag.ingestion.models import (
    DocumentPage,
    DocumentReference,
)


class ReferenceParser:

    REFERENCES_HEADING_PATTERN = re.compile(
        r"\bReferences\s*"
        r"(?=(?:\[\d+\]|\d+[.)]\s))",
        re.IGNORECASE,
    )

    BRACKET_REFERENCE_PATTERN = re.compile(
        r"\[(\d+)\]\s+"
        r"(.*?)"
        r"(?=\s*\[\d+\]\s+|\Z)",
        re.DOTALL,
    )

    PLAIN_REFERENCE_MARKER_PATTERN = re.compile(
        r"(?<!\d)"
        r"(?P<number>\d{1,3})"
        r"[.)]\s+"
    )

    def parse(
        self,
        pages: list[DocumentPage],
    ) -> list[DocumentReference]:

        if not pages:
            return []

        source = pages[0].source

        full_text = "\n".join(
            page.content
            for page in pages
        )

        heading_match = (
            self.REFERENCES_HEADING_PATTERN
            .search(full_text)
        )

        if heading_match is None:
            return []

        references_text = full_text[
            heading_match.end():
        ].strip()

        if not references_text:
            return []

        references = (
            self._parse_bracketed(
                references_text,
                source,
            )
        )

        if references:
            return references

        return self._parse_plain(
            references_text,
            source,
        )
    def _parse_bracketed(
        self,
        text: str,
        source: str,
    ) -> list[DocumentReference]:

        references = []

        for match in (
            self.BRACKET_REFERENCE_PATTERN
            .finditer(text)
        ):

            number = int(
                match.group(1)
            )

            content = (
                match.group(2)
                .strip()
            )

            if not content:
                continue

            references.append(
                DocumentReference(
                    number=number,
                    content=content,
                    source=source,
                )
            )

        return references
    def _parse_plain(
        self,
        text: str,
        source: str,
    ) -> list[DocumentReference]:

        candidates = list(
            self.PLAIN_REFERENCE_MARKER_PATTERN
            .finditer(text)
        )

        if not candidates:
            return []

        # Keep only a sequential reference chain:
        # 1, 2, 3, 4, ...
        markers = []

        expected_number = 1

        for match in candidates:

            number = int(
                match.group("number")
            )

            if number != expected_number:
                continue

            markers.append(
                match
            )

            expected_number += 1

        if not markers:
            return []

        references: list[
            DocumentReference
        ] = []

        for index, marker in enumerate(
            markers
        ):

            number = int(
                marker.group("number")
            )

            content_start = (
                marker.end()
            )

            if (
                index + 1
                < len(markers)
            ):
                content_end = (
                    markers[index + 1]
                    .start()
                )
            else:
                content_end = len(
                    text
                )

            raw_content = text[
                content_start:
                content_end
            ]

            content = (
                self._clean_reference_content(
                    raw_content
                )
            )
            if not content:
                continue

            references.append(
                DocumentReference(
                    number=number,
                    content=content,
                    source=source,
                )
            )

        return references
    @staticmethod
    def _clean_reference_content(
        content: str,
    ) -> str:

        content = content.strip()

        footer_patterns = [
            r"\s+\d+\s+of\s+\d+\s+"
            r"(?:Advanced Intelligent Discovery|"
            r"Global Journal.*?)?$",

            r"\s+\d+\s+of\s+\d+\s+"
            r"\d+,\s*\d+,\s*Downloaded from.*$",

            r"\s+\d+,\s*\d+,\s*Downloaded from.*$",
        ]

        for pattern in footer_patterns:
            content = re.sub(
                pattern,
                "",
                content,
                flags=re.IGNORECASE
                | re.DOTALL,
            )

        return content.strip()