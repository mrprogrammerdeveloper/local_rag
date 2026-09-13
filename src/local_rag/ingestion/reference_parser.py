import re

from local_rag.ingestion.models import (
    DocumentPage,
    DocumentReference,
)


class ReferenceParser:
    REFERENCE_HEADING_PATTERN = re.compile(
        r"\bReferences\s*(?=\[\d+\])",
        re.IGNORECASE,
    )

    REFERENCE_PATTERN = re.compile(
        r"\[(\d+)\]\s+(.*?)(?=\s*\[\d+\]\s+|\Z)",
        re.DOTALL,
    )

    def parse(
        self,
        pages: list[DocumentPage],
    ) -> list[DocumentReference]:

        if not pages:
            return []

        full_text = "\n".join(
            page.content
            for page in pages
        )

        heading_match = (
            self.REFERENCE_HEADING_PATTERN.search(
                full_text
            )
        )

        if not heading_match:
            return []

        reference_text = full_text[
            heading_match.end():
        ]

        references: list[
            DocumentReference
        ] = []

        source = pages[0].source

        for match in self.REFERENCE_PATTERN.finditer(
            reference_text
        ):
            number = int(
                match.group(1)
            )

            content = re.sub(
                r"\s+",
                " ",
                match.group(2),
            ).strip()

            references.append(
                DocumentReference(
                    number=number,
                    content=content,
                    source=source,
                )
            )

        return references