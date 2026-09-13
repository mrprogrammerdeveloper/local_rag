import re

from local_rag.ingestion.models import (
    DocumentPage,
)


class DocumentSectionSplitter:
    REFERENCES_PATTERN = re.compile(
        r"\bReferences\s*(?=\[\d+\])",
        re.IGNORECASE,
    )

    def split_main_content(
        self,
        pages: list[DocumentPage],
    ) -> list[DocumentPage]:

        main_pages: list[DocumentPage] = []

        references_started = False

        for page in pages:

            if references_started:
                continue

            match = self.REFERENCES_PATTERN.search(
                page.content
            )

            if match is None:
                main_pages.append(page)
                continue

            references_started = True

            main_text = (
                page.content[:match.start()]
                .strip()
            )

            if main_text:
                main_pages.append(
                    DocumentPage(
                        content=main_text,
                        page_number=page.page_number,
                        source=page.source,
                    )
                )

        return main_pages