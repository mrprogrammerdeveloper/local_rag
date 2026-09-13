from pathlib import Path

import fitz

from local_rag.ingestion.cleaner import TextCleaner
from local_rag.ingestion.models import DocumentPage


class PDFLoader:
    def __init__(
        self,
        cleaner: TextCleaner | None = None,
    ):
        self.cleaner = cleaner or TextCleaner()

    def load(
        self,
        file_path: str | Path,
    ) -> list[DocumentPage]:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"File is not a PDF: {file_path}"
            )

        pages: list[DocumentPage] = []

        with fitz.open(file_path) as document:
            for index, page in enumerate(document):
                raw_text = page.get_text(
                    "text"
                )

                text = self.cleaner.clean(
                    raw_text
                )

                if not text:
                    continue

                pages.append(
                    DocumentPage(
                        content=text,
                        page_number=index + 1,
                        source=file_path.name,
                    )
                )

        return pages