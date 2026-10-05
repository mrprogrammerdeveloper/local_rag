from pathlib import Path

from local_rag.ingestion.base import (
    BaseIngestionPipeline,
)

from local_rag.ingestion.chunker import (
    TextChunker,
)

from local_rag.ingestion.models import (
    IngestionResult,
)

from local_rag.ingestion.pdf_loader import (
    PDFLoader,
)

from local_rag.ingestion.reference_parser import (
    ReferenceParser,
)

from local_rag.ingestion.section_splitter import (
    DocumentSectionSplitter,
)


class LegacyIngestionPipeline(
    BaseIngestionPipeline
):

    name = "v1"

    description = (
        "Legacy paragraph-aware "
        "word-based chunking"
    )

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 75,
    ) -> None:

        self.loader = PDFLoader()

        self.reference_parser = (
            ReferenceParser()
        )

        self.section_splitter = (
            DocumentSectionSplitter()
        )

        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def ingest(
        self,
        file_path: str | Path,
    ) -> IngestionResult:

        file_path = Path(
            file_path
        )

        pages = self.loader.load(
            file_path
        )

        references = (
            self.reference_parser.parse(
                pages
            )
        )

        main_pages = (
            self.section_splitter
            .split_main_content(
                pages
            )
        )

        chunks = self.chunker.split(
            main_pages
        )

        return IngestionResult(
            source=file_path.name,
            pages=pages,
            main_pages=main_pages,
            references=references,
            chunks=chunks,
            pipeline_name=self.name,
        )