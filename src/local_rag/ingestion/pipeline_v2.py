from pathlib import Path

from local_rag.ingestion.base import (
    BaseIngestionPipeline,
)

from local_rag.ingestion.layout_pdf_loader import (
    LayoutPDFLoader,
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

from local_rag.ingestion.structure_chunker import (
    StructureAwareChunker,
)


class StructureAwareIngestionPipeline(
    BaseIngestionPipeline
):

    name = "v2-basic"

    description = (
        "Hierarchical structure-aware "
        "token-based chunking"
    )

    def __init__(
        self,
    ) -> None:

        # Layout-aware representation is used
        # for scientific body text and chunking.
        self.content_loader = (
            LayoutPDFLoader()
        )

        # Legacy page text extraction is kept
        # specifically for bibliography parsing.
        #
        # ReferenceParser was validated against
        # this representation and should remain
        # independent from chunking improvements.
        self.reference_loader = (
            PDFLoader()
        )

        self.reference_parser = (
            ReferenceParser()
        )

        self.section_splitter = (
            DocumentSectionSplitter()
        )

        self.chunker = (
            StructureAwareChunker(
                target_tokens=220,
                min_tokens=100,
                max_tokens=320,
                overlap_sentences=1,
                max_overlap_tokens=40,
            )
        )

    def ingest(
        self,
        file_path: str | Path,
    ) -> IngestionResult:

        file_path = Path(
            file_path
        )

        # -------------------------------------------------
        # Content representation
        # -------------------------------------------------

        pages = (
            self.content_loader.load(
                file_path
            )
        )

        # -------------------------------------------------
        # Bibliography representation
        # -------------------------------------------------

        reference_pages = (
            self.reference_loader.load(
                file_path
            )
        )

        references = (
            self.reference_parser.parse(
                reference_pages
            )
        )

        # -------------------------------------------------
        # Main scientific content
        # -------------------------------------------------

        main_pages = (
            self.section_splitter
            .split_main_content(
                pages
            )
        )

        chunks = (
            self.chunker.split(
                main_pages
            )
        )

        return IngestionResult(
            source=file_path.name,
            pages=pages,
            main_pages=main_pages,
            references=references,
            chunks=chunks,
            pipeline_name=self.name,
        )
