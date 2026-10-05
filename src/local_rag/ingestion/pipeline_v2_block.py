import re

from pathlib import Path

from local_rag.ingestion.base import (
    BaseIngestionPipeline,
)

from local_rag.ingestion.block_chunker import (
    BlockAwareChunker,
)

from local_rag.ingestion.block_classifier import (
    BlockClassifier,
)

from local_rag.ingestion.block_pdf_loader import (
    BlockPDFLoader,
)

from local_rag.ingestion.inline_heading_splitter import (
    InlineHeadingSplitter,
)

from local_rag.ingestion.models import (
    DocumentBlock,
    IngestionResult,
)

from local_rag.ingestion.noise_filter import (
    NoiseFilter,
)

from local_rag.ingestion.reference_parser import (
    ReferenceParser,
)

from local_rag.ingestion.section_splitter import (
    DocumentSectionSplitter,
)

from local_rag.ingestion.section_tree import (
    SectionTreeBuilder,
)


class BlockAwareIngestionPipeline(
    BaseIngestionPipeline
):

    name = "v2-block"

    description = (
        "Block-aware structure-aware "
        "token chunking"
    )

    REFERENCES_PATTERN = re.compile(
        r"^\s*References\s*$",
        re.IGNORECASE,
    )

    def __init__(
        self,
    ) -> None:

        self.loader = (
            BlockPDFLoader()
        )

        self.reference_parser = (
            ReferenceParser()
        )

        self.section_splitter = (
            DocumentSectionSplitter()
        )

        self.noise_filter = (
            NoiseFilter()
        )

        self.inline_splitter = (
            InlineHeadingSplitter()
        )

        self.classifier = (
            BlockClassifier()
        )

        self.section_tree = (
            SectionTreeBuilder()
        )

        self.chunker = (
            BlockAwareChunker(
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

        main_page_numbers = {
            page.page_number
            for page in main_pages
        }

        blocks: list[
            DocumentBlock
        ] = []

        for page in pages:

            if (
                page.page_number
                not in
                main_page_numbers
            ):
                continue

            blocks.extend(
                page.blocks
            )

        blocks = (
            self.noise_filter
            .clean_blocks(
                blocks
            )
        )

        blocks = (
            self.inline_splitter
            .split_blocks(
                blocks
            )
        )

        blocks = (
            self._truncate_at_references(
                blocks
            )
        )

        blocks = (
            self.classifier
            .classify(
                blocks
            )
        )

        blocks = (
            self.section_tree
            .apply(
                blocks
            )
        )

        chunks = (
            self.chunker.split(
                blocks
            )
        )

        return IngestionResult(
            source=file_path.name,
            pages=pages,
            main_pages=main_pages,
            references=references,
            chunks=chunks,
            pipeline_name=(
                self.name
            ),
            blocks=blocks,
        )

    def _truncate_at_references(
        self,
        blocks: list[DocumentBlock],
    ) -> list[DocumentBlock]:

        result: list[
            DocumentBlock
        ] = []

        for block in blocks:

            if (
                self.REFERENCES_PATTERN
                .match(
                    block.content
                )
            ):

                break

            result.append(
                block
            )

        return result