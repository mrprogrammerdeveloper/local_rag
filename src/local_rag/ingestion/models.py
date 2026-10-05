from dataclasses import (
    dataclass,
    field,
)


@dataclass
class DocumentBlock:

    content: str

    source: str

    page_number: int

    block_index: int

    content_type: str = "unknown"

    section_title: str | None = None

    section_path: list[str] = field(
        default_factory=list
    )

    x0: float | None = None

    y0: float | None = None

    x1: float | None = None

    y1: float | None = None

    font_size: float | None = None

    is_bold: bool = False


@dataclass
class DocumentPage:

    content: str

    page_number: int

    source: str

    blocks: list[
        DocumentBlock
    ] = field(
        default_factory=list
    )


@dataclass
class DocumentChunk:

    content: str

    page_number: int

    source: str

    chunk_index: int

    # -----------------------------
    # V2 metadata
    # -----------------------------

    page_end: int | None = None

    section_title: str | None = None

    section_path: list[str] = field(
        default_factory=list
    )

    content_type: str = "paragraph"

    parent_id: str | None = None

    token_count: int | None = None

    embedding_text: str | None = None


@dataclass
class DocumentReference:

    number: int

    content: str

    source: str


@dataclass
class IngestionResult:

    source: str

    pages: list[
        DocumentPage
    ]

    main_pages: list[
        DocumentPage
    ]

    references: list[
        DocumentReference
    ]

    chunks: list[
        DocumentChunk
    ]

    pipeline_name: str

    blocks: list[
        DocumentBlock
    ] = field(
        default_factory=list
    )