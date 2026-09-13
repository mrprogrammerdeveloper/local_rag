from pathlib import Path

from local_rag.ingestion.chunker import TextChunker
from local_rag.ingestion.pdf_loader import PDFLoader


pdf_path = Path(
    "data/pdfs/GJETA-2025-0260 (1).pdf"
)

loader = PDFLoader()

pages = loader.load(
    pdf_path
)

chunker = TextChunker(
    chunk_size=500,
    overlap=75,
)

chunks = chunker.split(
    pages
)


print(
    f"Pages: {len(pages)}"
)

print(
    f"Chunks: {len(chunks)}"
)


for chunk in chunks[:5]:

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Page: {chunk.page_number}"
    )

    print(
        f"Chunk: {chunk.chunk_index}"
    )

    print(
        f"Words: "
        f"{len(chunk.content.split())}"
    )

    print()

    print(
        chunk.content
    )