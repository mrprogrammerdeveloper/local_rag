from pathlib import Path

from local_rag.config import (
    PDF_PATH,
    REFERENCES_PATH,
)

from local_rag.embeddings.embedder import (
    Embedder,
)

from local_rag.ingestion.chunker import (
    TextChunker,
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

from local_rag.references.store import (
    ReferenceStore,
)

from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)


def main() -> None:

    pdf_directory = Path(
        PDF_PATH
    )

    pdf_files = sorted(
        pdf_directory.glob("*.pdf")
    )

    if not pdf_files:
        print(
            f"No PDF files found in "
            f"{pdf_directory}"
        )
        return

    print(
        f"Found {len(pdf_files)} PDF file(s)."
    )

    loader = PDFLoader()

    chunker = TextChunker()

    reference_parser = (
        ReferenceParser()
    )

    section_splitter = (
        DocumentSectionSplitter()
    )

    reference_store = (
        ReferenceStore(
            REFERENCES_PATH
        )
    )

    print(
        "Loading embedding model..."
    )

    embedder = Embedder()

    vector_store = (
        QdrantVectorStore(
            vector_size=embedder.dimension
        )
    )

    for pdf_file in pdf_files:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Processing: {pdf_file.name}"
        )

        print(
            "=" * 70
        )

        all_pages = loader.load(
            pdf_file
        )

        print(
            f"Total text pages: "
            f"{len(all_pages)}"
        )

        references = (
            reference_parser.parse(
                all_pages
            )
        )

        reference_store.save(
            source=pdf_file.name,
            references=references,
        )

        print(
            f"References parsed: "
            f"{len(references)}"
        )

        main_pages = (
            section_splitter
            .split_main_content(
                all_pages
            )
        )

        print(
            f"Main-content pages: "
            f"{len(main_pages)}"
        )

        excluded_pages = (
            len(all_pages)
            - len(main_pages)
        )

        print(
            f"Reference-only pages excluded: "
            f"{excluded_pages}"
        )

        chunks = (
            chunker.split(
                main_pages
            )
        )

        print(
            f"Chunks created: "
            f"{len(chunks)}"
        )

        if not chunks:
            print(
                "No chunks created. Skipping."
            )
            continue

        texts = [
            chunk.content
            for chunk in chunks
        ]

        embeddings = (
            embedder.embed_texts(
                texts
            )
        )

        vector_store.upsert(
            chunks=chunks,
            vectors=embeddings,
        )

        print(
            f"Indexed {len(chunks)} chunks."
        )

    print(
        "\nIngestion completed."
    )


if __name__ == "__main__":
    main()