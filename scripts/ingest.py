import argparse
from pathlib import Path

from local_rag.config import (
    PDF_PATH,
    REFERENCES_PATH,
)

from local_rag.embeddings.embedder import (
    Embedder,
)

from local_rag.ingestion.pipelines import (
    register_ingestion_pipelines,
)

from local_rag.ingestion.selector import (
    select_ingestion,
)

from local_rag.references.store import (
    ReferenceStore,
)

from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Ingest PDF documents into "
            "the local RAG vector store."
        )
    )

    parser.add_argument(
        "--ingestion",
        type=str,
        default=None,
        help=(
            "Ingestion pipeline to use. "
            "Example: v1 or v2. "
            "If omitted, INGESTION_PIPELINE "
            "or the interactive selector is used."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    # -----------------------------------------
    # Register available ingestion pipelines
    # -----------------------------------------

    register_ingestion_pipelines()

    # -----------------------------------------
    # Select ingestion strategy
    # -----------------------------------------

    ingestion = select_ingestion(
        requested=args.ingestion
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"Ingestion pipeline: "
        f"{ingestion.name}"
    )

    print(
        f"Description: "
        f"{ingestion.description}"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------
    # Discover PDFs
    # -----------------------------------------

    pdf_directory = Path(
        PDF_PATH
    )

    pdf_files = sorted(
        pdf_directory.glob(
            "*.pdf"
        )
    )

    if not pdf_files:

        print(
            f"No PDF files found in "
            f"{pdf_directory}"
        )

        return

    print(
        f"Found "
        f"{len(pdf_files)} "
        f"PDF file(s)."
    )

    # -----------------------------------------
    # Shared services
    # -----------------------------------------

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
            vector_size=(
                embedder.dimension
            )
        )
    )

    total_chunks = 0

    # -----------------------------------------
    # Process documents
    # -----------------------------------------

    for pdf_file in pdf_files:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Processing: "
            f"{pdf_file.name}"
        )

        print(
            f"Pipeline: "
            f"{ingestion.name}"
        )

        print(
            "=" * 70
        )

        # -------------------------------------
        # Ingestion strategy handles:
        #
        # PDF loading
        # reference parsing
        # bibliography exclusion
        # chunking
        # -------------------------------------

        result = ingestion.ingest(
            pdf_file
        )

        print(
            f"Total text pages: "
            f"{len(result.pages)}"
        )

        print(
            f"References parsed: "
            f"{len(result.references)}"
        )

        print(
            f"Main-content pages: "
            f"{len(result.main_pages)}"
        )

        excluded_pages = (
            len(result.pages)
            - len(result.main_pages)
        )

        print(
            f"Reference-only pages excluded: "
            f"{excluded_pages}"
        )

        print(
            f"Chunks created: "
            f"{len(result.chunks)}"
        )

        # -------------------------------------
        # Store references
        # -------------------------------------

        reference_store.save(
            source=result.source,
            references=(
                result.references
            ),
        )

        # -------------------------------------
        # Skip empty documents
        # -------------------------------------

        if not result.chunks:

            print(
                "No chunks created. "
                "Skipping."
            )

            continue

        # -------------------------------------
        # Build embedding inputs
        #
        # V1:
        # embedding_text is None
        # → use content
        #
        # V2:
        # embedding_text may contain:
        #
        # Section: ...
        # Subsection: ...
        # + original chunk content
        # -------------------------------------

        texts = [
            (
                chunk.embedding_text
                if chunk.embedding_text
                else chunk.content
            )
            for chunk
            in result.chunks
        ]

        # -------------------------------------
        # Embedding
        # -------------------------------------

        embeddings = (
            embedder.embed_texts(
                texts
            )
        )

        # -------------------------------------
        # Vector storage
        # -------------------------------------

        vector_store.upsert(
            chunks=result.chunks,
            vectors=embeddings,
        )

        total_chunks += len(
            result.chunks
        )

        print(
            f"Indexed "
            f"{len(result.chunks)} "
            f"chunks."
        )

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "INGESTION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Pipeline: "
        f"{ingestion.name}"
    )

    print(
        f"Documents: "
        f"{len(pdf_files)}"
    )

    print(
        f"Total indexed chunks: "
        f"{total_chunks}"
    )

    print(
        "=" * 70
    )

    print(
        "\nIngestion completed."
    )


if __name__ == "__main__":
    main()