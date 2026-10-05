import argparse
import statistics
from collections import Counter
from pathlib import Path

from local_rag.config import (
    PDF_PATH,
)

from local_rag.ingestion.pipelines import (
    register_ingestion_pipelines,
)

from local_rag.ingestion.registry import (
    IngestionRegistry,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ingestion",
        type=str,
        default="v2",
    )

    return parser.parse_args()


def print_chunk(
    chunk,
) -> None:

    print()
    print("-" * 80)

    print(
        f"Chunk: {chunk.chunk_index}"
    )

    print(
        f"Source: {chunk.source}"
    )

    print(
        f"Pages: "
        f"{chunk.page_number}"
        f" -> "
        f"{chunk.page_end}"
    )

    print(
        f"Tokens: "
        f"{chunk.token_count}"
    )

    print(
        f"Section: "
        f"{chunk.section_title}"
    )

    print(
        f"Section path: "
        f"{chunk.section_path}"
    )

    print(
        f"Content type: "
        f"{chunk.content_type}"
    )

    print()

    preview = (
        chunk.content[:500]
        .replace(
            "\n",
            " ",
        )
    )

    print(
        preview
    )


def main() -> None:

    args = parse_args()

    register_ingestion_pipelines()

    pipeline = (
        IngestionRegistry.create(
            args.ingestion
        )
    )

    pdf_directory = Path(
        PDF_PATH
    )

    pdf_files = sorted(
        pdf_directory.glob(
            "*.pdf"
        )
    )

    print()
    print("=" * 80)
    print(
        f"INGESTION INSPECTION: "
        f"{pipeline.name}"
    )
    print("=" * 80)

    for pdf_file in pdf_files:

        print()
        print("=" * 80)
        print(
            f"DOCUMENT: "
            f"{pdf_file.name}"
        )
        print("=" * 80)

        result = pipeline.ingest(
            pdf_file
        )

        chunks = (
            result.chunks
        )

        if not chunks:

            print(
                "No chunks."
            )

            continue

        token_counts = [
            chunk.token_count
            for chunk in chunks
            if chunk.token_count
            is not None
        ]

        sectionless = [
            chunk
            for chunk in chunks
            if not chunk.section_title
        ]

        cross_page = [
            chunk
            for chunk in chunks
            if (
                chunk.page_end
                is not None
                and chunk.page_end
                != chunk.page_number
            )
        ]

        too_small = [
            chunk
            for chunk in chunks
            if (
                chunk.token_count
                is not None
                and chunk.token_count
                < 100
            )
        ]

        too_large = [
            chunk
            for chunk in chunks
            if (
                chunk.token_count
                is not None
                and chunk.token_count
                > 320
            )
        ]

        empty_source = [
            chunk
            for chunk in chunks
            if not chunk.source
        ]

        normalized_contents = [
            " ".join(
                chunk.content
                .lower()
                .split()
            )
            for chunk in chunks
        ]

        content_counter = Counter(
            normalized_contents
        )

        exact_duplicates = sum(
            count - 1
            for count
            in content_counter.values()
            if count > 1
        )

        sections = {
            chunk.section_title
            for chunk in chunks
            if chunk.section_title
        }

        print()
        print(
            f"Pages: "
            f"{len(result.main_pages)}"
        )

        print(
            f"Chunks: "
            f"{len(chunks)}"
        )

        print(
            f"Detected sections: "
            f"{len(sections)}"
        )

        print(
            f"Sectionless chunks: "
            f"{len(sectionless)}"
        )

        print(
            f"Cross-page chunks: "
            f"{len(cross_page)}"
        )

        print(
            f"Chunks < 100 tokens: "
            f"{len(too_small)}"
        )

        print(
            f"Chunks > 320 tokens: "
            f"{len(too_large)}"
        )

        print(
            f"Empty source: "
            f"{len(empty_source)}"
        )

        print(
            f"Exact duplicate chunks: "
            f"{exact_duplicates}"
        )

        if token_counts:

            print()
            print(
                "TOKEN DISTRIBUTION"
            )

            print(
                "-" * 40
            )

            print(
                f"Min: "
                f"{min(token_counts)}"
            )

            print(
                f"Median: "
                f"{statistics.median(token_counts):.1f}"
            )

            print(
                f"Mean: "
                f"{statistics.mean(token_counts):.1f}"
            )

            print(
                f"Max: "
                f"{max(token_counts)}"
            )

        print()
        print(
            "DETECTED SECTIONS"
        )

        print(
            "-" * 40
        )

        for section in sorted(
            sections
        ):

            print(
                section
            )

        print()
        print(
            "SAMPLE CHUNKS"
        )

        sample_indexes = sorted(
            {
                0,
                len(chunks) // 4,
                len(chunks) // 2,
                (
                    3
                    * len(chunks)
                    // 4
                ),
                len(chunks) - 1,
            }
        )

        for index in sample_indexes:

            print_chunk(
                chunks[index]
            )

        if too_small:

            print()
            print("=" * 80)
            print(
                "SMALLEST CHUNKS"
            )
            print("=" * 80)

            smallest = sorted(
                too_small,
                key=lambda chunk: (
                    chunk.token_count
                    or 0
                ),
            )

            for chunk in smallest[:5]:

                print_chunk(
                    chunk
                )

        if too_large:

            print()
            print("=" * 80)
            print(
                "OVERSIZED CHUNKS"
            )
            print("=" * 80)

            for chunk in (
                too_large[:5]
            ):

                print_chunk(
                    chunk
                )
        if result.blocks:

            print()
            print(
                "BLOCK TYPES"
            )

            print(
                "-" * 40
            )

            block_type_counts = Counter(
                block.content_type
                for block
                in result.blocks
            )

            for (
                content_type,
                count,
            ) in sorted(
                block_type_counts.items()
            ):

                print(
                    f"{content_type:<20} "
                    f"{count}"
                )

            tables = [
                block
                for block
                in result.blocks
                if (
                    block.content_type
                    == "table"
                )
            ]

            figures = [
                block
                for block
                in result.blocks
                if (
                    block.content_type
                    == "figure_caption"
                )
            ]

            metadata = [
                block
                for block
                in result.blocks
                if (
                    block.content_type
                    == "metadata"
                )
            ]

            print()
            print(
                f"Tables excluded from prose index: "
                f"{len(tables)}"
            )

            print(
                f"Figure captions excluded: "
                f"{len(figures)}"
            )

            print(
                f"Metadata blocks excluded: "
                f"{len(metadata)}"
            )


if __name__ == "__main__":
    main()