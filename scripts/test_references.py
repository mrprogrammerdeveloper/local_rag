from pathlib import Path

from local_rag.ingestion.pdf_loader import PDFLoader
from local_rag.ingestion.reference_parser import ReferenceParser


EXPECTED_REFERENCES = {
    "GJETA-2025-0260 (1).pdf": 77,
    "test.pdf": 158,
}


def main():
    loader = PDFLoader()
    parser = ReferenceParser()

    pdf_directory = Path(
        "data/pdfs"
    )

    for filename, expected_count in (
        EXPECTED_REFERENCES.items()
    ):
        pdf_path = (
            pdf_directory
            / filename
        )

        print()
        print("=" * 70)
        print(
            f"Testing PDF: {filename}"
        )
        print("=" * 70)

        pages = loader.load(
            pdf_path
        )

        references = parser.parse(
            pages
        )

        print(
            f"Pages loaded: "
            f"{len(pages)}"
        )

        print(
            f"References parsed: "
            f"{len(references)}"
        )

        assert (
            len(references)
            == expected_count
        ), (
            f"{filename}: "
            f"expected "
            f"{expected_count} references, "
            f"got {len(references)}"
        )

        assert references, (
            f"{filename}: "
            "no references parsed"
        )

        assert (
            references[0].number
            == 1
        ), (
            f"{filename}: "
            "first reference is not 1"
        )

        assert (
            references[-1].number
            == expected_count
        ), (
            f"{filename}: "
            f"last reference should be "
            f"{expected_count}, "
            f"got "
            f"{references[-1].number}"
        )

        print()
        print(
            "First reference:"
        )
        print(
            f"[{references[0].number}] "
            f"{references[0].content}"
        )

        print()
        print(
            "Last reference:"
        )
        print(
            f"[{references[-1].number}] "
            f"{references[-1].content}"
        )

        print()
        print(
            "✓ Reference parsing passed"
        )

    print()
    print(
        "All reference parser "
        "regression tests passed."
    )


if __name__ == "__main__":
    main()