from local_rag.config import (
    PDF_DIRECTORY,
    REFERENCES_PATH,
)

from local_rag.ingestion.pdf_loader import PDFLoader
from local_rag.ingestion.reference_parser import (
    ReferenceParser,
)
from local_rag.references.store import (
    ReferenceStore,
)
from local_rag.retrieval.citation_extractor import (
    CitationExtractor,
)


def main() -> None:

    pdf_files = sorted(
        PDF_DIRECTORY.glob("*.pdf")
    )

    if not pdf_files:
        raise RuntimeError(
            f"No PDFs found in {PDF_DIRECTORY}"
        )

    pdf_file = pdf_files[0]

    print(
        f"Testing PDF: {pdf_file.name}"
    )

    loader = PDFLoader()

    pages = loader.load(
        pdf_file
    )

    print(
        f"Pages loaded: {len(pages)}"
    )

    parser = ReferenceParser()

    references = parser.parse(
        pages
    )

    print(
        f"References parsed: {len(references)}"
    )

    if not references:
        raise RuntimeError(
            "No references were extracted."
        )

    print(
        "\nFirst reference:"
    )

    print(
        f"[{references[0].number}] "
        f"{references[0].content}"
    )

    print(
        "\nLast reference:"
    )

    print(
        f"[{references[-1].number}] "
        f"{references[-1].content}"
    )

    store = ReferenceStore(
        REFERENCES_PATH
    )

    store.save(
        source=pdf_file.name,
        references=references,
    )

    print(
        "\nReference store saved."
    )

    reference_24 = store.get(
        source=pdf_file.name,
        number=24,
    )

    print(
        "\nReference 24:"
    )

    print(
        reference_24
    )

    if reference_24 is None:
        raise RuntimeError(
            "Reference 24 was not found."
        )

    extractor = CitationExtractor()

    fake_answer = (
        "Electromagnetic cloaking has been "
        f"demonstrated using metamaterials "
        f"[{pdf_file.name}, ref 24]."
    )

    used_references = extractor.extract(
        fake_answer
    )

    print(
        "\nExtracted citations:"
    )

    for reference in used_references:
        print(
            reference
        )

        content = store.get(
            source=reference.source,
            number=reference.number,
        )

        print(
            "Resolved reference:"
        )

        print(
            content
        )

    print(
        "\nAll reference tests passed."
    )


if __name__ == "__main__":
    main()