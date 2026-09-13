from pathlib import Path

from local_rag.config import (
    PDF_PATH,
)

from local_rag.ingestion.pdf_loader import (
    PDFLoader,
)

from local_rag.ingestion.section_splitter import (
    DocumentSectionSplitter,
)


def main() -> None:

    pdf_files = sorted(
        Path(PDF_PATH).glob("*.pdf")
    )

    if not pdf_files:
        print(
            "No PDF files found."
        )
        return

    pdf_file = pdf_files[0]

    print(
        f"Testing PDF: {pdf_file.name}"
    )

    loader = PDFLoader()

    splitter = (
        DocumentSectionSplitter()
    )

    all_pages = loader.load(
        pdf_file
    )

    main_pages = (
        splitter.split_main_content(
            all_pages
        )
    )

    print(
        f"All pages: "
        f"{len(all_pages)}"
    )

    print(
        f"Main-content pages: "
        f"{len(main_pages)}"
    )

    print(
        "\nIndexed page numbers:"
    )

    print(
        [
            page.page_number
            for page in main_pages
        ]
    )

    if main_pages:

        print(
            "\nLast indexed page:"
        )

        print(
            main_pages[-1].page_number
        )

        print(
            "\nLast 500 characters:"
        )

        print(
            main_pages[-1].content[
                -500:
            ]
        )


if __name__ == "__main__":
    main()