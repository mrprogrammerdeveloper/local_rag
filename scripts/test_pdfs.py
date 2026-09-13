from pathlib import Path

from local_rag.ingestion.pdf_loader import PDFLoader


pdf_path = Path(
    "data/pdfs/test.pdf"
)


loader = PDFLoader()

pages = loader.load(pdf_path)


print(
    "Pages:",
    len(pages)
)


for page in pages[:2]:
    print("----------------")
    print(
        "Source:",
        page.source
    )
    print(
        "Page:",
        page.page_number
    )
    print(
        page.content[:300]
    )