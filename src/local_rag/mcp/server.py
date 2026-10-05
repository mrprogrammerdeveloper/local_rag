from __future__ import annotations

from typing import Any

from mcp.server import (
    MCPServer,
)

from local_rag.excel.service import (
    ExcelService,
)

from local_rag.knowledge.service import (
    KnowledgeService,
)


mcp = MCPServer(
    "Local Academic RAG"
)

_knowledge: (
    KnowledgeService | None
) = None

_excel: (
    ExcelService | None
) = None


def knowledge(
) -> KnowledgeService:

    global _knowledge

    if _knowledge is None:

        _knowledge = (
            KnowledgeService()
        )

    return _knowledge


def excel(
) -> ExcelService:

    global _excel

    if _excel is None:

        _excel = (
            ExcelService()
        )

    return _excel


@mcp.tool()
def list_sources(
) -> dict[str, Any]:
    """
    List academic PDF documents and Excel workbooks
    currently available to the local knowledge system.
    """

    return (
        knowledge()
        .list_sources()
    )


@mcp.tool()
def search_knowledge(
    query: str,
    top_k: int = 5,
    source: str | None = None,
) -> dict[str, Any]:
    """
    Search indexed academic PDF content.

    Returns relevant source passages with page,
    chunk and reranker metadata. Use these passages
    as evidence when answering the user.

    When using a returned passage in an answer,
    cite at least the source filename and page.
    """

    return (
        knowledge()
        .search(
            query=query,
            top_k=top_k,
            source=source,
        )
    )


@mcp.tool()
def ingest_documents(
    paths: list[str],
    pipeline: str = "v2-basic",
) -> dict[str, Any]:
    """
    Add PDF or Excel files to the local knowledge system.

    PDFs are chunked, embedded and indexed in Qdrant.
    Excel files are stored for structured worksheet access.

    pipeline may be v1, v2, v2-basic, or another
    dynamically registered ingestion pipeline.
    """

    return (
        knowledge()
        .ingest_documents(
            paths=paths,
            pipeline=pipeline,
        )
    )


@mcp.tool()
def excel_list_sheets(
    path: str,
) -> dict[str, Any]:
    """
    List worksheets in an Excel workbook and report
    their row and column dimensions.
    """

    return (
        excel()
        .list_sheets(
            path
        )
    )


@mcp.tool()
def excel_read(
    path: str,
    sheet: str,
    start_row: int = 2,
    end_row: int = 50,
    header_row: int = 1,
) -> dict[str, Any]:
    """
    Read structured rows from an Excel worksheet.

    The header row is converted into JSON field names.
    At most 500 rows may be read in a single call.
    """

    return (
        excel()
        .read(
            path=path,
            sheet=sheet,
            start_row=start_row,
            end_row=end_row,
            header_row=header_row,
        )
    )


@mcp.tool()
def excel_search(
    path: str,
    query: str,
    sheet: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """
    Search cell values in an Excel workbook.

    This is lexical substring search and is useful for
    locating rows before calling excel_read.
    """

    return (
        excel()
        .search(
            path=path,
            query=query,
            sheet=sheet,
            max_results=max_results,
        )
    )


if __name__ == "__main__":

    mcp.run()
