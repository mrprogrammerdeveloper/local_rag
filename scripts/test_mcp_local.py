from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from openpyxl import Workbook

from mcp import (
    Client,
    StdioServerParameters,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PYTHON_BIN = sys.executable

SERVER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "mcp_server.py"
)

TEST_EXCEL = (
    PROJECT_ROOT
    / "data"
    / "excel"
    / "mcp_test.xlsx"
)


def pretty(
    value,
) -> None:

    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def create_test_excel(
) -> None:

    TEST_EXCEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    workbook = Workbook()

    sheet = workbook.active

    sheet.title = (
        "Metamaterial Results"
    )

    sheet.append(
        [
            "sample",
            "theta",
            "length",
            "poisson_ratio",
        ]
    )

    sheet.append(
        [
            "A",
            30,
            5.0,
            -0.42,
        ]
    )

    sheet.append(
        [
            "B",
            45,
            6.5,
            -0.31,
        ]
    )

    sheet.append(
        [
            "C",
            60,
            8.0,
            -0.18,
        ]
    )

    second_sheet = (
        workbook.create_sheet(
            "Material Data"
        )
    )

    second_sheet.append(
        [
            "material",
            "density",
            "elastic_modulus",
        ]
    )

    second_sheet.append(
        [
            "PLA",
            1.24,
            3500,
        ]
    )

    second_sheet.append(
        [
            "TPU",
            1.18,
            25,
        ]
    )

    workbook.save(
        TEST_EXCEL
    )

    workbook.close()


def get_result_data(
    result,
):

    structured = getattr(
        result,
        "structured_content",
        None,
    )

    if structured is not None:

        return structured

    content = getattr(
        result,
        "content",
        None,
    )

    if content is None:

        return str(
            result
        )

    return [
        getattr(
            item,
            "text",
            str(item),
        )
        for item in content
    ]


async def main() -> None:

    print()
    print(
        "=" * 70
    )
    print(
        "LOCAL MCP TEST"
    )
    print(
        "=" * 70
    )

    print()
    print(
        f"Project: {PROJECT_ROOT}"
    )

    print(
        f"Python: {PYTHON_BIN}"
    )

    print(
        f"Server: {SERVER_SCRIPT}"
    )

    create_test_excel()

    print(
        f"Test Excel: {TEST_EXCEL}"
    )

    server = (
        StdioServerParameters(
            command=PYTHON_BIN,
            args=[
                str(
                    SERVER_SCRIPT
                )
            ],
            cwd=str(
                PROJECT_ROOT
            ),
        )
    )

    print()
    print(
        "Starting MCP server "
        "through stdio..."
    )

    async with Client(
        server
    ) as client:

        print()
        print(
            "Connection: OK"
        )

        print(
            f"Protocol: "
            f"{client.protocol_version}"
        )

        if client.server_info:

            print(
                f"Server: "
                f"{client.server_info.name}"
            )

        print()
        print(
            "=" * 70
        )
        print(
            "1. LIST TOOLS"
        )
        print(
            "=" * 70
        )

        tools_result = (
            await client.list_tools()
        )

        tool_names = [
            tool.name
            for tool
            in tools_result.tools
        ]

        pretty(
            tool_names
        )

        expected_tools = {
            "list_sources",
            "search_knowledge",
            "ingest_documents",
            "excel_list_sheets",
            "excel_read",
            "excel_search",
        }

        missing = (
            expected_tools
            - set(
                tool_names
            )
        )

        if missing:

            raise RuntimeError(
                "Missing MCP tools: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        print()
        print(
            "=" * 70
        )
        print(
            "2. LIST SOURCES"
        )
        print(
            "=" * 70
        )

        result = (
            await client.call_tool(
                "list_sources",
                {},
            )
        )

        pretty(
            get_result_data(
                result
            )
        )

        print()
        print(
            "=" * 70
        )
        print(
            "3. EXCEL LIST SHEETS"
        )
        print(
            "=" * 70
        )

        result = (
            await client.call_tool(
                "excel_list_sheets",
                {
                    "path": str(
                        TEST_EXCEL
                    )
                },
            )
        )

        pretty(
            get_result_data(
                result
            )
        )

        print()
        print(
            "=" * 70
        )
        print(
            "4. EXCEL READ"
        )
        print(
            "=" * 70
        )

        result = (
            await client.call_tool(
                "excel_read",
                {
                    "path": str(
                        TEST_EXCEL
                    ),
                    "sheet": (
                        "Metamaterial Results"
                    ),
                    "start_row": 2,
                    "end_row": 4,
                    "header_row": 1,
                },
            )
        )

        pretty(
            get_result_data(
                result
            )
        )

        print()
        print(
            "=" * 70
        )
        print(
            "5. EXCEL SEARCH"
        )
        print(
            "=" * 70
        )

        result = (
            await client.call_tool(
                "excel_search",
                {
                    "path": str(
                        TEST_EXCEL
                    ),
                    "query": "PLA",
                },
            )
        )

        pretty(
            get_result_data(
                result
            )
        )

        print()
        print(
            "=" * 70
        )
        print(
            "6. SEARCH KNOWLEDGE"
        )
        print(
            "=" * 70
        )

        print(
            "This step may take a while "
            "because BGE-M3 and the "
            "reranker load lazily."
        )

        result = (
            await client.call_tool(
                "search_knowledge",
                {
                    "query": (
                        "What is inverse design "
                        "in metamaterials?"
                    ),
                    "top_k": 3,
                },
            )
        )

        search_data = (
            get_result_data(
                result
            )
        )

        pretty(
            search_data
        )

        print()
        print(
            "=" * 70
        )
        print(
            "ALL MCP TESTS COMPLETED"
        )
        print(
            "=" * 70
        )


if __name__ == "__main__":

    asyncio.run(
        main()
    )
