from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
)

from pathlib import Path
from typing import Any

from openpyxl import (
    load_workbook,
)


class ExcelService:
    """
    Structured Excel access for MCP.

    Excel is deliberately NOT converted into RAG chunks.
    Claude receives workbook rows as structured data instead.
    """

    SUPPORTED_SUFFIXES = {
        ".xlsx",
        ".xlsm",
    }

    def list_sheets(
        self,
        path: str,
    ) -> dict[str, Any]:

        workbook_path = (
            self._resolve_path(
                path
            )
        )

        workbook = load_workbook(
            workbook_path,
            read_only=True,
            data_only=True,
        )

        try:

            sheets = []

            for worksheet in (
                workbook.worksheets
            ):

                sheets.append(
                    {
                        "name": (
                            worksheet.title
                        ),
                        "rows": (
                            worksheet.max_row
                        ),
                        "columns": (
                            worksheet.max_column
                        ),
                    }
                )

            return {
                "file": str(
                    workbook_path
                ),
                "sheets": sheets,
            }

        finally:

            workbook.close()

    def read(
        self,
        path: str,
        sheet: str,
        start_row: int = 2,
        end_row: int = 50,
        header_row: int = 1,
    ) -> dict[str, Any]:

        workbook_path = (
            self._resolve_path(
                path
            )
        )

        if header_row < 1:
            raise ValueError(
                "header_row must be >= 1"
            )

        if start_row < 1:
            raise ValueError(
                "start_row must be >= 1"
            )

        if end_row < start_row:
            raise ValueError(
                "end_row must be >= start_row"
            )

        if (
            end_row
            - start_row
            > 500
        ):

            raise ValueError(
                "A single Excel read is limited "
                "to 500 rows."
            )

        workbook = load_workbook(
            workbook_path,
            read_only=True,
            data_only=True,
        )

        try:

            if sheet not in (
                workbook.sheetnames
            ):

                raise ValueError(
                    f"Unknown sheet: {sheet}"
                )

            worksheet = workbook[
                sheet
            ]

            header_values = next(
                worksheet.iter_rows(
                    min_row=header_row,
                    max_row=header_row,
                    values_only=True,
                )
            )

            headers = (
                self._normalize_headers(
                    header_values
                )
            )

            actual_start = max(
                start_row,
                header_row + 1,
            )

            actual_end = min(
                end_row,
                worksheet.max_row,
            )

            rows = []

            if actual_start <= actual_end:

                for (
                    excel_row_number,
                    values,
                ) in enumerate(
                    worksheet.iter_rows(
                        min_row=actual_start,
                        max_row=actual_end,
                        values_only=True,
                    ),
                    start=actual_start,
                ):

                    row = {
                        headers[index]: (
                            self._json_value(
                                value
                            )
                        )
                        for index, value
                        in enumerate(
                            values
                        )
                    }

                    row[
                        "_row"
                    ] = (
                        excel_row_number
                    )

                    rows.append(
                        row
                    )

            return {
                "file": str(
                    workbook_path
                ),
                "sheet": sheet,
                "header_row": header_row,
                "start_row": (
                    actual_start
                ),
                "end_row": (
                    actual_end
                ),
                "headers": headers,
                "rows": rows,
                "count": len(
                    rows
                ),
            }

        finally:

            workbook.close()

    def search(
        self,
        path: str,
        query: str,
        sheet: str | None = None,
        max_results: int = 50,
    ) -> dict[str, Any]:

        workbook_path = (
            self._resolve_path(
                path
            )
        )

        query = query.strip()

        if not query:

            raise ValueError(
                "query cannot be empty"
            )

        max_results = max(
            1,
            min(
                int(max_results),
                200,
            ),
        )

        workbook = load_workbook(
            workbook_path,
            read_only=True,
            data_only=True,
        )

        try:

            if sheet is None:

                worksheets = (
                    workbook.worksheets
                )

            else:

                if sheet not in (
                    workbook.sheetnames
                ):

                    raise ValueError(
                        f"Unknown sheet: "
                        f"{sheet}"
                    )

                worksheets = [
                    workbook[
                        sheet
                    ]
                ]

            query_lower = (
                query.lower()
            )

            matches = []

            for worksheet in worksheets:

                for row_number, values in (
                    enumerate(
                        worksheet.iter_rows(
                            values_only=True
                        ),
                        start=1,
                    )
                ):

                    for column_index, value in (
                        enumerate(
                            values,
                            start=1,
                        )
                    ):

                        if value is None:
                            continue

                        string_value = str(
                            value
                        )

                        if (
                            query_lower
                            not in
                            string_value.lower()
                        ):
                            continue

                        matches.append(
                            {
                                "sheet": (
                                    worksheet.title
                                ),
                                "row": (
                                    row_number
                                ),
                                "column": (
                                    column_index
                                ),
                                "value": (
                                    self._json_value(
                                        value
                                    )
                                ),
                            }
                        )

                        if (
                            len(matches)
                            >= max_results
                        ):

                            return {
                                "file": str(
                                    workbook_path
                                ),
                                "query": query,
                                "count": (
                                    len(matches)
                                ),
                                "matches": (
                                    matches
                                ),
                            }

            return {
                "file": str(
                    workbook_path
                ),
                "query": query,
                "count": len(
                    matches
                ),
                "matches": matches,
            }

        finally:

            workbook.close()

    def _resolve_path(
        self,
        path: str,
    ) -> Path:

        raw_path = Path(
            path
        ).expanduser()

        candidates = [
            raw_path,
            Path.cwd()
            / raw_path,
            Path.cwd()
            / "data"
            / "excel"
            / raw_path.name,
        ]

        for candidate in candidates:

            candidate = (
                candidate.resolve()
            )

            if not candidate.exists():
                continue

            if not candidate.is_file():
                continue

            if (
                candidate.suffix.lower()
                not in
                self.SUPPORTED_SUFFIXES
            ):

                raise ValueError(
                    "Supported Excel formats: "
                    ".xlsx and .xlsm"
                )

            return candidate

        raise FileNotFoundError(
            f"Excel workbook not found: "
            f"{path}"
        )

    @staticmethod
    def _normalize_headers(
        values: tuple[Any, ...],
    ) -> list[str]:

        headers: list[str] = []

        used: set[str] = set()

        for index, value in enumerate(
            values,
            start=1,
        ):

            if value is None:

                base = (
                    f"column_{index}"
                )

            else:

                base = (
                    str(value)
                    .strip()
                )

                if not base:

                    base = (
                        f"column_{index}"
                    )

            name = base

            suffix = 2

            while name in used:

                name = (
                    f"{base}_{suffix}"
                )

                suffix += 1

            used.add(
                name
            )

            headers.append(
                name
            )

        return headers

    @staticmethod
    def _json_value(
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            (
                datetime,
                date,
                time,
            ),
        ):

            return (
                value.isoformat()
            )

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ) or value is None:

            return value

        return str(
            value
        )
