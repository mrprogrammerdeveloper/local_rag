import re

from dataclasses import replace

from local_rag.ingestion.models import (
    DocumentBlock,
)


class SectionTreeBuilder:

    NUMBER_PATTERN = re.compile(
        r"^\s*"
        r"(?P<number>"
        r"\d+(?:\.\d+){0,4}"
        r")"
        r"\s*"
        r"(?:\|\s*)?"
    )

    def apply(
        self,
        blocks: list[DocumentBlock],
    ) -> list[DocumentBlock]:

        current_path: list[
            str
        ] = []

        current_title: (
            str | None
        ) = None

        result: list[
            DocumentBlock
        ] = []

        for block in blocks:

            if (
                block.content_type
                == "heading"
            ):

                heading = (
                    self._normalize_heading(
                        block.content
                    )
                )

                current_path = (
                    self._update_path(
                        heading=heading,
                        current_path=(
                            current_path
                        ),
                    )
                )

                current_title = (
                    heading
                )

                result.append(
                    replace(
                        block,
                        section_title=(
                            current_title
                        ),
                        section_path=list(
                            current_path
                        ),
                    )
                )

                continue

            result.append(
                replace(
                    block,
                    section_title=(
                        current_title
                    ),
                    section_path=list(
                        current_path
                    ),
                )
            )

        return result

    def _update_path(
        self,
        heading: str,
        current_path: list[str],
    ) -> list[str]:

        match = (
            self.NUMBER_PATTERN.match(
                heading
            )
        )

        if match is None:

            return [
                heading
            ]

        number = (
            match.group(
                "number"
            )
        )

        level = (
            number.count(".")
            + 1
        )

        path = list(
            current_path[
                :level - 1
            ]
        )

        path.append(
            heading
        )

        return path

    @staticmethod
    def _normalize_heading(
        heading: str,
    ) -> str:

        heading = (
            heading.strip()
        )

        heading = re.sub(
            r"\s+",
            " ",
            heading,
        )

        heading = heading.rstrip(
            "."
        )

        return heading