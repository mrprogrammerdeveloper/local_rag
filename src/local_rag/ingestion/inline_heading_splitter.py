import re

from dataclasses import replace

from local_rag.ingestion.models import (
    DocumentBlock,
)


class InlineHeadingSplitter:

    INLINE_NUMBERED_HEADING = (
        re.compile(
            r"^"
            r"(?P<heading>"
            r"\d+(?:\.\d+){1,4}"
            r"\s*(?:\|\s*)?"
            r"[A-Z]"
            r"[^.!?]{1,120}"
            r"[.!?]"
            r")"
            r"\s+"
            r"(?P<rest>.+)"
            r"$"
        )
    )

    def split_blocks(
        self,
        blocks: list[DocumentBlock],
    ) -> list[DocumentBlock]:

        result: list[
            DocumentBlock
        ] = []

        new_index = 0

        for block in blocks:

            lines = [
                line.strip()
                for line
                in block.content.splitlines()
                if line.strip()
            ]

            if not lines:
                continue

            for line in lines:

                pieces = (
                    self._split_line(
                        line
                    )
                )

                for piece in pieces:

                    result.append(
                        replace(
                            block,
                            content=piece,
                            block_index=(
                                new_index
                            ),
                        )
                    )

                    new_index += 1

        return result

    def _split_line(
        self,
        line: str,
    ) -> list[str]:

        match = (
            self.INLINE_NUMBERED_HEADING
            .match(
                line
            )
        )

        if match is None:

            return [
                line
            ]

        heading = (
            match.group(
                "heading"
            )
            .strip()
        )

        rest = (
            match.group(
                "rest"
            )
            .strip()
        )

        result = [
            heading
        ]

        if rest:

            result.append(
                rest
            )

        return result