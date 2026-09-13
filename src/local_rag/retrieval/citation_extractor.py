import re
from dataclasses import dataclass


@dataclass(frozen=True)
class UsedReference:
    source: str
    number: int


class CitationExtractor:
    PATTERN = re.compile(
        r"\[([^]]+?\.pdf),\s*ref\s+(\d+)\]",
        re.IGNORECASE,
    )

    def extract(
        self,
        text: str,
    ) -> list[UsedReference]:

        references: list[
            UsedReference
        ] = []

        seen: set[
            tuple[str, int]
        ] = set()

        for match in self.PATTERN.finditer(
            text
        ):
            source = (
                match.group(1).strip()
            )

            number = int(
                match.group(2)
            )

            key = (
                source,
                number,
            )

            if key in seen:
                continue

            seen.add(key)

            references.append(
                UsedReference(
                    source=source,
                    number=number,
                )
            )

        return references