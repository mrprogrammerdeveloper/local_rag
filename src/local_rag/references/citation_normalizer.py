import re

from local_rag.references.store import ReferenceStore


class CitationNormalizer:
    CITATION_PATTERN = re.compile(
        r"\[([0-9,\-\s]+)\]"
    )

    def __init__(
        self,
        reference_store: ReferenceStore,
    ):
        self.reference_store = reference_store

    def normalize(
        self,
        text: str,
        source: str,
    ) -> str:

        references = self.reference_store.load(
            source
        )

        if not references:
            return text

        def replace(
            match: re.Match,
        ) -> str:

            numbers = self._parse_numbers(
                match.group(1)
            )

            valid_numbers = [
                number
                for number in numbers
                if number in references
            ]

            if not valid_numbers:
                return match.group(0)

            return " ".join(
                f"[{source}, ref {number}]"
                for number in valid_numbers
            )

        return self.CITATION_PATTERN.sub(
            replace,
            text,
        )

    @staticmethod
    def _parse_numbers(
        value: str,
    ) -> list[int]:

        numbers: list[int] = []

        for part in value.split(","):
            part = part.strip()

            if not part:
                continue

            if "-" in part:
                start_text, end_text = part.split(
                    "-",
                    maxsplit=1,
                )

                start_text = start_text.strip()
                end_text = end_text.strip()

                if (
                    start_text.isdigit()
                    and end_text.isdigit()
                ):
                    start = int(start_text)
                    end = int(end_text)

                    numbers.extend(
                        range(
                            start,
                            end + 1,
                        )
                    )

            elif part.isdigit():
                numbers.append(
                    int(part)
                )

        return numbers