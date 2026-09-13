import re

from local_rag.retrieval.evidence import Evidence


class CitationRenderer:
    PATTERN = re.compile(
        r"\[(E\d+)\]"
    )

    def render(
        self,
        answer: str,
        evidence: list[Evidence],
    ) -> str:

        evidence_map = {
            item.id: item
            for item in evidence
        }

        def replace(
            match: re.Match,
        ) -> str:

            evidence_id = match.group(1)

            item = evidence_map.get(
                evidence_id
            )

            if item is None:
                return match.group(0)

            if item.reference_numbers:
                return " ".join(
                    f"[{item.source}, ref {number}]"
                    for number
                    in item.reference_numbers
                )

            return (
                f"[{item.source}, "
                f"page {item.page_number}]"
            )

        return self.PATTERN.sub(
            replace,
            answer,
        )