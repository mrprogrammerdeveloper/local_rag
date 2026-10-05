import re

from local_rag.retrieval.evidence import (
    Evidence,
)


class CitationRenderer:

    GROUP_PATTERN = re.compile(
        r"\[\s*"
        r"(E\d+"
        r"(?:\s*[,;]\s*E\d+)*)"
        r"\s*\]"
    )

    EVIDENCE_ID_PATTERN = re.compile(
        r"E\d+"
    )

    FINAL_CITATION_PATTERN = re.compile(
        r"\["
        r"[^\[\]]+?\.pdf,\s*"
        r"(?:ref\s+\d+|page\s+\d+)"
        r"\]",
        re.IGNORECASE,
    )

    FINAL_CITATION_BLOCK_PATTERN = re.compile(
        r"(?:"
        r"\["
        r"[^\[\]]+?\.pdf,\s*"
        r"(?:ref\s+\d+|page\s+\d+)"
        r"\]"
        r")"
        r"(?:"
        r"\s+"
        r"\["
        r"[^\[\]]+?\.pdf,\s*"
        r"(?:ref\s+\d+|page\s+\d+)"
        r"\]"
        r")+",
        re.IGNORECASE,
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

        rendered = (
            self.GROUP_PATTERN.sub(
                lambda match: (
                    self._replace_evidence_group(
                        match=match,
                        evidence_map=evidence_map,
                    )
                ),
                answer,
            )
        )

        rendered = (
            self.FINAL_CITATION_BLOCK_PATTERN.sub(
                self._deduplicate_citation_block,
                rendered,
            )
        )

        return rendered

    def _replace_evidence_group(
        self,
        match: re.Match,
        evidence_map: dict[str, Evidence],
    ) -> str:

        raw_group = (
            match.group(1)
        )

        evidence_ids = (
            self.EVIDENCE_ID_PATTERN.findall(
                raw_group
            )
        )

        for evidence_id in evidence_ids:

            if (
                evidence_id
                not in evidence_map
            ):
                return match.group(0)

        citations: list[str] = []

        seen: set[str] = set()

        for evidence_id in evidence_ids:

            item = evidence_map[
                evidence_id
            ]

            item_citations = (
                self._render_evidence(
                    item
                )
            )

            for citation in item_citations:

                if citation in seen:
                    continue

                seen.add(
                    citation
                )

                citations.append(
                    citation
                )

        if not citations:
            return match.group(0)

        return " ".join(
            citations
        )

    @staticmethod
    def _render_evidence(
        item: Evidence,
    ) -> list[str]:

        if item.reference_numbers:

            return [
                (
                    f"[{item.source}, "
                    f"ref {number}]"
                )
                for number
                in item.reference_numbers
            ]

        return [
            (
                f"[{item.source}, "
                f"page {item.page_number}]"
            )
        ]

    def _deduplicate_citation_block(
        self,
        match: re.Match,
    ) -> str:

        citations = (
            self.FINAL_CITATION_PATTERN.findall(
                match.group(0)
            )
        )

        unique: list[str] = []

        seen: set[str] = set()

        for citation in citations:

            normalized = (
                citation.lower()
            )

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            unique.append(
                citation
            )

        return " ".join(
            unique
        )