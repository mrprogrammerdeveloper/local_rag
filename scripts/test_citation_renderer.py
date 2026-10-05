from local_rag.references.citation_renderer import (
    CitationRenderer,
)

from local_rag.retrieval.evidence import (
    Evidence,
)


def main() -> None:

    renderer = CitationRenderer()

    evidence = [

        Evidence(
            id="E1",
            content="Evidence one.",
            source="paper_a.pdf",
            page_number=5,
            reference_numbers=[
                24
            ],
        ),

        Evidence(
            id="E2",
            content="Evidence two.",
            source="paper_a.pdf",
            page_number=8,
            reference_numbers=[],
        ),

        Evidence(
            id="E3",
            content="Evidence three.",
            source="paper_b.pdf",
            page_number=7,
            reference_numbers=[
                10,
                11,
            ],
        ),
    ]

    # -----------------------------
    # Single citation
    # -----------------------------

    result = renderer.render(
        answer=(
            "Claim [E1]."
        ),
        evidence=evidence,
    )

    assert result == (
        "Claim "
        "[paper_a.pdf, ref 24]."
    )

    # -----------------------------
    # Grouped citations
    # -----------------------------

    result = renderer.render(
        answer=(
            "Claim [E1, E2]."
        ),
        evidence=evidence,
    )

    assert result == (
        "Claim "
        "[paper_a.pdf, ref 24] "
        "[paper_a.pdf, page 8]."
    )

    # -----------------------------
    # Adjacent citations
    # -----------------------------

    result = renderer.render(
        answer=(
            "Claim [E1][E2]."
        ),
        evidence=evidence,
    )

    assert result == (
        "Claim "
        "[paper_a.pdf, ref 24]"
        "[paper_a.pdf, page 8]."
    )

    # -----------------------------
    # Multiple original references
    # -----------------------------

    result = renderer.render(
        answer=(
            "Claim [E3]."
        ),
        evidence=evidence,
    )

    assert result == (
        "Claim "
        "[paper_b.pdf, ref 10] "
        "[paper_b.pdf, ref 11]."
    )

    # -----------------------------
    # Mixed citation styles
    # -----------------------------

    result = renderer.render(
        answer=(
            "Claim "
            "[E1, E2][E3]."
        ),
        evidence=evidence,
    )

    assert (
        "[E1" not in result
    )

    assert (
        "[E2" not in result
    )

    assert (
        "[E3" not in result
    )

    print(
        "All citation renderer "
        "tests passed."
    )


if __name__ == "__main__":
    main()