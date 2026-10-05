from local_rag.config import (
    REFERENCES_PATH,
)

from local_rag.references.alignment_validator import (
    ReferenceAlignmentValidator,
)

from local_rag.references.store import (
    ReferenceStore,
)

from local_rag.retrieval.reranker import (
    Reranker,
)


SOURCE = (
    "GJETA-2025-0260 (1).pdf"
)


def print_result(
    title: str,
    result,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(title)

    print(
        "=" * 80
    )

    if result is None:

        print(
            "Reference not found."
        )

        return

    print(
        f"Reference: "
        f"{result.reference_number}"
    )

    print(
        f"Score: "
        f"{result.score:.6f}"
    )

    print()

    print(
        "Evidence:"
    )

    print(
        result.evidence_text
    )

    print()

    print(
        "Reference text:"
    )

    print(
        result.reference_text
    )


def main() -> None:

    reference_store = (
        ReferenceStore(
            REFERENCES_PATH
        )
    )

    reranker = Reranker()

    validator = (
        ReferenceAlignmentValidator(
            reference_store=(
                reference_store
            ),
            reranker=reranker,
        )
    )

    # ------------------------------------------------
    # Expected strong alignment
    # ------------------------------------------------

    electromagnetic = (
        "Nezaratizadeh et al. used deep learning "
        "to optimize split-ring resonator "
        "configurations for electromagnetic "
        "metamaterial inverse design and "
        "broadband microwave absorption."
    )

    result_55 = (
        validator.score(
            evidence_text=(
                electromagnetic
            ),
            source=SOURCE,
            reference_number=55,
        )
    )

    print_result(
        "EXPECTED GOOD — "
        "electromagnetic claim vs ref 55",
        result_55,
    )

    # ------------------------------------------------
    # Expected strong alignment
    # ------------------------------------------------

    acoustic = (
        "Bacigalupo et al. employed machine "
        "learning to optimize acoustic and "
        "phononic metamaterials using "
        "bandgap-related design properties."
    )

    result_56 = (
        validator.score(
            evidence_text=acoustic,
            source=SOURCE,
            reference_number=56,
        )
    )

    print_result(
        "EXPECTED GOOD — "
        "acoustic claim vs ref 56",
        result_56,
    )

    # ------------------------------------------------
    # Known suspicious citation in source paper
    # ------------------------------------------------

    thermal = (
        "Li et al. used inverse design with "
        "neural networks to create thermal "
        "cloaks that redirect heat flow and "
        "protect sensitive electronics."
    )

    result_57 = (
        validator.score(
            evidence_text=thermal,
            source=SOURCE,
            reference_number=57,
        )
    )

    print_result(
        "KNOWN SUSPICIOUS — "
        "thermal cloak claim vs ref 57",
        result_57,
    )

    # ------------------------------------------------
    # Alternative thematic candidate
    #
    # This does NOT mean ref 54 is necessarily
    # the correct citation. It is only useful
    # for score comparison.
    # ------------------------------------------------

    result_54 = (
        validator.score(
            evidence_text=thermal,
            source=SOURCE,
            reference_number=54,
        )
    )

    print_result(
        "ALTERNATIVE COMPARISON — "
        "thermal cloak claim vs ref 54",
        result_54,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 80
    )

    results = [
        (
            "EM → ref 55",
            result_55,
        ),
        (
            "Acoustic → ref 56",
            result_56,
        ),
        (
            "Thermal → ref 57",
            result_57,
        ),
        (
            "Thermal → ref 54",
            result_54,
        ),
    ]

    for label, result in results:

        if result is None:
            score = "N/A"
        else:
            score = (
                f"{result.score:.6f}"
            )

        print(
            f"{label:<25} "
            f"{score}"
        )


if __name__ == "__main__":
    main()