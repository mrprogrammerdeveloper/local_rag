from local_rag.config import (
    REFERENCES_PATH,
    RETRIEVAL_CANDIDATES,
    TOP_K,
)

from local_rag.embeddings.embedder import (
    Embedder,
)

from local_rag.generation.prompt_builder import (
    PromptBuilder,
)

from local_rag.llm.selector import (
    select_llm,
)

from local_rag.llm.usage import (
    print_usage,
)

from local_rag.references.citation_normalizer import (
    CitationNormalizer,
)

from local_rag.references.citation_renderer import (
    CitationRenderer,
)

from local_rag.references.store import (
    ReferenceStore,
)

from local_rag.retrieval.citation_extractor import (
    CitationExtractor,
)

from local_rag.retrieval.evidence_builder import (
    EvidenceBuilder,
)

from local_rag.retrieval.reranker import (
    Reranker,
)

from local_rag.retrieval.retriever import (
    Retriever,
)

from local_rag.vector_store.qdrant import (
    QdrantVectorStore,
)


def print_references(
    answer: str,
    citation_extractor: CitationExtractor,
    reference_store: ReferenceStore,
) -> None:

    used_references = (
        citation_extractor.extract(
            answer
        )
    )

    if not used_references:
        return

    print(
        "\nReferences:\n"
    )

    for reference in used_references:

        content = reference_store.get(
            source=reference.source,
            number=reference.number,
        )

        if content is None:
            continue

        print(
            f"[{reference.source}, "
            f"ref {reference.number}]"
        )

        print(
            content
        )

        print()


def print_retrieved_sources(
    chunks,
) -> None:

    if not chunks:
        return

    print(
        "\nRetrieved Context:\n"
    )

    for chunk in chunks:

        vector_score = (
            f"{chunk.score:.4f}"
        )

        if (
            chunk.rerank_score
            is not None
        ):
            rerank_score = (
                f"{chunk.rerank_score:.4f}"
            )
        else:
            rerank_score = "-"

        print(
            f"- {chunk.source} | "
            f"page {chunk.page_number} | "
            f"vector {vector_score} | "
            f"rerank {rerank_score}"
        )


def close_llm(
    llm,
) -> None:

    close_method = getattr(
        llm,
        "close",
        None,
    )

    if callable(
        close_method
    ):
        close_method()


def main() -> None:

    print(
        "Loading embedding model..."
    )

    embedder = Embedder()

    vector_store = (
        QdrantVectorStore(
            vector_size=(
                embedder.dimension
            ),
        )
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    reranker = Reranker()

    reference_store = (
        ReferenceStore(
            REFERENCES_PATH
        )
    )

    citation_normalizer = (
        CitationNormalizer(
            reference_store
        )
    )

    evidence_builder = (
        EvidenceBuilder(
            citation_normalizer
        )
    )

    citation_renderer = (
        CitationRenderer()
    )

    citation_extractor = (
        CitationExtractor()
    )

    prompt_builder = (
        PromptBuilder()
    )

    # --------------------------------
    # Select LLM provider / model
    # --------------------------------

    llm = select_llm()

    print(
        "\nRAG is ready."
    )

    print(
        "Type 'exit' to quit.\n"
    )

    try:

        while True:

            question = input(
                "Question: "
            ).strip()

            if not question:
                continue

            if question.lower() in {
                "exit",
                "quit",
            }:
                break

            print(
                "\nSearching documents..."
            )

            # --------------------------------
            # Dense Retrieval
            # --------------------------------

            candidates = (
                retriever.retrieve(
                    query=question,
                    top_k=(
                        RETRIEVAL_CANDIDATES
                    ),
                )
            )

            if not candidates:

                print(
                    "\nNo relevant context "
                    "found.\n"
                )

                continue

            print(
                f"Retrieved "
                f"{len(candidates)} "
                f"candidates."
            )

            # --------------------------------
            # Reranking
            # --------------------------------

            print(
                "Reranking candidates..."
            )

            chunks = (
                reranker.rerank(
                    query=question,
                    chunks=candidates,
                    top_k=TOP_K,
                )
            )

            if not chunks:

                print(
                    "\nNo relevant chunks "
                    "after reranking.\n"
                )

                continue

            # --------------------------------
            # Evidence
            # --------------------------------

            evidence = (
                evidence_builder.build(
                    chunks
                )
            )

            if not evidence:

                print(
                    "\nNo usable evidence "
                    "found.\n"
                )

                continue

            # --------------------------------
            # Prompt
            # --------------------------------

            user_prompt = (
                prompt_builder
                .build_user_prompt(
                    question=question,
                    evidence=evidence,
                )
            )

            print(
                "Generating answer..."
            )

            # --------------------------------
            # LLM Generation
            # --------------------------------

            try:

                llm_response = (
                    llm.generate(
                        system_prompt=(
                            PromptBuilder
                            .SYSTEM_PROMPT
                        ),
                        user_prompt=(
                            user_prompt
                        ),
                    )
                )

            except Exception as exc:

                print()
                print(
                    "LLM generation failed:"
                )

                print(
                    str(exc)
                )

                print(
                    "\n"
                    + "=" * 70
                    + "\n"
                )

                continue

            raw_answer = (
                llm_response.text
            )

            if not raw_answer:

                print(
                    "\nLLM returned an "
                    "empty answer.\n"
                )

                continue

            # --------------------------------
            # Citation Rendering
            # --------------------------------

            answer = (
                citation_renderer.render(
                    answer=raw_answer,
                    evidence=evidence,
                )
            )

            # --------------------------------
            # Final Answer
            # --------------------------------

            print(
                "\nAnswer:\n"
            )

            print(
                answer
            )

            # --------------------------------
            # Academic References
            # --------------------------------

            print_references(
                answer=answer,
                citation_extractor=(
                    citation_extractor
                ),
                reference_store=(
                    reference_store
                ),
            )

            # --------------------------------
            # Retrieved Context Debug Info
            # --------------------------------

            print_retrieved_sources(
                chunks
            )

            # --------------------------------
            # LLM Token Usage
            # --------------------------------

            print_usage(
                llm_response
            )

            print(
                "\n"
                + "=" * 70
                + "\n"
            )

    finally:

        close_llm(
            llm
        )

        print(
            "\nLLM client closed."
        )


if __name__ == "__main__":
    main()