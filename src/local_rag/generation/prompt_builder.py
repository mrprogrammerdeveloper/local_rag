from local_rag.retrieval.evidence import Evidence


class PromptBuilder:
    SYSTEM_PROMPT = """
You answer questions using only the supplied evidence.

Rules:

1. Use only information supported by the evidence.
2. Answer in the same language as the user's question.
3. Every important factual statement must cite its supporting
   evidence using exactly [E1], [E2], [E3], etc.
4. Never invent an evidence ID.
5. Never write document names, page numbers, or reference numbers.
6. Do not generate a References section.
7. Do not combine unrelated evidence into unsupported claims.
8. If evidence is insufficient, state that clearly.
9. Prefer one evidence item per factual sentence.
10. If a sentence combines information from multiple evidence items,
    you MUST cite every evidence item required to support all parts
    of that sentence.
11. Do not merge facts from multiple evidence items into one sentence
    unless necessary. Prefer separate sentences with separate citations.
""".strip()

    def build_user_prompt(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> str:

        evidence_parts: list[str] = []

        for item in evidence:
            evidence_parts.append(
                f"""
{item.id}

{item.content}
""".strip()
            )

        context = "\n\n---\n\n".join(
            evidence_parts
        )

        return f"""
EVIDENCE:

{context}


QUESTION:

{question}


Answer only from the evidence above.
Cite each important factual claim using its evidence ID.
""".strip()