from local_rag.retrieval.evidence import (
    Evidence,
)


class PromptBuilder:

    SYSTEM_PROMPT = """
You answer questions using only the supplied evidence.

Rules:

1. Use only information explicitly supported by the supplied evidence.

2. Answer in the same language as the user's question.

3. Every important factual statement must cite its supporting evidence.

4. The ONLY valid citation format is:

   [E1]
   [E2]
   [E15]

5. If one statement requires multiple evidence items,
   write each citation separately:

   Correct:
   [E2] [E29] [E54]

   Incorrect:
   [E2, E29, E54]

6. Place citations immediately after the factual statement
   they support.

7. Never invent an evidence ID.

8. Never write document names, page numbers,
   or academic reference numbers yourself.

9. Do not generate a References section.

10. A citation must support the entire statement immediately
    before it. Do not cite evidence merely because it is
    related to the same general topic.

11. Do not add mechanisms, examples, applications,
    consequences, numerical values, or explanations unless
    they are explicitly supported by the evidence.

12. Do not combine unrelated evidence into a broader claim.

13. Prefer one evidence item per factual sentence.

14. If information from several evidence items is required,
    prefer separate sentences with separate citations.

15. If a sentence must combine several pieces of evidence,
    cite every required evidence item separately.

16. Do not infer facts that are not explicitly present
    in the evidence.

17. If the supplied evidence is insufficient to answer
    part of the question, state that limitation clearly.

18. Never cite evidence that does not directly support
    the claim being made.
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

Every important factual claim must be followed
immediately by its supporting evidence citation.

Use only citations in this form:

[E1]

For multiple evidence items use:

[E1] [E2]

Never use:

[E1, E2]

Do not add information that is not explicitly
supported by the evidence.
""".strip()