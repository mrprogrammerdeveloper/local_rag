from local_rag.generation.prompt_builder import PromptBuilder
from local_rag.llm.ollama_client import OllamaLLM


llm = OllamaLLM()

answer = llm.generate(
    system_prompt=PromptBuilder.SYSTEM_PROMPT,
    user_prompt="""
CONTEXT:

Metamaterials are engineered materials whose properties
are primarily determined by their structure rather than
only their chemical composition.

QUESTION:

What are metamaterials?
""",
)

print(answer)