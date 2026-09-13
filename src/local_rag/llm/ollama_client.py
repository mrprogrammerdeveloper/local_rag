from ollama import Client

from local_rag.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
)


class OllamaLLM:
    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
    ):
        self.model = model
        self.client = Client(
            host=host
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            think=False,
            options={
                "temperature": 0.1,
                "num_ctx": 8192,
                "num_predict": 768,
            },
        )

        content = response.message.content

        print(
            "\n[LLM DEBUG]"
        )
        print(
            "Done reason:",
            getattr(
                response,
                "done_reason",
                None,
            ),
        )
        print(
            "Prompt tokens:",
            getattr(
                response,
                "prompt_eval_count",
                None,
            ),
        )
        print(
            "Generated tokens:",
            getattr(
                response,
                "eval_count",
                None,
            ),
        )

        if not content:
            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        return content.strip()