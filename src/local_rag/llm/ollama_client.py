import requests

from local_rag.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
)

from local_rag.llm.base import (
    BaseLLMClient,
)

from local_rag.llm.models import (
    LLMResponse,
    LLMUsage,
)


class OllamaClient(BaseLLMClient):

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
    ) -> None:

        self.model = model

        self.host = (
            host.rstrip("/")
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 8192,
                    "num_predict": 768,
                },
            },
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        input_tokens = data.get(
            "prompt_eval_count"
        )

        output_tokens = data.get(
            "eval_count"
        )

        total_tokens = None

        if (
            input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = (
                input_tokens
                + output_tokens
            )

        usage = LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            text=data.get(
                "response",
                "",
            ).strip(),
            provider="ollama",
            model=self.model,
            usage=usage,
            raw_response=data,
        )

    def close(
        self,
    ) -> None:
        pass