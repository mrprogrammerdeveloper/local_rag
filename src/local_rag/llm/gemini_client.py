import os

from google import genai

from local_rag.llm.base import (
    BaseLLMClient,
)

from local_rag.llm.models import (
    LLMResponse,
    LLMUsage,
)


class GeminiClient(BaseLLMClient):

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ) -> None:

        self.model = model

        self.api_key = (
            api_key
            or os.getenv(
                "GEMINI_API_KEY"
            )
            or os.getenv(
                "GOOGLE_API_KEY"
            )
        )

        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. "
                "Set GEMINI_API_KEY in .env."
            )

        self.client = genai.Client(
            api_key=self.api_key,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:

        interaction = (
            self.client
            .interactions
            .create(
                model=self.model,
                system_instruction=(
                    system_prompt
                ),
                input=(
                    user_prompt
                ),
            )
        )

        text = (
            interaction.output_text
            or ""
        ).strip()

        usage = self._extract_usage(
            interaction
        )

        return LLMResponse(
            text=text,
            provider="gemini",
            model=self.model,
            usage=usage,
            raw_response=interaction,
        )

    @staticmethod
    def _extract_usage(
        interaction,
    ) -> LLMUsage | None:

        usage = getattr(
            interaction,
            "usage",
            None,
        )

        if usage is None:
            return None

        return LLMUsage(
            input_tokens=getattr(
                usage,
                "total_input_tokens",
                None,
            ),
            output_tokens=getattr(
                usage,
                "total_output_tokens",
                None,
            ),
            thought_tokens=getattr(
                usage,
                "total_thought_tokens",
                None,
            ),
            cached_tokens=getattr(
                usage,
                "total_cached_tokens",
                None,
            ),
            tool_use_tokens=getattr(
                usage,
                "total_tool_use_tokens",
                None,
            ),
            total_tokens=getattr(
                usage,
                "total_tokens",
                None,
            ),
        )

    def close(
        self,
    ) -> None:

        self.client.close()