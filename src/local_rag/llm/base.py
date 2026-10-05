from abc import ABC, abstractmethod

from local_rag.llm.models import (
    LLMResponse,
)


class BaseLLMClient(ABC):

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        raise NotImplementedError