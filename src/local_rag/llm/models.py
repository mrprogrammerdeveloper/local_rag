from dataclasses import dataclass
from typing import Any


@dataclass
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    cached_tokens: int | None = None
    tool_use_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: LLMUsage | None = None
    raw_response: Any | None = None