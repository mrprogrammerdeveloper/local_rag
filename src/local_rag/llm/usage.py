from local_rag.llm.models import (
    LLMResponse,
)


def print_usage(
    response: LLMResponse,
) -> None:

    usage = response.usage

    print()
    print("=" * 60)
    print("LLM USAGE")
    print("=" * 60)

    print(
        f"Provider: "
        f"{response.provider}"
    )

    print(
        f"Model: "
        f"{response.model}"
    )

    if usage is None:
        print(
            "Usage information "
            "was not returned."
        )
        return

    if usage.input_tokens is not None:
        print(
            f"Input tokens: "
            f"{usage.input_tokens}"
        )

    if usage.output_tokens is not None:
        print(
            f"Output tokens: "
            f"{usage.output_tokens}"
        )

    if usage.thought_tokens is not None:
        print(
            f"Thought tokens: "
            f"{usage.thought_tokens}"
        )

    if usage.cached_tokens is not None:
        print(
            f"Cached tokens: "
            f"{usage.cached_tokens}"
        )

    if (
        usage.tool_use_tokens
        is not None
    ):
        print(
            f"Tool-use tokens: "
            f"{usage.tool_use_tokens}"
        )

    if usage.total_tokens is not None:
        print(
            f"Total tokens: "
            f"{usage.total_tokens}"
        )

    print("=" * 60)