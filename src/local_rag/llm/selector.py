import os

from local_rag.llm.base import (
    BaseLLMClient,
)

from local_rag.llm.gemini_client import (
    GeminiClient,
)

from local_rag.llm.ollama_client import (
    OllamaClient,
)


def _get_gemini_models() -> list[str]:

    raw_models = os.getenv(
        "GEMINI_MODELS",
        "gemini-3.8-flash",
    )

    return [
        model.strip()
        for model in raw_models.split(",")
        if model.strip()
    ]


def _select_from_list(
    title: str,
    options: list[str],
) -> str:

    if not options:
        raise ValueError(
            f"No options available for {title}"
        )

    print()
    print(title)
    print("-" * len(title))

    for index, option in enumerate(
        options,
        start=1,
    ):
        print(
            f"{index}. {option}"
        )

    while True:

        value = input(
            "\nSelect: "
        ).strip()

        try:
            selected_index = (
                int(value) - 1
            )

            if (
                0
                <= selected_index
                < len(options)
            ):
                return options[
                    selected_index
                ]

        except ValueError:
            pass

        print(
            "Invalid selection."
        )


def select_llm() -> BaseLLMClient:

    provider = _select_from_list(
        title="Select LLM provider",
        options=[
            "Local / Ollama",
            "Gemini API",
        ],
    )

    if provider == "Local / Ollama":

        model = os.getenv(
            "OLLAMA_MODEL",
            "qwen3.5:4b",
        )

        print()
        print(
            f"Using local model: "
            f"{model}"
        )

        return OllamaClient(
            model=model,
        )

    gemini_models = (
        _get_gemini_models()
    )

    selected_model = (
        _select_from_list(
            title="Select Gemini model",
            options=gemini_models,
        )
    )

    print()
    print(
        f"Using Gemini model: "
        f"{selected_model}"
    )

    return GeminiClient(
        model=selected_model,
    )