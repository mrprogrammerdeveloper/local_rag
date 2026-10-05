import os

from local_rag.ingestion.base import (
    BaseIngestionPipeline,
)

from local_rag.ingestion.registry import (
    IngestionRegistry,
)


def select_ingestion(
    requested: str | None = None,
) -> BaseIngestionPipeline:

    # CLI has highest priority.
    if requested:

        return (
            IngestionRegistry.create(
                requested
            )
        )

    # Environment variable
    env_pipeline = os.getenv(
        "INGESTION_PIPELINE"
    )

    if env_pipeline:

        return (
            IngestionRegistry.create(
                env_pipeline
            )
        )

    names = (
        IngestionRegistry.names()
    )

    if not names:
        raise RuntimeError(
            "No ingestion pipelines "
            "are registered."
        )

    print()
    print(
        "Select ingestion pipeline"
    )

    print(
        "-" * 30
    )

    for index, name in enumerate(
        names,
        start=1,
    ):

        pipeline = (
            IngestionRegistry.create(
                name
            )
        )

        print(
            f"{index}. "
            f"{name.upper()} - "
            f"{pipeline.description}"
        )

    while True:

        value = input(
            "\nSelect: "
        ).strip()

        try:

            index = (
                int(value) - 1
            )

            if (
                0
                <= index
                < len(names)
            ):

                selected = (
                    names[index]
                )

                pipeline = (
                    IngestionRegistry
                    .create(
                        selected
                    )
                )

                print()
                print(
                    "Using ingestion: "
                    f"{pipeline.name}"
                )

                return pipeline

        except ValueError:
            pass

        print(
            "Invalid selection."
        )