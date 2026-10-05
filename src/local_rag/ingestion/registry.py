from collections.abc import (
    Callable,
)

from local_rag.ingestion.base import (
    BaseIngestionPipeline,
)


PipelineFactory = Callable[
    [],
    BaseIngestionPipeline,
]


class IngestionRegistry:

    _pipelines: dict[
        str,
        PipelineFactory,
    ] = {}

    @classmethod
    def register(
        cls,
        name: str,
        factory: PipelineFactory,
    ) -> None:

        key = (
            name.strip().lower()
        )

        if not key:
            raise ValueError(
                "Pipeline name "
                "cannot be empty."
            )

        cls._pipelines[
            key
        ] = factory

    @classmethod
    def create(
        cls,
        name: str,
    ) -> BaseIngestionPipeline:

        key = (
            name.strip().lower()
        )

        factory = (
            cls._pipelines.get(
                key
            )
        )

        if factory is None:

            available = ", ".join(
                cls.names()
            )

            raise ValueError(
                f"Unknown ingestion "
                f"pipeline: {name}. "
                f"Available: {available}"
            )

        return factory()

    @classmethod
    def names(
        cls,
    ) -> list[str]:

        return list(
            cls._pipelines.keys()
        )