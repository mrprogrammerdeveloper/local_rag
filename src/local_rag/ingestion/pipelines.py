from local_rag.ingestion.pipeline_v1 import (
    LegacyIngestionPipeline,
)

from local_rag.ingestion.pipeline_v2 import (
    StructureAwareIngestionPipeline,
)

from local_rag.ingestion.pipeline_v2_block import (
    BlockAwareIngestionPipeline,
)

from local_rag.ingestion.registry import (
    IngestionRegistry,
)


def register_ingestion_pipelines(
) -> None:

    IngestionRegistry.register(
        name="v1",
        factory=lambda: (
            LegacyIngestionPipeline(
                chunk_size=500,
                overlap=75,
            )
        ),
    )

    IngestionRegistry.register(
        name="v2-basic",
        factory=lambda: (
            StructureAwareIngestionPipeline()
        ),
    )

    IngestionRegistry.register(
        name="v2-block",
        factory=lambda: (
            BlockAwareIngestionPipeline()
        ),
    )