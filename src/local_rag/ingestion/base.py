from abc import (
    ABC,
    abstractmethod,
)

from pathlib import Path

from local_rag.ingestion.models import (
    IngestionResult,
)


class BaseIngestionPipeline(ABC):

    name: str

    description: str

    @abstractmethod
    def ingest(
        self,
        file_path: str | Path,
    ) -> IngestionResult:
        raise NotImplementedError