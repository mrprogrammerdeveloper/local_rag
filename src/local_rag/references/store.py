import json
from pathlib import Path

from local_rag.ingestion.models import DocumentReference


class ReferenceStore:
    def __init__(
        self,
        directory: str | Path,
    ):
        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        source: str,
        references: list[DocumentReference],
    ) -> None:

        path = self._get_path(
            source
        )

        data = {
            str(reference.number): reference.content
            for reference in references
        }

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(
        self,
        source: str,
    ) -> dict[int, str]:

        path = self._get_path(
            source
        )

        if not path.exists():
            return {}

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return {
            int(number): content
            for number, content in data.items()
        }

    def get(
        self,
        source: str,
        number: int,
    ) -> str | None:

        references = self.load(
            source
        )

        return references.get(
            number
        )

    def _get_path(
        self,
        source: str,
    ) -> Path:

        name = Path(source).stem

        return (
            self.directory
            / f"{name}.json"
        )