from local_rag.config import (
    REFERENCES_PATH,
)
from local_rag.references.citation_normalizer import (
    CitationNormalizer,
)
from local_rag.references.store import (
    ReferenceStore,
)


source = "GJETA-2025-0260 (1).pdf"

text = """
Schurig et al. [24] demonstrated electromagnetic
cloaking. Morphing wings were explored in [25].
Other studies are discussed in [24, 25].
"""


store = ReferenceStore(
    REFERENCES_PATH
)

normalizer = CitationNormalizer(
    store
)


result = normalizer.normalize(
    text=text,
    source=source,
)


print(result)