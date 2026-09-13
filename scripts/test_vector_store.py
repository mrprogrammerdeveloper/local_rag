from pathlib import Path

from local_rag.embeddings.embedder import Embedder
from local_rag.ingestion.chunker import TextChunker
from local_rag.ingestion.pdf_loader import PDFLoader
from local_rag.vector_store.qdrant import QdrantVectorStore


pdf_path = Path(
    "data/pdfs/test.pdf"
)


print("Loading PDF...")

loader = PDFLoader()

pages = loader.load(
    pdf_path
)

print(
    f"Pages: {len(pages)}"
)


print("Creating chunks...")

chunker = TextChunker(
    chunk_size=500,
    overlap=75,
)

chunks = chunker.split(
    pages
)

print(
    f"Chunks: {len(chunks)}"
)


print("Loading embedding model...")

embedder = Embedder()


print("Creating embeddings...")

texts = [
    chunk.content
    for chunk in chunks
]

vectors = embedder.embed_texts(
    texts
)

print(
    f"Embeddings: {len(vectors)}"
)

print(
    f"Vector dimension: {len(vectors[0])}"
)


print("Saving to Qdrant...")

vector_store = QdrantVectorStore(
    vector_size=len(vectors[0]),
)

vector_store.upsert(
    chunks=chunks,
    vectors=vectors,
)


print("Done.")


print("\nSearching...")


question = (
    "How was finite element analysis used?"
)


query_vector = embedder.embed_text(
    question
)


results = vector_store.search(
    query_vector=query_vector,
    limit=5,
)


for index, result in enumerate(
    results,
    start=1,
):
    print(
        "\n"
        + "=" * 60
    )

    print(
        f"Result {index}"
    )

    print(
        f"Score: {result.score}"
    )

    print(
        f"Source: {result.payload['source']}"
    )

    print(
        f"Page: {result.payload['page_number']}"
    )

    print(
        f"Chunk: {result.payload['chunk_index']}"
    )

    print()

    print(
        result.payload["content"][:500]
    )