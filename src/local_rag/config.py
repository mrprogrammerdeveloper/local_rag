import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Ollama
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3.5:4b"
)


# Documents
PDF_DIRECTORY = BASE_DIR / os.getenv(
    "PDF_PATH",
    "data/pdfs"
)


# Vector Store
VECTOR_STORE_PATH = BASE_DIR / os.getenv(
    "VECTOR_PATH",
    "data/vector_store"
)


# Embedding
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3"
)


# Chunking
CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "500"
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "75"
    )
)


# Retrieval
TOP_K = int(
    os.getenv(
        "TOP_K",
        "5"
    )
)


QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "documents",
)

VECTOR_STORE_PATH = BASE_DIR / os.getenv(
    "VECTOR_PATH",
    "data/vector_store",
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "documents",
)

REFERENCES_PATH = BASE_DIR / os.getenv(
    "REFERENCES_PATH",
    "data/references",
)

RETRIEVAL_CANDIDATES = int(
    os.getenv(
        "RETRIEVAL_CANDIDATES",
        "12",
    )
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "jinaai/jina-reranker-v2-base-multilingual",
)

RERANKER_DEVICE = os.getenv(
    "RERANKER_DEVICE",
    "cpu",
)

RERANKER_BATCH_SIZE = int(
    os.getenv(
        "RERANKER_BATCH_SIZE",
        "4",
    )
)

PDF_PATH = BASE_DIR / os.getenv(
    "PDF_PATH",
    "data/pdfs",
)

REFERENCES_PATH = BASE_DIR / os.getenv(
    "REFERENCES_PATH",
    "data/references",
)