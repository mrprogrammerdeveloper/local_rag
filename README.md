# Local Academic PDF RAG

A fully local Retrieval-Augmented Generation (RAG) system for question answering over academic PDF documents.

The project is designed for research and experimentation with:

- academic PDF parsing
- bibliography/reference extraction
- multilingual dense retrieval
- local vector search
- cross-encoder reranking
- sentence-level evidence tracking
- deterministic citation rendering
- local LLM generation with Ollama
- retrieval evaluation and parameter tuning

The current pipeline runs locally without sending document content to an external LLM API.

---

## Overview

The system separates the main body of an academic paper from its bibliography, indexes only the main content, and stores the paper's references separately.

At query time, it retrieves candidate chunks using dense vector search, reranks them with a multilingual reranker, converts the selected context into sentence-level evidence, and sends only that evidence to the local LLM.

The LLM cites evidence IDs such as `[E1]` and `[E2]`. Python then deterministically converts those IDs into either:

```text
[document.pdf, ref 24]
```

when the original paper contains a direct academic reference, or:

```text
[document.pdf, page 8]
```

when the supporting statement exists in the paper but has no direct reference marker.

This design reduces the chance of the LLM inventing or attaching the wrong reference number to a claim.

---

# Architecture

```text
                              PDF Files
                                  │
                                  ▼
                           ┌──────────────┐
                           │  PDFLoader   │
                           │   PyMuPDF    │
                           └──────┬───────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             ┌──────────────┐           ┌─────────────────┐
             │ReferenceParser│          │SectionSplitter  │
             │ Parse [1]...  │          │Remove bibliography
             └──────┬───────┘           └────────┬────────┘
                    │                            │
                    ▼                            ▼
             ┌──────────────┐             ┌──────────────┐
             │ReferenceStore│             │ TextChunker  │
             │ JSON per PDF │             │paragraph-aware
             └──────────────┘             └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │   BGE-M3     │
                                          │  Embeddings  │
                                          └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ Qdrant Local │
                                          └──────┬───────┘
                                                 │
                                                 ▼
Question ──► BGE-M3 query embedding ──► Dense retrieval
                                                 │
                                                 ▼
                                          Top-N candidates
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │BGE Reranker  │
                                          │    v2-m3     │
                                          └──────┬───────┘
                                                 │
                                                 ▼
                                             Top-K chunks
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │EvidenceBuilder│
                                          │ sentence-level
                                          └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ Qwen 3.5 4B  │
                                          │ via Ollama   │
                                          └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │CitationRenderer
                                          │ E1 -> ref/page
                                          └──────┬───────┘
                                                 │
                                                 ▼
                                      Answer + References
```

---

# Main Components

| Component | Technology |
|---|---|
| PDF parsing | PyMuPDF |
| Text cleaning | Python / regex |
| Chunking | Paragraph-aware custom chunker |
| Embeddings | `BAAI/bge-m3` |
| Vector database | Qdrant Local |
| Dense retrieval | Cosine similarity |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Local LLM | `qwen3.5:4b` |
| LLM runtime | Ollama |
| Citation handling | Deterministic Python pipeline |
| Evaluation | Custom retrieval evaluation scripts |

---

# Project Structure

```text
local_rag/
├── data/
│   ├── pdfs/
│   ├── vector_store/
│   ├── references/
│   └── evaluation/
│       └── retrieval_dataset.json
│
├── src/
│   └── local_rag/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── pdf_loader.py
│       │   ├── cleaner.py
│       │   ├── chunker.py
│       │   ├── reference_parser.py
│       │   └── section_splitter.py
│       │
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── embedder.py
│       │
│       ├── vector_store/
│       │   ├── __init__.py
│       │   └── qdrant.py
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── retriever.py
│       │   ├── reranker.py
│       │   ├── evidence.py
│       │   ├── evidence_builder.py
│       │   └── citation_extractor.py
│       │
│       ├── references/
│       │   ├── __init__.py
│       │   ├── store.py
│       │   ├── citation_normalizer.py
│       │   └── citation_renderer.py
│       │
│       ├── generation/
│       │   ├── __init__.py
│       │   └── prompt_builder.py
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   └── ollama_client.py
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── models.py
│           └── retrieval_evaluator.py
│
├── scripts/
│   ├── ingest.py
│   ├── chat.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_k_sweep.py
│   ├── test_config.py
│   ├── test_pdf.py
│   ├── test_chunker.py
│   ├── test_embedding.py
│   ├── test_vector_store.py
│   ├── test_references.py
│   ├── test_section_splitter.py
│   └── test_citation_normalizer.py
│
├── requirements.txt
├── pyproject.toml
├── .env
├── .env.example
├── .gitignore
└── README.md
```

---

# Prerequisites

Before installing the project, make sure you have:

- Python 3.11+  
- `pip`
- Git
- Ollama
- enough disk space for the embedding, reranker, and LLM model files
- internet access for the first model download

Recommended:

- Python 3.12
- 16 GB RAM minimum
- 32 GB RAM for a smoother local workflow
- NVIDIA GPU for local LLM acceleration

The reranker can run entirely on CPU.

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd local_rag
```

Replace `<YOUR_REPOSITORY_URL>` with the actual GitHub repository URL.

---

## 2. Create a virtual environment

Linux / WSL / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` is important because the project uses a `src/` package layout.

Without the editable install, imports such as:

```python
from local_rag.config import ...
```

may fail.

---

# Ollama Setup

Install Ollama from its official distribution for your operating system.

Pull the local LLM:

```bash
ollama pull <model>
```

Test it:

```bash
ollama run <model>
```

Ollama should expose its local HTTP service at:

```text
http://localhost:11434
```

You can verify it with:

```bash
curl http://localhost:11434/api/tags
```

If you are using WSL while Ollama runs on Windows, `localhost:11434` usually works in modern WSL environments. If it does not, configure `OLLAMA_HOST` using the reachable Windows host address.

---

# Environment Configuration

Create a `.env` file in the project root.

Example:

```env
OLLAMA_MODEL=<model>
OLLAMA_HOST=http://localhost:11434

PDF_PATH=data/pdfs
VECTOR_PATH=data/vector_store
REFERENCES_PATH=data/references

EMBEDDING_MODEL=BAAI/bge-m3

CHUNK_SIZE=500
CHUNK_OVERLAP=75

QDRANT_COLLECTION=documents

RETRIEVAL_CANDIDATES=12
TOP_K=5

RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cpu
RERANKER_BATCH_SIZE=4
```

---

# Configuration Reference

| Variable | Description | Current value |
|---|---|---|
| `OLLAMA_MODEL` | Local generation model | `<model>` |
| `OLLAMA_HOST` | Ollama HTTP endpoint | `http://localhost:11434` |
| `PDF_PATH` | Input PDF directory | `data/pdfs` |
| `VECTOR_PATH` | Qdrant Local storage | `data/vector_store` |
| `REFERENCES_PATH` | Parsed bibliography storage | `data/references` |
| `EMBEDDING_MODEL` | Dense embedding model | `BAAI/bge-m3` |
| `CHUNK_SIZE` | Approximate chunk size in words | `500` |
| `CHUNK_OVERLAP` | Word overlap between chunks | `75` |
| `QDRANT_COLLECTION` | Qdrant collection name | `documents` |
| `RETRIEVAL_CANDIDATES` | Candidates retrieved before reranking | `12` |
| `TOP_K` | Final chunks after reranking | `5` |
| `RERANKER_MODEL` | Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_DEVICE` | Reranker device | `cpu` |
| `RERANKER_BATCH_SIZE` | Reranker batch size | `4` |

`TOP_K=5` is based on the current preliminary K-sweep. It should be reevaluated when the evaluation dataset grows.

---

# Add PDF Documents

Place academic PDF files in:

```text
data/pdfs/
```

Example:

```text
data/pdfs/
├── paper_1.pdf
├── paper_2.pdf
└── paper_3.pdf
```

Do not commit copyrighted PDFs to a public repository unless you have permission to redistribute them.

---

# Ingest Documents

Run:

```bash
python3 scripts/ingest.py
```

The ingestion pipeline performs:

```text
PDF
 ↓
Text extraction
 ↓
Text cleaning
 ↓
Reference extraction
 ↓
Bibliography exclusion
 ↓
Paragraph-aware chunking
 ↓
BGE-M3 embeddings
 ↓
Qdrant Local indexing
```

Example output:

```text
Found 2 PDF file(s).
Loading embedding model...

Processing: example.pdf
Total text pages: 25
References parsed: 77
Main-content pages: 21
Reference-only pages excluded: 4
Chunks created: 34
Indexed 34 chunks.

Ingestion completed.
```

---

# Rebuild the Vector Store

Rebuild the vector store whenever you change:

- the embedding model
- chunking logic
- chunk size
- overlap
- bibliography filtering
- indexing policy
- document set, if you want a clean full rebuild

Delete the existing index:

```bash
rm -rf data/vector_store/*
```

Then ingest again:

```bash
python3 scripts/ingest.py
```

> Warning: this deletes the current local Qdrant index.

---

# Run the RAG Chat

After ingestion:

```bash
python3 scripts/chat.py
```

Example:

```text
Question: What are the aerospace applications of metamaterials?

Searching documents...
Retrieved 12 candidates.
Reranking candidates...
Generating answer...
```

Exit with:

```text
exit
```

or:

```text
quit
```

---

# Retrieval Pipeline

At query time:

```text
Question
 ↓
BGE-M3 query embedding
 ↓
Qdrant dense retrieval
 ↓
Top 12 candidates
 ↓
Deduplication
 ↓
BGE reranker
 ↓
Top 5 chunks
 ↓
Sentence-level evidence
 ↓
Qwen
```

The dense retriever is primarily responsible for recall.

The reranker is responsible for improving ranking precision and moving the most useful evidence toward the top.

---

# Reranking

The current reranker is:

```text
BAAI/bge-reranker-v2-m3
```

It is used as a cross-encoder over:

```text
question + candidate chunk
```

The reranker score is a raw relevance score, not a probability.

For example:

```text
page 2 | vector=0.6865 | rerank=3.8074
page 8 | vector=0.6843 | rerank=3.2019
page 9 | vector=0.6416 | rerank=2.4091
```

Higher reranker scores indicate greater relevance within that candidate set.

---

# Citation Design

The LLM does not choose final academic reference numbers directly.

Instead:

```text
Retrieved chunk
 ↓
EvidenceBuilder
 ↓
Sentence-level evidence

E1
E2
E3
...
 ↓
LLM answer using [E1], [E2], ...
 ↓
CitationRenderer
 ↓
Final citation
```

Example:

```text
[E7]
```

can be converted by Python into:

```text
[paper.pdf, ref 24]
```

or:

```text
[paper.pdf, page 8]
```

depending on whether the supporting sentence contains an original reference marker.

This reduces citation-number hallucination and prevents the model from selecting arbitrary bibliography entries from the same chunk.

---

# Reference Handling

Each PDF gets its own reference store.

Example:

```text
data/references/
└── GJETA-2025-0260 (1).json
```

Example JSON:

```json
{
  "24": "Schurig, D., Mock, J. J., Justice, B. J., ...",
  "25": "Jenett, B., Calisch, S., Cellucci, D., ...",
  "32": "Crawley, E. F., and De Luis, J. ..."
}
```

This avoids ambiguity when multiple documents contain the same reference number.

---

# Bibliography Exclusion

The bibliography is parsed and stored, but it is not indexed as answer evidence.

The pipeline is:

```text
Full PDF
 ├── ReferenceParser
 │      ↓
 │  ReferenceStore
 │
 └── SectionSplitter
        ↓
   Main content only
        ↓
      Chunker
        ↓
      Qdrant
```

This prevents bibliography entries from being retrieved as if they were part of the paper's scientific discussion.

---

# Tests

The project contains several test scripts.

## Configuration

```bash
python3 scripts/test_config.py
```

---

## PDF Parsing

```bash
python3 scripts/test_pdf.py
```

---

## Chunking

```bash
python3 scripts/test_chunker.py
```

---

## Embedding

```bash
python3 scripts/test_embedding.py
```

---

## Vector Store

```bash
python3 scripts/test_vector_store.py
```

---

## Reference Parsing

```bash
python3 scripts/test_references.py
```

Observed result for the main evaluation paper:

```text
Pages loaded: 25
References parsed: 77
```

The parser successfully extracted references from `[1]` through `[77]`.

---

## Citation Normalization

```bash
python3 scripts/test_citation_normalizer.py
```

---

## Section Splitter

```bash
python3 scripts/test_section_splitter.py
```

Observed result:

```text
All pages: 25
Main-content pages: 21

Indexed page numbers:
[1, 2, 3, ..., 20, 21]

Last indexed page:
21
```

The final indexed text ends with article content such as:

```text
Compliance with ethical standards
Acknowledgments
Disclosure of conflict of interest
```

rather than bibliography entries.

This confirms that the bibliography is excluded from the vector index while valid main-document text on the same page is preserved.

---

# Retrieval Evaluation

The evaluation dataset is stored in:

```text
data/evaluation/retrieval_dataset.json
```

Run:

```bash
python3 scripts/evaluate_retrieval.py
```

The current evaluation dataset contains 5 manually defined questions and relevant pages.

These results should therefore be treated as preliminary rather than final benchmark results.

---

# Preliminary Retrieval Results

## Dense Retrieval vs Dense Retrieval + Reranker

| Metric | Dense Retrieval | Dense + Reranker |
|---|---:|---:|
| Hit@K | 1.0000 | 1.0000 |
| Page Precision@K | 0.7833 | 0.8333 |
| Page Recall@K | 0.5833 | 0.7667 |
| MRR | 0.7500 | 1.0000 |

Observed changes:

```text
Page Recall:
0.5833 -> 0.7667

MRR:
0.7500 -> 1.0000
```

The reranker improved both retrieval coverage and the rank of the first relevant result in the current evaluation set.

---

# Example Reranking Improvement

Question:

```text
What is inverse design in metamaterials?
```

Dense retrieval:

```text
page 7
page 16
page 16
page 12

Recall@4 = 0.333
MRR      = 0.250
```

After reranking:

```text
page 12
page 7
page 11
page 8

Recall@4 = 0.667
MRR      = 1.000
```

The reranker moved a relevant result from rank 4 to rank 1.

---

# Another Retrieval Example

Question:

```text
How does additive manufacturing contribute to metamaterial development?
```

Dense retrieval:

```text
Recall@4 = 0.333
MRR      = 1.000
```

Dense retrieval + reranker:

```text
Recall@4 = 1.000
MRR      = 1.000
```

---

# K-Sweep Evaluation

Run:

```bash
python3 scripts/evaluate_k_sweep.py
```

Current results:

| K | Hit@K | Page Precision | Page Recall | MRR |
|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 0.533 | 1.000 |
| 3 | 1.000 | 1.000 | 0.717 | 1.000 |
| 4 | 1.000 | 0.833 | 0.767 | 1.000 |
| **5** | **1.000** | **0.800** | **0.833** | **1.000** |
| 6 | 1.000 | 0.740 | 0.833 | 1.000 |

Current choice:

```env
TOP_K=5
```

Why:

- K=5 improves recall compared with K=4.
- K=6 does not improve recall further.
- K=6 reduces precision.
- K=5 therefore provides a better trade-off in the current dataset.

This value should be revalidated on a larger evaluation set.

---

# Evaluation Metric Note

The current implementation evaluates retrieved **pages**, not gold chunks.

The metric currently called `Precision@K` in the evaluation code is based on unique retrieved pages.

A more precise interpretation is:

```text
Page Precision@K
```

Likewise:

```text
Page Recall@K
```

This distinction matters when multiple retrieved chunks come from the same page.

---

# Example RAG Output

Question:

```text
What are the aerospace applications of metamaterials?
```

Example answer:

```text
In aerospace, metamaterials enable lightweight components with
superior vibration damping and radar absorption, which enhances
stealth and fuel efficiency
[GJETA-2025-0260 (1).pdf, page 2].

Schurig et al. demonstrated the use of metamaterials for
electromagnetic cloaking
[GJETA-2025-0260 (1).pdf, ref 24].

Jenett et al. explored how mechanical metamaterials enable
morphing wings
[GJETA-2025-0260 (1).pdf, ref 25].
```

Generated references:

```text
[GJETA-2025-0260 (1).pdf, ref 24]
Schurig, D., Mock, J. J., Justice, B. J., Cummer, S. A.,
Pendry, J. B., Starr, A. F., and Smith, D. R. (2006).
Metamaterial electromagnetic cloak at microwave frequencies.
Science, 314(5801), 977-980.

[GJETA-2025-0260 (1).pdf, ref 25]
Jenett, B., Calisch, S., Cellucci, D., Cramer, N.,
Gershenfeld, N., Swei, S., and Cheung, K. C. (2017).
Digital morphing wing: active wing shaping concept using
composite lattice-based cellular structures.
Soft Robotics, 4(1), 33-48.
```

Example retrieved context:

```text
GJETA-2025-0260 (1).pdf | page 2
vector=0.6865 | rerank=3.8074

GJETA-2025-0260 (1).pdf | page 8
vector=0.6843 | rerank=3.2019

GJETA-2025-0260 (1).pdf | page 2
vector=0.6513 | rerank=2.5907

GJETA-2025-0260 (1).pdf | page 9
vector=0.6416 | rerank=2.4091
```

---

# LLM Generation Settings

The current local LLM is:

```text
qwen3.5:4b
```

Recommended generation configuration:

```python
options={
    "temperature": 0.1,
    "num_ctx": 8192,
    "num_predict": 768,
}
```

Thinking mode is disabled:

```python
think=False
```

Debug output can include:

```text
Done reason
Prompt tokens
Generated tokens
```

Example successful generation:

```text
Done reason: stop
Prompt tokens: 2836
Generated tokens: 173
```

---

# Hugging Face Downloads

On first run, the embedding and reranker models are downloaded from Hugging Face.

You may see:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

This does not prevent public models from loading.

To increase rate limits, configure a Hugging Face token in your shell:

```bash
export HF_TOKEN=<YOUR_TOKEN>
```

Do not commit the token to Git.

---

# First-Run Expectations

The first execution may take significantly longer because the system needs to download and cache:

- BGE-M3
- BGE reranker v2-m3
- Ollama model weights, if not already installed

Subsequent runs should load the cached models.

CPU reranking is intentionally slower than GPU reranking.

---

# Troubleshooting

## `ModuleNotFoundError: No module named 'local_rag'`

Run:

```bash
pip install -e .
```

from the repository root.

---

## Ollama connection error

Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

Also confirm:

```env
OLLAMA_HOST=http://localhost:11434
```

---

## Hugging Face authentication warning

This is only a warning for public models.

Optionally set:

```bash
export HF_TOKEN=<YOUR_TOKEN>
```

---

## Old chunks still appear after changing ingestion logic

Delete the local vector store and rebuild:

```bash
rm -rf data/vector_store/*
python3 scripts/ingest.py
```

---

## Reranker is slow

The default tested configuration runs:

```env
RERANKER_DEVICE=cpu
```

This is intentional for low-VRAM GPUs.

If you have enough VRAM and a compatible PyTorch installation, you can experiment with:

```env
RERANKER_DEVICE=cuda
```

Memory requirements should be tested on your hardware.

---

## References are not extracted from a PDF

The current reference parser is optimized for numbered bibliographies such as:

```text
References
[1] ...
[2] ...
```

Documents using different bibliography styles may require additional parsing rules.

---

## `References parsed: 0`

This means the current parser did not detect a supported bibliography structure.

The document may still be indexed, but original-reference citation resolution will not be available for that file.

---

# What Should Not Be Committed

For a public GitHub repository, avoid committing:

```text
.env
data/vector_store/
large local model files
Hugging Face cache files
Ollama model files
copyrighted PDFs without redistribution permission
private evaluation data
```

A typical `.gitignore` should include at least:

```gitignore
.env
.venv/
venv/
__pycache__/
*.pyc

data/vector_store/

.DS_Store
.idea/
.vscode/
```

Depending on your use case, you may also want:

```gitignore
data/pdfs/*
!data/pdfs/.gitkeep
```

---

# Reproducibility Notes

For reproducible experiments:

1. keep the same PDF set
2. keep the same embedding model
3. keep the same reranker model
4. keep the same chunk size and overlap
5. rebuild the vector store after ingestion changes
6. keep the evaluation dataset versioned
7. record `TOP_K` and candidate-count settings
8. pin dependency versions before publishing final benchmark results

---

# Current Limitations

The project is still under active development.

Current limitations:

1. the evaluation dataset currently contains only 5 questions
2. `TOP_K=5` is still a preliminary choice
3. candidate-count sweep has not been completed yet
4. generation quality evaluation has not been implemented yet
5. citation precision and citation recall are not yet reported as separate metrics
6. hallucination rate is not yet measured automatically
7. hybrid BM25 + dense retrieval is not yet implemented
8. document metadata extraction is not yet implemented
9. bibliography parsing currently assumes supported numbered-reference formats
10. retrieval evaluation is page-based rather than gold-chunk-based
11. scanned/image-only PDFs are not currently handled with OCR

---

# Roadmap

```text
[✓] PDF parsing
[✓] Text cleaning
[✓] Paragraph-aware chunking
[✓] BGE-M3 embeddings
[✓] Qdrant Local
[✓] Dense retrieval
[✓] Candidate retrieval
[✓] Deduplication
[✓] BGE reranking
[✓] Bibliography exclusion
[✓] Reference parsing
[✓] Per-document reference storage
[✓] Sentence-level evidence generation
[✓] Deterministic citation rendering
[✓] Retrieval evaluation
[✓] Dense vs reranker evaluation
[✓] K-sweep evaluation

[ ] Candidate-count sweep
[ ] Larger evaluation dataset
[ ] Generation evaluation
[ ] Groundedness / faithfulness evaluation
[ ] Citation precision and recall
[ ] Hallucination evaluation
[ ] Hybrid BM25 + dense retrieval
[ ] Document metadata extraction
[ ] Support for additional bibliography formats
```

---

# Planned Evaluation

The final evaluation set should include multiple question types:

```text
Definition
Factual
Application
Comparison
Cause and effect
Methods
Challenges
Broad questions
Narrow questions
```

A reasonable target is:

```text
30-50 manually verified evaluation questions
```

Planned experiments:

```text
Experiment A
BGE-M3 + Qdrant

Experiment B
BGE-M3 + Qdrant + BGE Reranker

Experiment C
Hybrid Retrieval + Reranker
```

---

# Current Evaluation Status

The current results are preliminary.

They are useful for validating the retrieval pipeline, but they are not large enough to claim final benchmark performance.

Current observed improvement from reranking:

```text
Page Recall:
0.5833 -> 0.7667

MRR:
0.7500 -> 1.0000
```

---

# Quick Start

For an already configured environment:

```bash
source .venv/bin/activate
```

Add PDFs to:

```text
data/pdfs/
```

Build the index:

```bash
python3 scripts/ingest.py
```

Start the chat:

```bash
python3 scripts/chat.py
```

Run retrieval evaluation:

```bash
python3 scripts/evaluate_retrieval.py
```

Run K-sweep:

```bash
python3 scripts/evaluate_k_sweep.py
```

---

# Security and Privacy

The RAG pipeline is designed to run locally.

Document text is processed locally by:

- PyMuPDF
- BGE-M3
- Qdrant Local
- BGE reranker
- Ollama

However, model files may be downloaded from external model repositories during setup.

If you work with sensitive documents, review your model-download, logging, cache, and source-control policies before use.

---

# License

Add a project license before publishing the repository if you want others to reuse, modify, or redistribute the code.

Also review the licenses and usage terms of all third-party models and libraries before commercial use.

---

# Notes for Contributors

When changing retrieval or ingestion behavior:

1. add or update tests
2. rebuild the vector store
3. rerun retrieval evaluation
4. rerun K-sweep if ranking behavior changed
5. document metric changes in the README or experiment logs

Avoid presenting results from the current 5-question dataset as final benchmark performance.

---

# Summary

The current system provides an end-to-end local academic RAG pipeline that:

- parses academic PDFs
- extracts and stores numbered references
- excludes bibliographies from answer evidence
- performs multilingual dense retrieval
- reranks candidate chunks with a multilingual cross-encoder
- builds sentence-level evidence
- generates answers locally with Qwen through Ollama
- maps evidence deterministically to academic references or page citations
- prints full references used in the answer
- evaluates retrieval quality numerically
- supports K-sweep experiments for retrieval tuning

The project is intended as a research-oriented foundation for building a more rigorous academic RAG system with stronger retrieval, citation, and generation evaluation.
