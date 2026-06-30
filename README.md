# SSNanopore-RAG

> A **local-first** Retrieval-Augmented Generation (RAG) system for the **solid-state
> nanopore** literature — ask questions about nanopore sequencing, biophysics, and
> electronics and get answers grounded in real papers, all running on your own machine.

> ⚠️ **Work in progress.** The individual building blocks below work today; the
> end-to-end pipeline that wires them together is still being assembled.

> 🪦 **Pinecone is shelved.** After hitting two upstream blockers I raised ([#678](https://github.com/pinecone-io/python-sdk/issues/678) and [#679](https://github.com/pinecone-io/python-sdk/issues/679)) (Unlikely to be fixed), `pinecone-local` forces an HTTPS host and can't create sparse indexes — I've given up on it; development proceeds mainly on **Qdrant** and **ChromaDB**.

---

## 🔑 Local-first, by design

The whole point of this project is that the **entire pipeline can run on your own
hardware** — embedding, vector search, reranking, and the language model itself. Cloud
services are strictly *opt-in* and never required to get a working system.

| Stage            | Local default                                          | Optional cloud   |
| ---------------- | ------------------------------------------------------ | ---------------- |
| **Embeddings**   | SPECTER2, BioBERT, MiniLM, SPLADE (HuggingFace, on-device) | Google Gemini    |
| **Vector store** | Qdrant (in-memory), ChromaDB (on-disk)                 | Pinecone         |
| **Reranking**    | ColBERT + BM25 (via fastembed, on-device)              | —                |
| **LLM**          | Any tool-calling model running on Ollama               | —                |

**Why local-first?**

- 🔒 **Private** — your documents and questions never leave your machine.
- 💸 **No keys, no bills** — no API signups, no per-token costs, no rate limits.
- 🔁 **Reproducible & offline** — local stores and pinned models work without a network.
- 🧪 **Hackable** — embeddings, vector stores, and retrieval strategies sit behind clean, swappable interfaces.

---

## ✅ What we have so far

Each piece is functional and can be used on its own while the full pipeline comes together.

### 1. Bibliography ingestion
Parses RIS bibliographic exports into clean, structured records — title, authors,
abstract, keywords, DOI, URL, publisher, date — ready to be embedded and indexed.

### 2. Pluggable embeddings & ranking models
One common interface, several interchangeable backends for embedding, lexical scoring, and reranking:

| Model        | Role     | Local | Notes                                                 |
| ------------ | -------- | :---: | ----------------------------------------------------- |
| **SPECTER2** | dense    | ✅    | Scientific-paper embeddings (`allenai/specter2`).      |
| **BioBERT**  | dense    | ✅    | Biomedical language model (`dmis-lab/biobert-v1.1`).   |
| **MiniLM**   | dense    | ✅    | Lightweight general-purpose sentence embeddings.       |
| **SPLADE**   | sparse   | ✅    | Learned sparse lexical expansion (`naver/splade-v3`).  |
| **BM25**     | lexical  | ✅    | Classic IDF-weighted term scoring, computed locally via fastembed (`Qdrant/bm25`). |
| **ColBERT**  | reranker | ✅    | Late-interaction reranking of candidates (`answerdotai/answerai-colbert-small-v1`). |
| **Gemini**   | dense    | ☁️    | Hosted Google embeddings — optional, needs an API key. |

### 3. Vector stores & retrieval strategies
Multiple stores behind a single `add_embeddings` / `query` interface, so you can compare
strategies on the same corpus and trade off recall, precision, and speed:

| Strategy             | What it does                                                            |
| -------------------- | ----------------------------------------------------------------------- |
| **Dense**            | Semantic search (cosine) over neural embeddings.                        |
| **Sparse (SPLADE)**  | Learned sparse lexical retrieval.                                       |
| **BM25**             | Classic IDF-weighted lexical search, computed locally via fastembed.    |
| **Hybrid**           | Dense + sparse, fused with Reciprocal Rank Fusion (RRF).                |
| **Rerank**           | Gathers candidates from dense + sparse + BM25, then reorders with a **ColBERT** late-interaction reranker. |

Backends available today: **Qdrant** (in-memory; dense, sparse, hybrid, BM25, and rerank),
**ChromaDB** (on-disk dense), and **Pinecone** (dense, optional).

### 4. Local LLM
A chat wrapper around [**Ollama**](https://ollama.com/) with a multi-step
tool/function-calling loop and a system prompt tuned for nanoscience — so the model can
reason and call tools before answering.

---

## 🔭 What we're building next

- Wiring the pieces into a single end-to-end **ingest → embed → retrieve → answer** flow.
- Grounding the local LLM's answers in retrieved passages (the full RAG loop, with citations).
- Making the embedding model, vector store, and retrieval strategy selectable through configuration.

---

## 🧰 Major packages

- **PyTorch** + **HuggingFace Transformers** / **adapters** — running embedding models on-device
- **qdrant-client** (with **fastembed**) — vector store, BM25, and ColBERT reranking
- **ChromaDB** — on-disk vector store
- **Pinecone** — optional hosted/local vector store
- **Ollama** — local LLM inference with tool-calling
- **Pydantic** — structured records
- **google-genai** — optional hosted embeddings
- **uv** — dependency management; **ruff** / **black** / **pre-commit** — code quality

---

## 🚀 Getting started

### Prerequisites

**Required**
- **Python 3.13+**
- [**uv**](https://github.com/astral-sh/uv) for dependency management
- [**Ollama**](https://ollama.com/) running locally — needed for the LLM step

**Optional**
- **Docker** — only needed to run a standalone vector-store server. The default Qdrant
  store runs **in-memory**, so you need nothing extra to get started; Docker is required
  only for a persistent Qdrant server or the Pinecone-local backend (see below).
- **A CUDA GPU** — the on-device embedding/reranking models run on CPU, but a GPU is much
  faster. A CUDA build of PyTorch is pinned in `pyproject.toml`; adjust it for CPU-only.

### Install
```bash
uv sync
```

### Pull a local model
```bash
ollama pull <your-model>      # any chat model with tool-calling support
```

### Optional: start a vector-store server
The default Qdrant store is in-memory and needs no setup. To run a persistent backend
instead, bring one up with the matching Compose profile:

```bash
docker compose --profile qdrant   up -d   # Qdrant server   → localhost:6333
docker compose --profile pinecone up -d   # Pinecone-local  → localhost:5080
```

### Bring your own corpus
Point the ingestion step at an RIS bibliographic export to build a structured corpus,
then index it with the embedding and vector-store backend of your choice. Your data and
any local indexes stay on your machine and out of version control.

### Try the building blocks
A couple of entry-point scripts let you exercise the layers independently:

```bash
uv run check      # runs the vector-store / retrieval layer (Qdrant rerank demo)
uv run check2     # runs the local Ollama LLM with tool-calling
```

> ℹ️ **Optional cloud:** to use the hosted embedding backend, supply an API key via a
> local `.env` file — it's picked up automatically. Everything else runs fully offline.

---

## 🛠️ Development

Linting and formatting are handled by **ruff** and **black** via **pre-commit**:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Ruff runs with a 100-char line length and pycodestyle, pyflakes, isort, pyupgrade,
bugbear, and simplify rules enabled.

---

## 📌 Status & notes

- The building blocks (ingestion, embeddings, vector stores, local LLM) work today; the
  single end-to-end entry point that connects them is still in progress.
- The Qdrant stores run **in-memory** — fast and zero-setup, but the index is rebuilt
  each run; a persisted/hosted client can be swapped in for durability.
- **The local Pinecone backend is partly blocked by upstream bugs** (both filed by the
  author, Deekshant), which is why some Pinecone classes are currently unusable:
  - [pinecone-io/python-sdk#678](https://github.com/pinecone-io/python-sdk/issues/678) —
    `pinecone-local` forces an `https://` data-plane host. Worked around by using the plain
    `Pinecone` client with `ssl_verify=False` and rewriting the host to `http://`.
  - [pinecone-io/python-sdk#679](https://github.com/pinecone-io/python-sdk/issues/679) —
    sparse index creation is impossible against `pinecone-local` (it is effectively
    dense-only), so the Pinecone **sparse** store is disabled.
  - **Use the Qdrant sparse/hybrid/rerank stores instead** — they support all of this locally.
- Interfaces and structure are still evolving as the project develops.
