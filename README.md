# Archive — AI Research Assistant (RAG + LLM)

A full-stack Retrieval-Augmented Generation app. Upload documents (PDF, DOCX,
TXT, Markdown), ask questions in plain English, and get answers grounded in
your own sources — with inline citations pointing back to the exact chunk
they came from.

```
┌─────────────┐        REST (/api/*)        ┌──────────────────┐
│  React (Vite) │  <----------------------> │   FastAPI backend │
│  frontend/    │                            │   backend/         │
└─────────────┘                            └────────┬───────────┘
                                                       │
                                     ┌─────────────────┼─────────────────┐
                                     │                 │                 │
                              document_loader     embeddings        vector_store
                              (pdf/docx/txt)   (sentence-transformers)  (FAISS)
                                                       │
                                                     llm.py
                                              (Anthropic / OpenAI)
```

## Features

- **Document ingestion** — PDF, DOCX, TXT, and Markdown, extracted and split
  into overlapping chunks for good retrieval quality.
- **Local embeddings** — uses `sentence-transformers` on CPU, so no API key
  or per-document cost is needed just to index files.
- **Vector search** — a persistent FAISS index (cosine similarity via
  normalized inner product), saved to disk so your index survives restarts.
- **Grounded answers with citations** — the LLM is instructed to answer only
  from retrieved context and cite sources as `[1]`, `[2]`, etc. Pluggable
  between **Anthropic (Claude)** and **OpenAI** via one config flag.
- **Document management** — list and delete ingested documents from the UI.
- **A real frontend** — a separate React (Vite) single-page app, not a
  backend-rendered template — talking to the API over `fetch`.

## Project layout

```
rag-research-assistant/
├── backend/
│   ├── main.py            # FastAPI app & routes
│   ├── config.py          # env-driven settings
│   ├── document_loader.py # pdf/docx/txt/md -> raw text
│   ├── chunking.py        # recursive text splitter with overlap
│   ├── embeddings.py      # sentence-transformers wrapper
│   ├── vector_store.py    # FAISS index + metadata persistence
│   ├── llm.py             # Anthropic / OpenAI answer generation
│   └── rag_pipeline.py    # ties ingestion + retrieval + generation together
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── styles.css
│   │   └── components/    # Sidebar, UploadZone, DocumentList, ChatThread, ...
│   ├── index.html
│   ├── package.json
│   └── vite.config.js     # dev proxy: /api -> http://localhost:8000
├── data/
│   ├── uploads/           # raw uploaded files land here
│   └── index/             # FAISS index + metadata.json (persisted)
├── requirements.txt
├── .env.example
└── run.py                 # `python run.py` to start the backend
```

## Setup

### 1. Backend

```bash
cd rag-research-assistant
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER (anthropic|openai) and the matching API key
```

Start the API:

```bash
python run.py
# or: uvicorn backend.main:app --reload --port 8000
```

The backend runs at `http://localhost:8000`. Check `http://localhost:8000/api/health`.

### 2. Frontend

In a second terminal:

```bash
cd rag-research-assistant/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite's dev server proxies `/api/*` calls to the
backend on port 8000 (see `vite.config.js`), so no CORS setup is needed in dev.

### 3. Production build (optional)

To serve the frontend directly from FastAPI as one deployable app:

```bash
cd frontend
npm run build          # outputs frontend/dist
cd ..
python run.py           # FastAPI now also serves the built React app at /
```

## Configuration reference (`.env`)

| Variable          | Default                                       | Notes                                   |
|--------------------|-----------------------------------------------|------------------------------------------|
| `LLM_PROVIDER`     | `anthropic`                                   | `anthropic` or `openai`                  |
| `ANTHROPIC_API_KEY`| —                                              | required if provider is `anthropic`      |
| `ANTHROPIC_MODEL`  | `claude-sonnet-4-6`                           |                                            |
| `OPENAI_API_KEY`   | —                                              | required if provider is `openai`         |
| `OPENAI_MODEL`     | `gpt-4o-mini`                                 |                                            |
| `EMBEDDING_MODEL`  | `sentence-transformers/all-MiniLM-L6-v2`      | any sentence-transformers model, local   |
| `CHUNK_SIZE`       | `1000`                                        | characters per chunk                     |
| `CHUNK_OVERLAP`    | `150`                                         | characters shared between chunks         |
| `TOP_K`            | `5`                                           | chunks retrieved per question            |
| `UPLOAD_DIR`       | `data/uploads`                                |                                            |
| `INDEX_DIR`        | `data/index`                                  |                                            |

## API reference

| Method | Path                    | Description                                  |
|--------|-------------------------|-----------------------------------------------|
| GET    | `/api/health`           | Health check                                  |
| GET    | `/api/stats`            | `{ total_documents, total_chunks }`           |
| GET    | `/api/documents`        | List ingested documents                       |
| POST   | `/api/upload`           | Multipart upload, field name `files` (multiple)|
| DELETE | `/api/documents/{id}`   | Remove a document and its chunks from the index|
| POST   | `/api/chat`             | `{ "question": "...", "top_k": 5 }` → `{ answer, sources }` |

## How retrieval + generation works

1. On upload, a file is saved, text is extracted, split into overlapping
   chunks, embedded locally, and added to the FAISS index with metadata
   (`doc_id`, `filename`).
2. On a question, the query is embedded with the same model, the FAISS index
   returns the top-k most similar chunks, and those chunks are inserted into
   a prompt template (`backend/llm.py`) instructing the model to answer only
   from the given context and cite sources by number.
3. The API returns both the generated answer and the source chunks used, so
   the frontend can show exactly what backed each claim.

## Extending this project

- **Swap the vector store** for Chroma, Qdrant, or pgvector by reimplementing
  `vector_store.py`'s small interface (`add`, `search`, `list_documents`,
  `delete_document`).
- **Add streaming answers** by switching `llm.py` to the providers' streaming
  APIs and exposing a Server-Sent-Events or WebSocket endpoint.
- **Add auth / multi-user support** by namespacing the FAISS index and
  `data/uploads` by user ID.
- **Re-rank retrieved chunks** with a cross-encoder before generation for
  higher precision on large corpora.
