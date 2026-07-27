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

## Deploying (frontend on Vercel + backend elsewhere)

Vercel only hosts static sites and serverless functions — it can't run the
FastAPI backend, its FAISS index, or the local embedding model long-term. So
the split is:

- **Frontend → Vercel** (the `frontend/` folder)
- **Backend → a regular server host** that can run a long-lived Python
  process, e.g. Render, Railway, Fly.io, an EC2/DigitalOcean box, or a
  container platform.

### 1. Deploy the backend first

Pick a host and deploy the `backend/` app. Two files are included to make this
smooth on Render specifically: `.python-version` (pins Python to 3.11.9) and
`render.yaml` (a ready-made service blueprint). Both matter because Render's
current default Python (3.14.x) doesn't have prebuilt wheels yet for pinned
versions of `numpy`/`faiss-cpu` in `requirements.txt`, which fails the build
with an error like:
```
ERROR: Could not find a version that satisfies the requirement faiss-cpu==1.8.0.post1
```

**Important:** Render dropped support for `runtime.txt` — as of now it only
reads the Python version from, in order of precedence:
1. The `PYTHON_VERSION` environment variable (must be a full version like `3.11.9`)
2. A `.python-version` file at the repo root (this repo includes one)

If you created your Render service **manually** (not via the `render.yaml`
Blueprint), the `.python-version` file alone should be enough — but if the
build still shows `Using Python version 3.14.3 (default)`, go to your
service's **Environment** tab in the Render dashboard and add:
```
PYTHON_VERSION=3.11.9
```
then trigger a fresh deploy ("Manual Deploy → Clear build cache & deploy").
The env var always wins over the file, so this is the most reliable fix.

**On Render**, either:
- Click "New → Blueprint", point it at this repo, and it'll read
  `render.yaml` automatically (which also sets `PYTHON_VERSION`), or
- Create a Web Service manually with:
  - **Build command:** `pip install -r requirements.txt`
  - **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
  - **Environment variable:** `PYTHON_VERSION=3.11.9`, plus `LLM_PROVIDER`,
    `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, etc. from the table below.

On any host, set its environment variables (`LLM_PROVIDER`, `ANTHROPIC_API_KEY`
/ `OPENAI_API_KEY`, etc. — see table below). Note the public URL you get,
e.g. `https://rag-backend.onrender.com`.

#### If you hit "Out of memory" or "No open ports detected"

The local embedding model (`EMBEDDING_PROVIDER=local`) pulls in `torch` +
`sentence-transformers`, which alone need well over 512MB of RAM just to
import — more than Render's (and many other) free tiers allow. If your
deploy log shows the build succeeding but then:
```
==> Out of memory (used over 512Mi)
==> No open ports detected, continuing to scan...
```
switch to OpenAI embeddings instead, which have no heavy local dependency:
1. Install from `requirements-cloud.txt` instead of `requirements.txt`
   (build command: `pip install -r requirements-cloud.txt`) — this is
   already what `render.yaml` does.
2. Set `EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY=...` in your host's
   environment variables (needed even if `LLM_PROVIDER=anthropic` — the two
   are independent).
3. Redeploy. Memory use drops enormously since no ML runtime needs to load.

If you'd rather keep local embeddings, you'll need a host/plan with at least
~1–2GB RAM (Render's paid plans, Railway, Fly.io with a bigger machine, etc.).

#### If you hit `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`

This is a known break between older `anthropic`/`openai` SDK releases and
`httpx` 0.28+, which removed a deprecated argument those older SDK versions
still pass internally. It surfaces the first time the app tries to create an
Anthropic or OpenAI client — e.g. on your first chat request. Fixed by
upgrading past the versions that had the bug: `anthropic>=0.40.0` and
`openai>=1.55.3` (this repo pins `anthropic==0.40.0` and `openai==1.109.1`,
both well past the fix). If you still see it, you likely have an older
`requirements.txt`/`requirements-cloud.txt` deployed — redeploy with
"Clear build cache" so pip actually re-resolves the pinned versions.

Also set, on the backend host:
```
CORS_ORIGINS=https://your-app.vercel.app
```
(You can add your Vercel preview-deployment pattern too, comma-separated,
once you know it — or leave `CORS_ORIGINS=*` while testing.)

> Note: `data/uploads` and `data/index` are local disk. Most of these hosts
> reset local disk on redeploy/restart unless you attach a persistent volume
> or switch `vector_store.py` to a hosted vector DB. For a quick demo this is
> fine; for production, mount a persistent disk or swap the store.

### 2. Deploy the frontend on Vercel

From the `frontend/` folder (this repo already includes `vercel.json`):

1. Push the repo to GitHub (or run `vercel` CLI directly from `frontend/`).
2. In Vercel: **New Project → import this repo → set Root Directory to `frontend`**.
3. Vercel auto-detects Vite (build command `npm run build`, output `dist`,
   both already declared in `frontend/vercel.json`).
4. Add an **Environment Variable**: `VITE_API_URL` = your backend's URL from
   step 1 (e.g. `https://rag-backend.onrender.com`, no trailing slash). Add it
   for Production, Preview, and Development as needed.
5. Deploy.

The frontend calls `${VITE_API_URL}/api/...` when that variable is set (see
`frontend/src/api.js`), and falls back to relative `/api/...` (via the Vite
dev proxy) when it isn't — so local dev is unaffected.

### CLI alternative

```bash
cd frontend
npm i -g vercel
vercel link
vercel env add VITE_API_URL production   # paste your backend URL
vercel --prod
```

## Configuration reference (`.env`)

| Variable          | Default                                       | Notes                                   |
|--------------------|-----------------------------------------------|------------------------------------------|
| `LLM_PROVIDER`     | `anthropic`                                   | `anthropic` or `openai`                  |
| `ANTHROPIC_API_KEY`| —                                              | required if provider is `anthropic`      |
| `ANTHROPIC_MODEL`  | `claude-sonnet-4-6`                           |                                            |
| `OPENAI_API_KEY`   | —                                              | required if `LLM_PROVIDER=openai` or `EMBEDDING_PROVIDER=openai` |
| `OPENAI_MODEL`     | `gpt-4o-mini`                                 |                                            |
| `EMBEDDING_PROVIDER`| `local`                                      | `local` (sentence-transformers) or `openai` (lightweight, recommended for small hosts) |
| `EMBEDDING_MODEL`  | `sentence-transformers/all-MiniLM-L6-v2`      | used when `EMBEDDING_PROVIDER=local`     |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small`                | used when `EMBEDDING_PROVIDER=openai`    |
| `CHUNK_SIZE`       | `1000`                                        | characters per chunk                     |
| `CHUNK_OVERLAP`    | `150`                                         | characters shared between chunks         |
| `TOP_K`            | `5`                                           | chunks retrieved per question            |
| `UPLOAD_DIR`       | `data/uploads`                                |                                            |
| `INDEX_DIR`        | `data/index`                                  |                                            |
| `CORS_ORIGINS`     | `*`                                           | comma-separated allowed origins          |
| `RELOAD`           | `false`                                       | set `true` for local dev auto-reload; keep off in production |

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
