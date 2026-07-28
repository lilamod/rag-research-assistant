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
  between **Gemini** (default — free, no card required), **Anthropic
  (Claude)**, and **OpenAI** via one config flag.
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

> **Two backend hosting options are documented here:**
> - **Render** (below) — quick to set up, free tier has a 512MB RAM cap, so
>   it needs a remote embeddings API (Voyage/OpenAI) instead of local
>   embeddings, and those APIs rate-limit or bill past their free tier.
> - **Oracle Cloud Always Free VM** (see [`ORACLE_DEPLOY.md`](./ORACLE_DEPLOY.md)) —
>   more setup work (you manage the VM yourself via Docker), but gives you
>   enough RAM to run local embeddings with zero rate limits and zero
>   per-request cost, at any scale, forever. Better fit if this needs to be
>   reachable 24/7 for other people to use.

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
  `render.yaml` automatically (which also sets `PYTHON_VERSION` and defaults
  to Gemini for both chat and embeddings — just add your `GEMINI_API_KEY`),
  or
- Create a Web Service manually with:
  - **Build command:** `pip install -r requirements-cloud.txt`
  - **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
  - **Environment variables:** `PYTHON_VERSION=3.11.9`, `LLM_PROVIDER=gemini`,
    `EMBEDDING_PROVIDER=gemini`, `GEMINI_API_KEY=...` (get one free, no card,
    at https://aistudio.google.com/apikey), plus `CORS_ORIGINS` from the
    table below.

On any host, set its environment variables (`LLM_PROVIDER`, `GEMINI_API_KEY`,
etc. — see table below). Note the public URL you get, e.g.
`https://rag-backend.onrender.com`.

#### If you hit "Out of memory" or "No open ports detected"

The local embedding model (`EMBEDDING_PROVIDER=local`) pulls in `torch` +
`sentence-transformers`, which alone need well over 512MB of RAM just to
import — more than Render's (and many other) free tiers allow. If your
deploy log shows the build succeeding but then:
```
==> Out of memory (used over 512Mi)
==> No open ports detected, continuing to scan...
```
switch to a lightweight remote embeddings provider instead, which has no
heavy local dependency:
1. Install from `requirements-cloud.txt` instead of `requirements.txt`
   (build command: `pip install -r requirements-cloud.txt`) — this is
   already what `render.yaml` does.
2. Set `EMBEDDING_PROVIDER=gemini` and `GEMINI_API_KEY=...` in your host's
   environment variables (needed even if `LLM_PROVIDER=anthropic` — the two
   are independent). Gemini is recommended here because it has a genuinely
   free tier (1,500 requests/day) with **no card required at all** — get a
   key at https://aistudio.google.com/apikey. `EMBEDDING_PROVIDER=voyage`
   also works (free tier, 200M tokens, but rate-limited to 3 RPM unless you
   add a non-charging card), and `EMBEDDING_PROVIDER=openai` works too but
   has no free quota at all (needs a funded account).
3. Redeploy. Memory use drops enormously since no ML runtime needs to load.

If you'd rather keep local embeddings with zero external API calls at all
(no rate limits, ever, at any scale), you'll need a host with more RAM. See
[`ORACLE_DEPLOY.md`](./ORACLE_DEPLOY.md) for a genuinely free-forever option
(Oracle Cloud Always Free tier) — more setup work than Render, but no
rate limits or per-request billing at any usage level.

#### If you hit `voyageai.error.RateLimitError: ... reduced rate limits of 3 RPM`

This means you're on `EMBEDDING_PROVIDER=voyage` without a payment method on
file — Voyage throttles free-tier accounts with no card to 3 requests/min.
Your free 200M tokens still apply either way; adding a card only unlocks
normal rate limits (Voyage's own docs confirm the free tokens still apply
after adding one). If you'd rather not add a card anywhere, switch to
`EMBEDDING_PROVIDER=gemini` instead (see above) — no card needed at all.

#### If you hit `openai.RateLimitError: ... insufficient_quota`

This means your OpenAI account has no billing set up — the OpenAI **API** is
a separate, pay-as-you-go product from ChatGPT's free web tier, and it isn't
usable at all without a payment method on file, even for tiny amounts of
usage. Switch to `EMBEDDING_PROVIDER=gemini` (free, no card — see above), or
add a few dollars of credit at
https://platform.openai.com/settings/organization/billing/overview if you'd
rather keep using OpenAI.

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
| `LLM_PROVIDER`     | `gemini`                                      | `gemini` (free, no card, recommended), `anthropic`, or `openai` (both paid only) |
| `GEMINI_API_KEY`   | —                                              | required if `LLM_PROVIDER=gemini` and/or `EMBEDDING_PROVIDER=gemini` (same key covers both) - free at https://aistudio.google.com/apikey, no card needed |
| `GEMINI_MODEL`     | `gemini-2.5-flash`                            | used when `LLM_PROVIDER=gemini`          |
| `ANTHROPIC_API_KEY`| —                                              | required if `LLM_PROVIDER=anthropic` (needs a funded Anthropic account - no free API quota) |
| `ANTHROPIC_MODEL`  | `claude-sonnet-4-6`                           |                                            |
| `OPENAI_API_KEY`   | —                                              | required if `LLM_PROVIDER=openai` or `EMBEDDING_PROVIDER=openai` (needs a funded OpenAI account - no free API quota) |
| `OPENAI_MODEL`     | `gpt-4o-mini`                                 |                                            |
| `EMBEDDING_PROVIDER`| `gemini`                                     | `gemini` (free, no card, recommended), `voyage` (free tier, rate-limited without a card), `openai` (paid only), or `local` (sentence-transformers, needs 1GB+ RAM) |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001`                  | used when `EMBEDDING_PROVIDER=gemini`    |
| `EMBEDDING_MODEL`  | `sentence-transformers/all-MiniLM-L6-v2`      | used when `EMBEDDING_PROVIDER=local`     |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small`                | used when `EMBEDDING_PROVIDER=openai`    |
| `VOYAGE_API_KEY`   | —                                              | required if `EMBEDDING_PROVIDER=voyage` - free at https://dashboard.voyageai.com |
| `VOYAGE_EMBEDDING_MODEL` | `voyage-4-lite`                         | used when `EMBEDDING_PROVIDER=voyage`    |
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
