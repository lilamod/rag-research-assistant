"""
AI Research Assistant (RAG + LLM) - FastAPI backend.

Endpoints:
  GET  /api/health              - health check
  GET  /api/stats               - index stats (doc/chunk counts)
  GET  /api/documents           - list ingested documents
  POST /api/upload               - upload + ingest one or more documents
  DELETE /api/documents/{doc_id}- remove a document from the index
  POST /api/chat                 - ask a question, get a grounded answer + sources

Also serves the static frontend at /.
"""
import json
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from . import rag_pipeline
from .document_loader import UnsupportedFileTypeError

app = FastAPI(title="AI Research Assistant (RAG + LLM)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    return rag_pipeline.get_stats()


@app.get("/api/documents")
def list_documents():
    return {"documents": rag_pipeline.list_documents()}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    deleted = rag_pipeline.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id}


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    errors = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            errors.append({"filename": upload.filename, "error": f"Unsupported type '{suffix}'"})
            continue

        temp_name = f"{uuid.uuid4()}{suffix}"
        dest_path = settings.UPLOAD_DIR / temp_name
        try:
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            result = rag_pipeline.ingest_document(dest_path, upload.filename)
            results.append(result)
        except (UnsupportedFileTypeError, ValueError) as e:
            errors.append({"filename": upload.filename, "error": str(e)})
        except Exception as e:
            errors.append({"filename": upload.filename, "error": f"Unexpected error: {e}"})

    if not results and errors:
        raise HTTPException(status_code=400, detail=errors)

    return {"ingested": results, "errors": errors}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        result = rag_pipeline.answer_question(request.question, top_k=request.top_k)
    except RuntimeError as e:
        # e.g. missing API key
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    def event_generator():
        try:
            for event_type, payload in rag_pipeline.answer_question_stream(
                request.question, top_k=request.top_k
            ):
                yield f"data: {json.dumps({'type': event_type, event_type: payload})}\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Unexpected error: {e}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# In production, serve the built React app (frontend/dist) at the root path.
# In development, run the frontend separately with `npm run dev` (Vite on :5173,
# proxying /api to this server) — nothing to mount here in that case.
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
