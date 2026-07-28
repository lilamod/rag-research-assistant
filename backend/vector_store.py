"""
Persistent vector store backed by FAISS (IndexFlatIP for cosine similarity,
since embeddings are L2-normalized). Chunk text + metadata are stored
alongside the index in a JSON sidecar file, together with a small schema
header (dimension + version) so a provider/dimension change is detected and
triggers a clean rebuild instead of silently returning garbage results.
"""
import json
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from .config import settings
from .embeddings import embedding_dim

METADATA_SCHEMA_VERSION = 1


class VectorStore:
    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = index_dir or settings.INDEX_DIR
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "metadata.json"
        self._lock = threading.Lock()

        self.dim = embedding_dim()
        self.index: faiss.Index
        self.records: List[Dict] = []  # parallel to FAISS vector order

        self._load_or_create()

    def _load_or_create(self):
        if self.index_path.exists() and self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if isinstance(meta, list):
                # Old format from before dimension tracking was added: a
                # bare list of records, no way to verify the stored
                # dimension. Safer to rebuild than assume compatibility.
                stored_dim = None
            else:
                stored_dim = meta.get("dim")

            if stored_dim != self.dim:
                # EMBEDDING_PROVIDER (or its dimension) changed since this
                # index was built - the old vectors are a different size and
                # comparing them against new query vectors would silently
                # produce meaningless similarity scores. Safer to start
                # fresh than to load incompatible data.
                print(
                    f"[vector_store] Index dimension mismatch (stored={stored_dim}, "
                    f"current={self.dim}) - rebuilding empty index. Re-upload your "
                    f"documents to restore search."
                )
                self.index = faiss.IndexFlatIP(self.dim)
                self.records = []
                self._save()
                return

            self.index = faiss.read_index(str(self.index_path))
            self.records = meta.get("records", [])
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.records = []

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        meta = {
            "version": METADATA_SCHEMA_VERSION,
            "dim": self.dim,
            "records": self.records,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def add(self, vectors: np.ndarray, chunk_texts: List[str], doc_metadata: Dict) -> int:
        """Add chunk vectors + their text/metadata. Returns number added."""
        if vectors.shape[0] == 0:
            return 0
        with self._lock:
            self.index.add(vectors)
            for text in chunk_texts:
                self.records.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "doc_id": doc_metadata["doc_id"],
                        "filename": doc_metadata["filename"],
                    }
                )
            self._save()
        return len(chunk_texts)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        with self._lock:
            if self.index.ntotal == 0:
                return []
            top_k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(
                query_vector.reshape(1, -1).astype("float32"), top_k
            )
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((self.records[idx], float(score)))
            return results

    def list_documents(self) -> List[Dict]:
        seen = {}
        for rec in self.records:
            doc_id = rec["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {"doc_id": doc_id, "filename": rec["filename"], "chunks": 0}
            seen[doc_id]["chunks"] += 1
        return list(seen.values())

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks for a document and rebuild the index.

        Rebuilds by reconstructing each kept vector directly from the
        existing FAISS index (index.reconstruct) rather than re-calling the
        embeddings API - deleting one document should never re-spend API
        quota/rate-limit budget on every other document you happen to have
        indexed.
        """
        with self._lock:
            keep_positions = [
                i for i, r in enumerate(self.records) if r["doc_id"] != doc_id
            ]
            if len(keep_positions) == len(self.records):
                return False  # nothing matched

            new_index = faiss.IndexFlatIP(self.dim)
            if keep_positions:
                vectors = np.vstack(
                    [self.index.reconstruct(i) for i in keep_positions]
                ).astype("float32")
                new_index.add(vectors)

            self.index = new_index
            self.records = [self.records[i] for i in keep_positions]
            self._save()
            return True

    def stats(self) -> Dict:
        return {
            "total_chunks": len(self.records),
            "total_documents": len(self.list_documents()),
        }
