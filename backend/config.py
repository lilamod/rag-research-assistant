"""
Centralized configuration for the RAG Research Assistant.
All settings are loaded from environment variables (see .env.example).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    # LLM (chat generation) - Gemini only. Free tier, no card required -
    # get a key at https://aistudio.google.com/apikey
    LLM_PROVIDER: str = "gemini"

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    # Lower temperature = more focused, faster responses (less creative sampling)
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    # Higher top_p narrows the sampling window → faster token generation
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.85"))

    # Embeddings
    # "gemini" -> Google's Gemini embeddings API, lightweight install, same
    #             GEMINI_API_KEY as above. Genuinely free tier (1,500
    #             requests/day), no card required. Default.
    # "local"  -> sentence-transformers, runs on CPU, no API key, no rate
    #             limits at any scale, but pulls in torch (heavy: needs
    #             1GB+ RAM). Too heavy for small free-tier hosts, but the
    #             right choice for a self-hosted VM with RAM to spare
    #             (see ORACLE_DEPLOY.md).
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # gemini-embedding-001 is deprecated. Use text-embedding-004 which is
    # faster and still supports 768 dim for quality or 384 dim for speed.
    GEMINI_EMBEDDING_MODEL: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "text-embedding-004"
    )
    # 384 is ~2x faster than 768 with minimal quality loss for retrieval.
    # Bump to 768 if you have very nuanced queries and need the extra precision.
    GEMINI_EMBEDDING_DIM: int = int(os.getenv("GEMINI_EMBEDDING_DIM", "384"))

    # Chunking — smaller = less context for the LLM to chew through
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "3"))

    # Storage (resolved relative to project root)
    UPLOAD_DIR: Path = PROJECT_ROOT / os.getenv("UPLOAD_DIR", "data/uploads")
    INDEX_DIR: Path = PROJECT_ROOT / os.getenv("INDEX_DIR", "data/index")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    # Auto-reload on file changes - convenient for local dev, wasteful (extra
    # watcher process + memory) in production. Off unless explicitly enabled.
    RELOAD: bool = os.getenv("RELOAD", "false").lower() in ("1", "true", "yes")

    # CORS - comma-separated list of allowed origins, e.g.
    # "https://your-app.vercel.app,https://your-app-*.vercel.app"
    # Defaults to "*" (allow all) for easy local/dev use.
    CORS_ORIGINS: list = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
    ]

    # Conversation memory
    CONVERSATION_TTL: int = int(os.getenv("CONVERSATION_TTL", "3600"))
    CONVERSATION_MAX_HISTORY: int = int(os.getenv("CONVERSATION_MAX_HISTORY", "10"))

    # Query rewriting
    QUERY_REWRITE_ENABLED: bool = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() in ("1", "true", "yes")

    # Hybrid search
    HYBRID_SEARCH_ENABLED: bool = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() in ("1", "true", "yes")
    HYBRID_SEARCH_ALPHA: float = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.5"))

    # Chunking strategy: "recursive" (recommended) or "sliding" (legacy)
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "recursive").lower()

    # Re-ranking: "none" (default) or "cross-encoder"
    RERANKER: str = os.getenv("RERANKER", "none").lower()

    def ensure_dirs(self):
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
