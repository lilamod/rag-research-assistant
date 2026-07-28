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
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

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
    GEMINI_EMBEDDING_MODEL: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
    )
    GEMINI_EMBEDDING_DIM: int = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))

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

    def ensure_dirs(self):
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
