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
    # LLM
    # "gemini"    -> Google's Gemini API for chat generation too, not just
    #                embeddings. Reuses the same GEMINI_API_KEY. Genuinely
    #                free tier (no card), good default if you want a single
    #                provider for everything.
    # "anthropic" -> Claude, needs a funded Anthropic account (no free
    #                API quota).
    # "openai"    -> needs a funded OpenAI account (no free API quota).
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Embeddings
    # "local"  -> sentence-transformers, runs on CPU, no API key, but pulls in
    #             torch (heavy: needs 1GB+ RAM, unsuitable for small hosts).
    # "gemini" -> Google's Gemini embeddings API, lightweight install, needs
    #             GEMINI_API_KEY. Genuinely free tier (1,500 requests/day),
    #             no card required - get a key at
    #             https://aistudio.google.com/apikey. Default recommendation.
    # "openai" -> OpenAI's embeddings API, lightweight install, needs
    #             OPENAI_API_KEY (independent of LLM_PROVIDER for generation).
    #             No free quota - needs a funded OpenAI account.
    # "voyage" -> Voyage AI's embeddings API, lightweight install, needs
    #             VOYAGE_API_KEY. Free tier (200M tokens) exists but is
    #             capped at 3 RPM until you add a (non-charging) card.
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    OPENAI_EMBEDDING_DIM: int = int(os.getenv("OPENAI_EMBEDDING_DIM", "1536"))

    VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
    VOYAGE_EMBEDDING_MODEL: str = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-lite")
    VOYAGE_EMBEDDING_DIM: int = int(os.getenv("VOYAGE_EMBEDDING_DIM", "1024"))

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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
