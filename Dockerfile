# Backend image for self-hosting (e.g. Oracle Cloud Always Free VM).
# Uses the full requirements.txt (local sentence-transformers embeddings) --
# unlike the Render deploy, we're not fighting a 512MB RAM ceiling here.
FROM python:3.11-slim

WORKDIR /app

# System deps needed to build a couple of the Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time so the first real request
# isn't slowed down (and so it works even if outbound internet is flaky
# right after a VM reboot). Done BEFORE copying app code so this layer
# stays cached across rebuilds that only change backend/ - otherwise every
# code change would force a slow model re-download too.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY backend/ ./backend/
COPY run.py .

RUN mkdir -p data/uploads data/index

EXPOSE 8000

# No --reload here: this is production. RELOAD env var still works if you
# ever want to override it (see backend/config.py).
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
