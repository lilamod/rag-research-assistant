"""
Convenience entrypoint: `python run.py`
Starts the FastAPI app with uvicorn using settings from .env.
"""
import uvicorn
from backend.config import settings

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
