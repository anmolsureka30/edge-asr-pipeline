"""FastAPI backend for the Edge Audio Intelligence Lab.

Serves the existing Python pipeline as a REST + WebSocket API.
The React frontend at localhost:5173 consumes this API.

Usage:
    cd edge_audio_intelligence
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, HOST, PORT

# Ensure the project root is on sys.path for imports
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Edge Audio Intelligence API",
    description="REST + WebSocket API for the acoustic simulation lab pipeline.",
    version="1.0.0",
)

# CORS — allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from .routers import scene, pipeline, audio, librispeech, history

app.include_router(scene.router, prefix="/api/scene", tags=["Scene"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(audio.router, prefix="/api/audio", tags=["Audio"])
app.include_router(librispeech.router, prefix="/api/librispeech", tags=["LibriSpeech"])
app.include_router(history.router, prefix="/api/history", tags=["History"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
