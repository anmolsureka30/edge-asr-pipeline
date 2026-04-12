"""Backend configuration."""

from pathlib import Path

# Project root: edge_audio_intelligence/
PACKAGE_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Data paths
LIBRISPEECH_DIR = PROJECT_ROOT / "data" / "librispeech" / "LibriSpeech" / "test-clean"
RESULTS_DIR = PROJECT_ROOT / "results"

# Server
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]

# Run store
MAX_STORED_RUNS = 10
