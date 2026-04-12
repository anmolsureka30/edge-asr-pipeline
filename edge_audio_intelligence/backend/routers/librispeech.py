"""LibriSpeech utterance index endpoint."""

import logging
from typing import List

from fastapi import APIRouter

from ..config import LIBRISPEECH_DIR
from ..models.responses import LibriSpeechUtterance

logger = logging.getLogger(__name__)
router = APIRouter()

# Cached index (built once, reused)
_index: List[dict] = None


def _build_index() -> List[dict]:
    global _index
    if _index is not None:
        return _index

    _index = []
    if not LIBRISPEECH_DIR.exists():
        logger.warning(f"LibriSpeech not found at {LIBRISPEECH_DIR}")
        return _index

    for trans_file in sorted(LIBRISPEECH_DIR.rglob("*.trans.txt")):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    flac = trans_file.parent / f"{utt_id}.flac"
                    if flac.exists():
                        speaker_id = utt_id.split("-")[0]
                        _index.append({
                            "id": utt_id,
                            "path": str(flac),
                            "text": text,
                            "speaker": speaker_id,
                        })

    logger.info(f"LibriSpeech index: {len(_index)} utterances")
    return _index


@router.get("/index", response_model=List[LibriSpeechUtterance])
def get_index(limit: int = 500):
    """Get the LibriSpeech utterance index (cached, lazy-built)."""
    index = _build_index()
    return index[:limit]
