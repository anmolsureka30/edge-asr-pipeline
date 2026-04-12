"""Audio streaming endpoint — serves WAV files from RunStore."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..store import run_store, numpy_to_wav_bytes

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{run_id}/{signal}")
def stream_audio(run_id: str, signal: str):
    """Stream an audio signal as a WAV file.

    Signals: mic_0..mic_N, beamformed, enhanced, clean_0..clean_N
    """
    run_data = run_store.get_run(run_id)
    if run_data is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    audio = run_data.get_audio(signal)
    if audio is None:
        raise HTTPException(
            status_code=404,
            detail=f"Signal '{signal}' not found. Available: {run_data.list_audio_signals()}",
        )

    sr = run_data.scene.sample_rate
    wav_bytes = numpy_to_wav_bytes(audio, sr)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{run_id}_{signal}.wav"',
            "Accept-Ranges": "bytes",
        },
    )


@router.get("/{run_id}")
def list_signals(run_id: str):
    """List available audio signals for a run."""
    run_data = run_store.get_run(run_id)
    if run_data is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return {"signals": run_data.list_audio_signals()}
