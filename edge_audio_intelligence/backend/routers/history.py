"""Run history endpoints — CRUD for past pipeline runs."""

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..store import run_store

logger = logging.getLogger(__name__)
router = APIRouter()


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    elif isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    elif isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@router.get("")
def list_history():
    """List all runs in history (most recent first)."""
    runs = run_store.list_runs()
    items = []
    for run_id in reversed(runs):
        run_data = run_store.get_run(run_id)
        if run_data and run_data.result:
            r = run_data.result
            items.append(_sanitize({
                "run_id": run_id,
                "snr_db": r.snr_db,
                "rt60": r.rt60,
                "n_sources": r.n_sources,
                "total_latency_ms": r.total_latency_ms,
                "total_rtf": r.total_rtf,
                "end_to_end_wer": r.end_to_end_wer,
                "module_results": [
                    {"name": mr.module_name, "metrics": mr.metrics, "latency_ms": mr.latency_ms}
                    for mr in r.module_results
                ],
            }))
    return items


@router.get("/{run_id}")
def get_run_details(run_id: str):
    """Get full details for a specific run."""
    run_data = run_store.get_run(run_id)
    if run_data is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    r = run_data.result
    return _sanitize({
        "run_id": run_id,
        "audio_signals": run_data.list_audio_signals(),
        "result": r.to_dict() if r else None,
        "transcriptions": run_data.pipeline_data.get("transcriptions", []),
        "vad_method": run_data.pipeline_data.get("vad_method", ""),
        "enhancement_applied": run_data.pipeline_data.get("enhancement_applied"),
        "enhancement_gate_reason": run_data.pipeline_data.get("enhancement_gate_reason", ""),
    })


@router.delete("/{run_id}")
def delete_run(run_id: str):
    """Delete a run from the store."""
    if run_store.delete_run(run_id):
        return {"status": "deleted", "run_id": run_id}
    raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
