"""Pipeline execution endpoint with WebSocket progress."""

import asyncio
import json
import logging
import threading
import uuid
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..models.requests import PipelineRequest
from ..models.responses import PipelineRunResponse
from ..services.pipeline_service import run_pipeline
from ..services.plot_service import generate_all_plots
from ..store import run_store


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

logger = logging.getLogger(__name__)
router = APIRouter()

# In-flight tasks: task_id -> {status, progress_messages[], result}
_tasks: Dict[str, dict] = {}
_tasks_lock = threading.Lock()


def _execute_pipeline(task_id: str, scene, req: PipelineRequest):
    """Run pipeline in a background thread. Stores progress messages."""
    progress_messages = []

    def on_progress(stage, pct, msg, latency_ms, metrics):
        entry = _sanitize({
            "stage": stage, "progress": pct, "message": msg,
            "latency_ms": latency_ms, "metrics": metrics,
        })
        with _tasks_lock:
            _tasks[task_id]["progress"].append(entry)

    try:
        result = run_pipeline(
            scene=scene,
            ssl_val=req.ssl,
            bf_val=req.bf,
            enh_val=req.enh,
            asr_val=req.asr,
            vad_val=req.vad,
            diarization_val=req.diarization,
            enh_gate=req.enh_gate,
            on_progress=on_progress,
        )

        # Generate plots
        plots = generate_all_plots(result["data"], scene)

        # Store in RunStore
        run_id = run_store.store_run(
            scene=scene,
            pipeline_data=result["data"],
            result=result["result"],
            plots=plots,
        )

        # Build response summary
        data = result["data"]
        stages = []
        for mr in result["result"].module_results:
            stages.append({
                "stage": mr.module_name,
                "method": mr.module_name,
                "latency_ms": mr.latency_ms,
                "metrics": mr.metrics,
            })

        run_data = run_store.get_run(run_id)
        audio_signals = run_data.list_audio_signals() if run_data else []

        final = {
            "run_id": run_id,
            "scene_id": req.scene_id,
            "stages": stages,
            "audio_signals": audio_signals,
            "transcriptions": data.get("transcriptions", []),
            "metrics": {k: v for mr in result["result"].module_results
                        for k, v in mr.metrics.items()},
            "total_latency_ms": result["result"].total_latency_ms,
            "total_rtf": result["result"].total_rtf,
            "plots": {k: v for k, v in plots.items() if v is not None},
            "module_names": result["module_names"],
            "vad_method": data.get("vad_method", ""),
            "enhancement_applied": data.get("enhancement_applied", True),
            "enhancement_gate_reason": data.get("enhancement_gate_reason", ""),
            "speaker_segments": [list(s) for s in data.get("speaker_segments", [])],
            "diarization_method": data.get("diarization_method", ""),
        }

        with _tasks_lock:
            _tasks[task_id]["status"] = "complete"
            _tasks[task_id]["result"] = _sanitize(final)
            _tasks[task_id]["progress"].append({
                "stage": "complete", "progress": 1.0,
                "message": "Pipeline complete", "latency_ms": 0, "metrics": {},
            })

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        with _tasks_lock:
            _tasks[task_id]["status"] = "error"
            _tasks[task_id]["error"] = str(e)
            _tasks[task_id]["progress"].append({
                "stage": "error", "progress": 0,
                "message": str(e), "latency_ms": 0, "metrics": {},
            })


@router.post("/run", response_model=PipelineRunResponse)
def start_run(req: PipelineRequest):
    """Start a pipeline run. Returns task_id immediately.

    Connect to WS /ws/pipeline/{task_id} for progress updates.
    """
    scene = run_store.get_scene(req.scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Scene {req.scene_id} not found")

    task_id = str(uuid.uuid4())[:8]

    with _tasks_lock:
        _tasks[task_id] = {
            "status": "running",
            "progress": [],
            "result": None,
            "error": None,
        }

    # Run in background thread (ML models may not be async-safe)
    thread = threading.Thread(
        target=_execute_pipeline, args=(task_id, scene, req), daemon=True,
    )
    thread.start()

    return PipelineRunResponse(task_id=task_id, status="started")


@router.get("/status/{task_id}")
def get_status(task_id: str):
    """Poll pipeline status (fallback if WebSocket disconnects)."""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "status": task["status"],
        "progress": task["progress"],
        "result": task["result"],
        "error": task["error"],
    }


@router.websocket("/ws/{task_id}")
async def pipeline_ws(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for streaming pipeline progress.

    Sends JSON progress messages as they become available.
    """
    await websocket.accept()

    last_idx = 0
    try:
        while True:
            with _tasks_lock:
                task = _tasks.get(task_id)

            if task is None:
                await websocket.send_json({"stage": "error", "message": "Task not found"})
                break

            # Send new progress messages
            progress = task["progress"]
            while last_idx < len(progress):
                msg = progress[last_idx]
                await websocket.send_json(msg)
                last_idx += 1

                if msg["stage"] in ("complete", "error"):
                    # Send final result
                    if task["result"]:
                        await websocket.send_json({
                            "stage": "result",
                            "data": task["result"],
                        })
                    await websocket.close()
                    return

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
