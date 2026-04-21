"""Pydantic response models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SceneResponse(BaseModel):
    """Returned after scene generation."""
    scene_id: str
    n_sources: int
    true_doas: List[float]
    transcriptions: List[str]
    mic_positions: List[List[float]]
    duration_s: float
    sample_rate: int


class StageProgress(BaseModel):
    """WebSocket progress message."""
    stage: str
    progress: float
    message: str = ""
    latency_ms: float = 0.0
    metrics: Dict[str, Any] = {}


class PipelineRunResponse(BaseModel):
    """Returned immediately when a pipeline run is started."""
    task_id: str
    status: str = "started"


class StageResult(BaseModel):
    """Results from a single pipeline stage."""
    stage: str
    method: str
    latency_ms: float
    metrics: Dict[str, float]
    plot: Optional[Dict[str, Any]] = None  # Plotly figure JSON


class PipelineResultResponse(BaseModel):
    """Full pipeline result."""
    run_id: str
    scene_id: str
    stages: List[StageResult]
    audio_signals: List[str]  # Available signal keys
    transcriptions: List[str]
    metrics: Dict[str, Any]
    total_latency_ms: float
    total_rtf: float
    speaker_segments: List[List] = []  # [(start, end, speaker_id), ...]
    diarization_method: str = ""


class RunHistoryItem(BaseModel):
    """Summary of a run for the history list."""
    run_id: str
    timestamp: str
    pipeline_summary: str
    scene_summary: str
    key_metrics: Dict[str, Any]
    remark: str = ""


class LibriSpeechUtterance(BaseModel):
    """A single LibriSpeech utterance."""
    id: str
    text: str
    speaker: str
    path: str
