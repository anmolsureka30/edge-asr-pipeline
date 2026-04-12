"""Pydantic request models — mirror the Python dataclasses in testbench/scene.py."""

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceRequest(BaseModel):
    """Mirrors SourceConfig."""
    position: List[float] = Field(..., min_length=3, max_length=3)
    signal_type: str = "speech"
    audio_path: Optional[str] = None
    frequency: float = 440.0
    amplitude: float = 1.0
    label: str = "S0"
    transcription: Optional[str] = None
    onset_s: float = 0.0
    offset_s: float = -1.0


class MicArrayRequest(BaseModel):
    """Mirrors MicArrayConfig."""
    array_type: str = "linear"
    n_mics: int = 4
    spacing: float = 0.015
    center: List[float] = Field(default=[3.0, 3.0])
    height: float = 1.2


class SceneRequest(BaseModel):
    """Mirrors SceneConfig."""
    room_dim: List[float] = Field(default=[6.0, 5.0, 3.0], min_length=3, max_length=3)
    rt60: float = 0.3
    snr_db: float = 15.0
    sources: List[SourceRequest] = Field(default_factory=list)
    mic_array: MicArrayRequest = Field(default_factory=MicArrayRequest)
    duration_s: float = 3.0
    fs: int = 16000
    noise_type: str = "white"
    seed: int = 42


class PipelineRequest(BaseModel):
    """Pipeline configuration from the frontend."""
    scene_id: str
    ssl: str = "gcc_phat"
    bf: str = "delay_and_sum"
    enh: str = "spectral_subtraction"
    asr: str = "none"
    vad: str = "none"
    enh_gate: bool = False


class SetupSaveRequest(BaseModel):
    """Save a scene + pipeline setup."""
    name: str
    scene: SceneRequest
    pipeline: PipelineRequest
    is_default: bool = False


class RemarkRequest(BaseModel):
    """Update remark on a run."""
    remark: str
