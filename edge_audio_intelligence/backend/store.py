"""Thread-safe in-memory store for active pipeline runs.

Replaces the module-level globals (_current_pipeline_data, _current_scene)
from the Dash dashboard. Supports multiple concurrent runs and auto-evicts
old runs to prevent unbounded memory growth.
"""

import io
import struct
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np


class RunData:
    """All data from a single pipeline run."""

    def __init__(
        self,
        scene: Any,           # AcousticScene
        pipeline_data: Dict,  # Full pipeline output dict
        result: Any,          # PipelineResult from evaluator
        plots: Dict[str, Any] = None,  # stage -> Plotly figure dict
    ):
        self.scene = scene
        self.pipeline_data = pipeline_data
        self.result = result
        self.plots = plots or {}

    def get_audio(self, signal_key: str) -> Optional[np.ndarray]:
        """Retrieve an audio signal by key.

        Keys: mic_0..mic_N, beamformed, enhanced, clean_0..clean_N
        """
        if signal_key.startswith("mic_"):
            idx = int(signal_key.split("_")[1])
            if self.scene.multichannel_audio is not None:
                return self.scene.multichannel_audio[idx]

        elif signal_key == "beamformed":
            audio = self.pipeline_data.get("beamformed_audio")
            if audio is not None:
                return audio[0] if audio.ndim > 1 else audio

        elif signal_key == "enhanced":
            audio = self.pipeline_data.get("enhanced_audio")
            if audio is not None:
                return audio[0] if audio.ndim > 1 else audio

        elif signal_key.startswith("clean_"):
            idx = int(signal_key.split("_")[1])
            if idx < len(self.scene.clean_sources):
                return self.scene.clean_sources[idx]

        return None

    def list_audio_signals(self) -> List[str]:
        """List all available audio signal keys."""
        signals = []
        if self.scene.multichannel_audio is not None:
            for i in range(self.scene.multichannel_audio.shape[0]):
                signals.append(f"mic_{i}")
        if "beamformed_audio" in self.pipeline_data:
            signals.append("beamformed")
        if "enhanced_audio" in self.pipeline_data:
            signals.append("enhanced")
        for i in range(len(self.scene.clean_sources)):
            signals.append(f"clean_{i}")
        return signals


class RunStore:
    """Thread-safe in-memory store for pipeline run data."""

    def __init__(self, max_runs: int = 10):
        self._runs: OrderedDict[str, RunData] = OrderedDict()
        self._scenes: Dict[str, Any] = {}  # scene_id -> AcousticScene
        self._lock = threading.Lock()
        self._max_runs = max_runs

    # ── Scene management ──

    def store_scene(self, scene: Any) -> str:
        """Store a generated scene and return its ID."""
        scene_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._scenes[scene_id] = scene
            # Evict old scenes (keep last 20)
            while len(self._scenes) > 20:
                self._scenes.popitem(last=False)
        return scene_id

    def get_scene(self, scene_id: str) -> Optional[Any]:
        with self._lock:
            return self._scenes.get(scene_id)

    # ── Run management ──

    def store_run(
        self, scene: Any, pipeline_data: Dict, result: Any, plots: Dict = None,
    ) -> str:
        """Store a completed run and return its ID."""
        run_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._runs[run_id] = RunData(scene, pipeline_data, result, plots)
            while len(self._runs) > self._max_runs:
                self._runs.popitem(last=False)
        return run_id

    def get_run(self, run_id: str) -> Optional[RunData]:
        with self._lock:
            return self._runs.get(run_id)

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            if run_id in self._runs:
                del self._runs[run_id]
                return True
            return False

    def list_runs(self) -> List[str]:
        with self._lock:
            return list(self._runs.keys())


def numpy_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Convert a NumPy audio array to WAV bytes for streaming.

    Args:
        audio: 1D float audio signal, range [-1, 1].
        sr: Sample rate.

    Returns:
        WAV file as bytes.
    """
    if audio.ndim > 1:
        audio = audio[0]

    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.95 / peak)

    # Convert to int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    # Build WAV in memory
    buf = io.BytesIO()
    n_samples = len(audio_int16)
    data_size = n_samples * 2  # 16-bit = 2 bytes per sample

    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")

    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))        # chunk size
    buf.write(struct.pack("<H", 1))         # PCM format
    buf.write(struct.pack("<H", 1))         # mono
    buf.write(struct.pack("<I", sr))        # sample rate
    buf.write(struct.pack("<I", sr * 2))    # byte rate
    buf.write(struct.pack("<H", 2))         # block align
    buf.write(struct.pack("<H", 16))        # bits per sample

    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_int16.tobytes())

    return buf.getvalue()


# Global singleton
run_store = RunStore()
