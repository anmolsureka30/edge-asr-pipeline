"""
Run history persistence: save/load run results to JSON on disk.

Each run stores: pipeline config, scene config, metrics, remark,
and paths to saved figures and audio files.
"""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Default paths
_BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = _BASE_DIR / "results"
HISTORY_FILE = RESULTS_DIR / "run_history.json"
FIGURES_DIR = RESULTS_DIR / "figures"
AUDIO_DIR = RESULTS_DIR / "audio"
SAVED_SETUPS_FILE = RESULTS_DIR / "saved_setups.json"


class SavedSetup:
    """A saved scene + pipeline configuration for quick reloading."""

    def __init__(
        self,
        name: str = "Default",
        room_dim: List[float] = None,
        rt60: float = 0.3,
        snr_db: float = 15.0,
        duration_s: float = 3.0,
        noise_type: str = "white",
        sources: List[Dict[str, Any]] = None,
        mic_array_type: str = "linear_4",
        pipeline_config: Dict[str, str] = None,
        is_default: bool = False,
    ):
        self.name = name
        self.room_dim = room_dim or [6.0, 5.0, 3.0]
        self.rt60 = rt60
        self.snr_db = snr_db
        self.duration_s = duration_s
        self.noise_type = noise_type
        self.sources = sources or []
        self.mic_array_type = mic_array_type
        self.pipeline_config = pipeline_config or {}
        self.is_default = is_default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "room_dim": self.room_dim,
            "rt60": self.rt60,
            "snr_db": self.snr_db,
            "duration_s": self.duration_s,
            "noise_type": self.noise_type,
            "sources": self.sources,
            "mic_array_type": self.mic_array_type,
            "pipeline_config": self.pipeline_config,
            "is_default": self.is_default,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SavedSetup":
        return cls(**d)


class SetupManager:
    """Manages saved scene+pipeline setups on disk."""

    def __init__(self, path: str = None):
        self.path = Path(path) if path else SAVED_SETUPS_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.setups: List[SavedSetup] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.setups = [SavedSetup.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self.setups = []
        else:
            self.setups = []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump([s.to_dict() for s in self.setups], f, indent=2, default=str)

    def save_setup(self, setup: SavedSetup) -> None:
        """Save a setup. If is_default=True, unset previous default."""
        if setup.is_default:
            for s in self.setups:
                s.is_default = False

        # Replace if same name exists
        for i, s in enumerate(self.setups):
            if s.name == setup.name:
                self.setups[i] = setup
                self._save()
                return

        self.setups.append(setup)
        self._save()

    def set_as_default(self, name: str) -> bool:
        """Mark a setup as the default."""
        found = False
        for s in self.setups:
            if s.name == name:
                s.is_default = True
                found = True
            else:
                s.is_default = False
        if found:
            self._save()
        return found

    def get_default(self) -> Optional[SavedSetup]:
        """Get the default setup, or None."""
        for s in self.setups:
            if s.is_default:
                return s
        return None

    def get_setup(self, name: str) -> Optional[SavedSetup]:
        for s in self.setups:
            if s.name == name:
                return s
        return None

    def delete_setup(self, name: str) -> bool:
        for i, s in enumerate(self.setups):
            if s.name == name:
                self.setups.pop(i)
                self._save()
                return True
        return False

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.setups]


class RunRecord:
    """A single pipeline run record."""

    def __init__(
        self,
        run_id: str = "",
        timestamp: str = "",
        scene_config: Dict[str, Any] = None,
        pipeline_config: Dict[str, str] = None,
        sources: List[Dict[str, Any]] = None,
        metrics: Dict[str, float] = None,
        remark: str = "",
        figures_dir: str = "",
        audio_dir: str = "",
    ):
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.timestamp = timestamp or datetime.now().isoformat(timespec="seconds")
        self.scene_config = scene_config or {}
        self.pipeline_config = pipeline_config or {}
        self.sources = sources or []
        self.metrics = metrics or {}
        self.remark = remark
        self.figures_dir = figures_dir
        self.audio_dir = audio_dir

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "scene_config": self.scene_config,
            "pipeline_config": self.pipeline_config,
            "sources": self.sources,
            "metrics": self.metrics,
            "remark": self.remark,
            "figures_dir": self.figures_dir,
            "audio_dir": self.audio_dir,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        return cls(**d)

    @property
    def pipeline_summary(self) -> str:
        pc = self.pipeline_config
        parts = []
        for key in ["ssl", "beamforming", "enhancement", "asr"]:
            if key in pc and pc[key]:
                parts.append(pc[key])
        return " → ".join(parts) if parts else "No pipeline"

    @property
    def scene_summary(self) -> str:
        sc = self.scene_config
        snr = sc.get("snr_db", "?")
        rt60 = sc.get("rt60", "?")
        return f"SNR={snr}dB RT60={rt60}s"


class RunHistory:
    """Persistent run history manager.

    Loads/saves run records to a JSON file on disk.
    Also manages per-run figures and audio directories.
    """

    def __init__(self, history_file: str = None):
        self.history_file = Path(history_file) if history_file else HISTORY_FILE
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.runs: List[RunRecord] = []
        self._load()

    def _load(self):
        """Load run history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self.runs = [RunRecord.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.runs)} runs from {self.history_file}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load history: {e}. Starting fresh.")
                self.runs = []
        else:
            self.runs = []

    def _save(self):
        """Save run history to disk."""
        with open(self.history_file, "w") as f:
            json.dump([r.to_dict() for r in self.runs], f, indent=2, default=str)

    def add_run(self, record: RunRecord) -> str:
        """Add a new run record and save.

        Returns:
            The run_id.
        """
        # Create figure/audio directories
        fig_dir = FIGURES_DIR / record.run_id
        audio_dir = AUDIO_DIR / record.run_id
        fig_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        record.figures_dir = str(fig_dir)
        record.audio_dir = str(audio_dir)

        self.runs.insert(0, record)  # Most recent first
        self._save()
        logger.info(f"Saved run {record.run_id}")
        return record.run_id

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and its associated files."""
        for i, r in enumerate(self.runs):
            if r.run_id == run_id:
                # Delete figure/audio dirs
                if r.figures_dir and Path(r.figures_dir).exists():
                    shutil.rmtree(r.figures_dir, ignore_errors=True)
                if r.audio_dir and Path(r.audio_dir).exists():
                    shutil.rmtree(r.audio_dir, ignore_errors=True)
                self.runs.pop(i)
                self._save()
                logger.info(f"Deleted run {run_id}")
                return True
        return False

    def update_remark(self, run_id: str, remark: str) -> bool:
        """Update remark for a run."""
        for r in self.runs:
            if r.run_id == run_id:
                r.remark = remark
                self._save()
                return True
        return False

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Get a specific run by ID."""
        for r in self.runs:
            if r.run_id == run_id:
                return r
        return None

    @property
    def run_count(self) -> int:
        return len(self.runs)

    def next_run_number(self) -> int:
        return self.run_count + 1


def save_audio_signal(
    audio: np.ndarray,
    sr: int,
    run_id: str,
    name: str,
) -> str:
    """Save an audio signal as WAV for a given run.

    Returns the file path.
    """
    audio_dir = AUDIO_DIR / run_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / f"{name}.wav"

    # Normalize to prevent clipping
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    sf.write(str(path), audio.astype(np.float32), sr)
    return str(path)


def encode_audio_data_uri(audio: np.ndarray, sr: int) -> str:
    """Encode audio array as a base64 data URI for HTML5 audio playback."""
    import base64
    import io
    import struct

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    audio_int16 = (audio * 32767).astype(np.int16)

    # Write WAV to memory buffer
    buf = io.BytesIO()
    n_samples = len(audio_int16)
    n_channels = 1
    sample_width = 2  # 16-bit
    byte_rate = sr * n_channels * sample_width
    block_align = n_channels * sample_width
    data_size = n_samples * block_align

    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", n_channels))
    buf.write(struct.pack("<I", sr))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", 16))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_int16.tobytes())

    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"
