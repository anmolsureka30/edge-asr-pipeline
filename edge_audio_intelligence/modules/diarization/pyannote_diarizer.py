# edge_audio_intelligence/modules/diarization/pyannote_diarizer.py
"""
Pyannote Speaker Diarization module.

Wraps pyannote.audio Pipeline for "who spoke when" attribution.
Supports pyannote/speaker-diarization-3.0 and 3.1.

Key properties:
- Uses pre-trained segmentation + embedding + clustering
- Requires HF_TOKEN (set via .env or environment variable)
- Global model cache avoids re-downloading on each pipeline run (~700MB models)

Reference: Bredin et al., "pyannote.audio 2.1: speaker diarization
pipeline," Interspeech 2023.
"""

import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# Force weights_only=False to allow Pyannote 3.1 to load on PyTorch 2.6+
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

from pyannote.audio import Pipeline

from .base import BaseDiarizer

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading on each pipeline run
_cached_pipelines: Dict[str, Pipeline] = {}


class PyannoteDiarizer(BaseDiarizer):
    """Pyannote-based speaker diarization implementing BaseDiarizer.

    Args:
        name: Module name for logging.
        model_name: HuggingFace model identifier.
        use_auth_token: HF token (defaults to HF_TOKEN env var).
        min_duration_s: Minimum segment duration to keep (seconds).
    """

    def __init__(
        self,
        name: str = "pyannote_diarizer",
        model_name: str = "pyannote/speaker-diarization-3.1",
        use_auth_token: str = None,
        min_duration_s: float = 0.3,
        sample_rate: int = 16000,
    ):
        super().__init__(name=name, sample_rate=sample_rate)
        self.model_name = model_name
        self.min_duration_s = min_duration_s
        self._pipeline = None

        token = use_auth_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "Hugging Face token is required. "
                "Set the 'HF_TOKEN' environment variable or pass use_auth_token."
            )

        self._load_pipeline(model_name, token)

    def _load_pipeline(self, model_name: str, token: str):
        """Load the pyannote pipeline with global caching."""
        global _cached_pipelines

        if model_name in _cached_pipelines:
            self._pipeline = _cached_pipelines[model_name]
            logger.info(f"[{self.name}] Using cached pipeline: {model_name}")
            return

        logger.info(f"[{self.name}] Loading Pyannote pipeline: {model_name}...")
        t0 = time.perf_counter()

        try:
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        except Exception as e:
            logger.error(f"[{self.name}] Failed loading model from hub: {e}")
            raise

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            logger.info(f"[{self.name}] Pipeline moved to CUDA")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon acceleration
            pipeline.to(torch.device("mps"))
            logger.info(f"[{self.name}] Pipeline moved to MPS (Apple Silicon)")

        load_time = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[{self.name}] Pipeline loaded in {load_time:.0f}ms"
        )

        _cached_pipelines[model_name] = pipeline
        self._pipeline = pipeline

    def get_config(self) -> Dict[str, Any]:
        """Return module configuration for the pipeline logger."""
        return {
            "name": self.name,
            "model_name": self.model_name,
            "min_duration_s": self.min_duration_s,
            "sample_rate": self.sample_rate,
        }

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float, str]]:
        """Run pyannote diarization on audio.

        Args:
            audio: Mono audio signal [n_samples], float32.
            sample_rate: Sample rate.

        Returns:
            List of (start_time, end_time, speaker_id) tuples.
        """
        if self._pipeline is None:
            raise RuntimeError(f"[{self.name}] Pipeline not loaded")

        # Minimum audio length check (pyannote needs at least ~0.5s)
        min_samples = int(0.5 * sample_rate)
        if len(audio) < min_samples:
            logger.warning(
                f"[{self.name}] Audio too short ({len(audio)} samples, "
                f"need {min_samples}). Returning empty segments."
            )
            return []

        # Prepare input tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()

        # Run the diarization pipeline
        output = self._pipeline({
            "waveform": audio_tensor,
            "sample_rate": sample_rate,
        })

        # Handle pyannote API changes (v3.x vs v4.x+)
        if hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization
        else:
            annotation = output

        # Extract segments
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if duration >= self.min_duration_s:
                segments.append((turn.start, turn.end, speaker))

        logger.info(
            f"[{self.name}] Found {len(segments)} segments, "
            f"{len(set(s[2] for s in segments))} speakers"
        )

        return segments

    def count_parameters(self) -> int:
        """Estimate total parameters (segmentation + embedding models)."""
        # Pyannote 3.1: PyanNet segmentation (~1M) + WeSpeaker embeddings (~6M)
        return 7_000_000

    def estimate_macs(self, data: Dict[str, Any]) -> int:
        """Rough MAC estimate for pyannote inference."""
        sr = data.get("sample_rate", 16000)
        n_samples = 0
        for key in ["enhanced_audio", "beamformed_audio", "multichannel_audio"]:
            if key in data:
                n_samples = data[key].shape[-1]
                break
        # ~100M MACs per second of audio (segmentation + embedding + clustering)
        return int(n_samples / sr * 100_000_000)