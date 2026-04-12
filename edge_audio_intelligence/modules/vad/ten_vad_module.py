"""
TEN VAD module: High-precision, low-latency streaming VAD.

Uses the prebuilt C library via ctypes (no Python ML framework needed).
Processes audio frame-by-frame (256 samples = 16ms at 16kHz).

Key properties:
- RTF: 0.005-0.015 (50-150x faster than real-time)
- Library size: 306-727 KB depending on platform
- Maintains internal RNN state between frames (do NOT reinitialize per segment)
- Input MUST be int16 PCM, exactly hop_size samples per call

Reference: TEN Team, "TEN VAD: Low-Latency, Lightweight and
High-Performance Streaming Voice Activity Detector," 2025.
"""

import logging
import os
import sys
from typing import Any, Dict

import numpy as np

from .base import BaseVAD

logger = logging.getLogger(__name__)

# Path to the copied TEN VAD library and wrapper
_TEN_VAD_LIB_DIR = os.path.join(os.path.dirname(__file__), "ten_vad_lib")


class TENVadModule(BaseVAD):
    """TEN VAD wrapper implementing BaseVAD interface.

    Args:
        hop_size: Frame size in samples. 256 (16ms) or 160 (10ms).
        speech_threshold: Threshold for speech decision (default 0.5, optimized).
        noise_threshold: Threshold for noise decision (default 0.3).
        raw_threshold: Internal TEN VAD threshold (default 0.5).
    """

    def __init__(
        self,
        hop_size: int = 256,
        speech_threshold: float = 0.5,
        noise_threshold: float = 0.3,
        raw_threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        super().__init__(
            name="TEN-VAD",
            speech_threshold=speech_threshold,
            noise_threshold=noise_threshold,
            sample_rate=sample_rate,
        )
        self.hop_size = hop_size
        self.raw_threshold = raw_threshold
        self._vad = None
        self._init_vad()

    def _init_vad(self):
        """Initialize the TEN VAD C library wrapper."""
        # Add the lib directory to path so ten_vad.py can find the .so/.framework
        if _TEN_VAD_LIB_DIR not in sys.path:
            sys.path.insert(0, _TEN_VAD_LIB_DIR)

        try:
            from ten_vad import TenVad
            self._vad = TenVad(self.hop_size, self.raw_threshold)
            logger.info(
                f"TEN VAD initialized: hop_size={self.hop_size}, "
                f"threshold={self.raw_threshold}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize TEN VAD: {e}")
            raise

    def _detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio frame-by-frame and return per-frame probabilities.

        TEN VAD requires int16 input, exactly hop_size samples per call.
        Internal state is maintained between frames automatically.
        """
        if self._vad is None:
            raise RuntimeError("TEN VAD not initialized")

        # Convert float32 [-1, 1] to int16
        audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

        n_frames = len(audio_int16) // self.hop_size
        probs = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            frame = audio_int16[i * self.hop_size : (i + 1) * self.hop_size]
            prob, _ = self._vad.process(frame)
            probs[i] = prob

        return probs

    def get_frame_duration_ms(self) -> float:
        return (self.hop_size / self.sample_rate) * 1000.0

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hop_size": self.hop_size,
            "raw_threshold": self.raw_threshold,
            "frame_duration_ms": self.get_frame_duration_ms(),
        })
        return config

    def count_parameters(self) -> int:
        # TEN VAD ONNX model has ~5K parameters (RNN with 64-dim hidden)
        return 5000

    def estimate_macs(self, data: Dict[str, Any]) -> int:
        """Estimate MACs for TEN VAD inference.

        Per frame: ~5,000-20,000 MACs (RNN + mel filterbank)
        """
        sr = data.get("sample_rate", 16000)
        n_samples = 0
        for key in ["enhanced_audio", "beamformed_audio", "multichannel_audio"]:
            if key in data:
                audio = data[key]
                n_samples = audio.shape[-1]
                break

        n_frames = n_samples // self.hop_size
        macs_per_frame = 15000  # Approximate: mel(5K) + RNN(10K)
        return n_frames * macs_per_frame

    def __del__(self):
        """Clean up TEN VAD instance."""
        if self._vad is not None:
            del self._vad
            self._vad = None
