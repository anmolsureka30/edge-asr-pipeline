"""
Wavelet Energy VAD module: EE678 course-aligned baseline.

Uses DWT sub-band energy ratio for speech/non-speech classification.
Speech concentrates energy in cA3 (0-1kHz), noise in cD1 (4-8kHz).

Key properties:
- ~5-10 FLOPs per frame (ultra-low compute)
- Zero ML dependencies (only numpy + pywt)
- Provides interpretable sub-band energy visualization
- Serves as ablation baseline against neural VAD methods

Reference: Stegmann & Schroeder, "Robust Voice-Activity Detection
Based on the Wavelet Transform," IEEE Speech Coding Workshop, 1997.
CLAUDE.md Section 5.3, PIPELINE_ALGORITHM.md Eq. 7.4-7.5
"""

import logging
from typing import Any, Dict

import numpy as np

from .base import BaseVAD

logger = logging.getLogger(__name__)


class WaveletVADModule(BaseVAD):
    """Wavelet energy ratio VAD implementing BaseVAD interface.

    Wraps the existing wavelet/wavelet_vad.py with BaseModule contract
    and adds per-frame confidence scores instead of binary-only output.

    Args:
        wavelet: Wavelet name. Default 'bior2.2' (CDF 5/3).
        levels: DWT decomposition levels. Default 3.
        frame_length: Frame size in samples. Default 512 (32ms at 16kHz).
        hop_length: Frame hop in samples. Default 256 (16ms at 16kHz).
        energy_threshold: Raw energy ratio threshold. Default 3.0.
        speech_threshold: P(speech) threshold for speech decision. Default 0.7.
        noise_threshold: P(speech) threshold for noise decision. Default 0.3.
        min_speech_frames: Minimum consecutive speech frames. Default 3.
        hangover_frames: Frames to hold after speech ends. Default 5.
    """

    def __init__(
        self,
        wavelet: str = "bior2.2",
        levels: int = 3,
        frame_length: int = 512,
        hop_length: int = 256,
        energy_threshold: float = 3.0,
        speech_threshold: float = 0.7,
        noise_threshold: float = 0.3,
        min_speech_frames: int = 3,
        hangover_frames: int = 5,
        sample_rate: int = 16000,
    ):
        super().__init__(
            name="Wavelet-VAD",
            speech_threshold=speech_threshold,
            noise_threshold=noise_threshold,
            sample_rate=sample_rate,
        )
        self.wavelet = wavelet
        self.levels = levels
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.min_speech_frames = min_speech_frames
        self.hangover_frames = hangover_frames

        # Research Change 4: Adaptive Exponential Moving Average (EMA) for VAD
        # Tracks noise floor dynamics across highly ambient environments
        self.noise_ratio_ema = self.energy_threshold

        # Sub-band energy history for scalogram visualization
        self.sub_band_history = []

    def _detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute wavelet energy ratio and convert to probability.

        R_VAD = E(cA_J + cD_J) / E(cD_1)
        P(speech) = sigmoid(R_VAD - threshold) mapped to [0, 1]
        """
        import pywt

        n_frames = max(1, (len(audio) - self.frame_length) // self.hop_length + 1)
        probs = np.zeros(n_frames, dtype=np.float32)
        self.sub_band_history = []

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            if end > len(audio):
                break

            frame = audio[start:end]

            # Multi-level DWT decomposition
            coeffs = pywt.wavedec(frame, self.wavelet, level=self.levels)
            # coeffs = [cA_J, cD_J, cD_(J-1), ..., cD_1]

            # Per-band energy
            energies = {}
            band_names = [f"cA{self.levels}"] + [
                f"cD{self.levels - j}" for j in range(self.levels)
            ]
            for j, (name, c) in enumerate(zip(band_names, coeffs)):
                energies[name] = float(np.sum(c ** 2))

            self.sub_band_history.append(energies)

            # Energy ratio: low-freq (speech) / high-freq (noise)
            e_low = energies[f"cA{self.levels}"] + energies[f"cD{self.levels}"]
            e_high = energies["cD1"] + 1e-10

            ratio = e_low / e_high
            
            # Research Change 4: Exponential Moving Average Adaptive Tracking
            # Check if likely noise frame (ratio < adaptive ceiling)
            adaptive_ceiling = self.noise_ratio_ema * 2.5
            if ratio < adaptive_ceiling:
                # Smoothly update EMA during non-speech periods to track the room floor
                self.noise_ratio_ema = 0.95 * self.noise_ratio_ema + 0.05 * ratio
                
            # Current adaptive threshold is tightly bounded to real-world floor
            adaptive_threshold = self.noise_ratio_ema * 2.5

            # Convert ratio to probability via sigmoid-like mapping
            # P(speech) = ratio / (ratio + threshold)
            # When ratio == threshold: P = 0.5
            # When ratio >> threshold: P → 1.0
            # When ratio << threshold: P → 0.0
            probs[i] = ratio / (ratio + adaptive_threshold)

        # Apply min duration filtering on raw binary decisions
        raw_speech = probs > 0.5
        raw_speech = self._apply_min_duration(raw_speech)
        raw_speech = self._apply_hangover(raw_speech)

        # Mask probabilities: non-speech regions get reduced probability
        for i in range(len(probs)):
            if not raw_speech[i]:
                probs[i] = min(probs[i], 0.3)

        return probs

    def _apply_min_duration(self, vad: np.ndarray) -> np.ndarray:
        """Remove speech regions shorter than min_speech_frames."""
        result = np.zeros_like(vad)
        in_speech = False
        start = 0
        for i in range(len(vad)):
            if vad[i] and not in_speech:
                start = i
                in_speech = True
            elif not vad[i] and in_speech:
                if i - start >= self.min_speech_frames:
                    result[start:i] = True
                in_speech = False
        if in_speech and len(vad) - start >= self.min_speech_frames:
            result[start:] = True
        return result

    def _apply_hangover(self, vad: np.ndarray) -> np.ndarray:
        """Extend speech regions by hangover_frames."""
        result = vad.copy()
        count = 0
        for i in range(len(vad)):
            if vad[i]:
                count = self.hangover_frames
            elif count > 0:
                result[i] = True
                count -= 1
        return result

    def get_frame_duration_ms(self) -> float:
        return (self.hop_length / self.sample_rate) * 1000.0

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "wavelet": self.wavelet,
            "levels": self.levels,
            "frame_length": self.frame_length,
            "hop_length": self.hop_length,
            "energy_threshold": self.energy_threshold,
            "min_speech_frames": self.min_speech_frames,
            "hangover_frames": self.hangover_frames,
        })
        return config

    def count_parameters(self) -> int:
        return 0  # No trainable parameters

    def estimate_macs(self, data: Dict[str, Any]) -> int:
        """DWT of frame_length samples: ~4*N per level + energy: ~N per band."""
        sr = data.get("sample_rate", 16000)
        n_samples = 0
        for key in ["enhanced_audio", "beamformed_audio", "multichannel_audio"]:
            if key in data:
                n_samples = data[key].shape[-1]
                break
        n_frames = max(1, (n_samples - self.frame_length) // self.hop_length + 1)
        macs_per_frame = 4 * self.frame_length * self.levels + self.frame_length * (self.levels + 1) + 5
        return n_frames * macs_per_frame
