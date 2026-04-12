"""
Wavelet Energy VAD: Ultra-low-power Voice Activity Detection.

KEY WAVELET INTEGRATION POINT (CLAUDE.md Section 5.3).

Uses DWT sub-band energy ratios for speech detection.
Speech has high energy in cA_3 (0-1kHz) and cD_3 (1-2kHz) relative
to cD_1 (4-8kHz). A simple threshold on E(cA_3)/E(cD_1) provides
effective VAD with minimal computation.

Target: <5mW power consumption for always-on operation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dwt_features import DWTFeatureExtractor

logger = logging.getLogger(__name__)


class WaveletVAD:
    """Wavelet energy-based Voice Activity Detector.

    Computes per-frame DWT energy ratio between low-frequency
    (speech-dominant) and high-frequency (noise-dominant) sub-bands.

    Args:
        wavelet: Wavelet name. Default 'bior2.2'.
        levels: DWT decomposition levels. Default 3.
        frame_length: Frame size in samples. Default 512 (32ms at 16kHz).
        hop_length: Frame hop in samples. Default 256 (16ms at 16kHz).
        threshold: Energy ratio threshold for speech detection. Default 3.0.
        min_speech_frames: Minimum consecutive speech frames. Default 3.
            Prevents single-frame false positives.
        hangover_frames: Frames to keep after speech ends. Default 5.
            Prevents premature cutoff.
    """

    def __init__(
        self,
        wavelet: str = "bior2.2",
        levels: int = 3,
        frame_length: int = 512,
        hop_length: int = 256,
        threshold: float = 3.0,
        min_speech_frames: int = 3,
        hangover_frames: int = 5,
    ):
        self.feature_extractor = DWTFeatureExtractor(
            wavelet=wavelet,
            levels=levels,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        self.threshold = threshold
        self.min_speech_frames = min_speech_frames
        self.hangover_frames = hangover_frames
        self.hop_length = hop_length
        self.frame_length = frame_length

    def detect(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> np.ndarray:
        """Detect speech frames using wavelet energy ratio.

        Ratio = E(cA_J) / (E(cD_1) + epsilon)

        High ratio => speech present (energy concentrated in low freq).
        Low ratio => noise only (energy spread across all bands).

        Args:
            audio: Audio signal [n_samples].
            sr: Sample rate.

        Returns:
            Binary mask [n_frames]: 1 = speech, 0 = non-speech.
        """
        # Extract per-frame DWT energies
        features = self.feature_extractor.frame_features(audio, sr)
        # features: [n_frames, n_bands]
        # bands: [E(cA_J), E(cD_J), ..., E(cD_1)]

        n_frames = features.shape[0]

        # Energy ratio: low-freq / high-freq
        e_low = features[:, 0]  # cA_J (lowest frequency band)
        e_high = features[:, -1]  # cD_1 (highest frequency band)

        # Also include cD_J (next lowest detail band) in speech energy
        if features.shape[1] > 2:
            e_low = e_low + features[:, 1]  # cA_J + cD_J

        ratio = e_low / (e_high + 1e-10)

        # Raw decision
        raw_vad = (ratio > self.threshold).astype(np.int32)

        # Apply minimum speech duration
        vad = self._apply_min_duration(raw_vad, self.min_speech_frames)

        # Apply hangover
        vad = self._apply_hangover(vad, self.hangover_frames)

        return vad

    def detect_segments(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> List[Tuple[float, float]]:
        """Detect speech segments as (start_time, end_time) pairs.

        Args:
            audio: Audio signal [n_samples].
            sr: Sample rate.

        Returns:
            List of (start_s, end_s) tuples.
        """
        vad = self.detect(audio, sr)
        segments = []
        in_speech = False
        start = 0

        for i, is_speech in enumerate(vad):
            if is_speech and not in_speech:
                start = i
                in_speech = True
            elif not is_speech and in_speech:
                start_s = start * self.hop_length / sr
                end_s = i * self.hop_length / sr
                segments.append((start_s, end_s))
                in_speech = False

        if in_speech:
            start_s = start * self.hop_length / sr
            end_s = len(audio) / sr
            segments.append((start_s, end_s))

        return segments

    def estimate_flops_per_frame(self) -> int:
        """Estimate FLOPs for one frame.

        DWT of frame_length samples: ~4*N per level
        Energy computation: ~N per band
        Ratio + threshold: ~3

        Total should be extremely low for edge deployment.
        """
        n = self.frame_length
        levels = self.feature_extractor.levels
        dwt_flops = 4 * n * levels
        energy_flops = n * (levels + 1)
        decision_flops = 5
        return dwt_flops + energy_flops + decision_flops

    def _apply_min_duration(
        self,
        vad: np.ndarray,
        min_frames: int,
    ) -> np.ndarray:
        """Remove speech regions shorter than min_frames."""
        result = np.zeros_like(vad)
        in_speech = False
        start = 0

        for i in range(len(vad)):
            if vad[i] and not in_speech:
                start = i
                in_speech = True
            elif not vad[i] and in_speech:
                if i - start >= min_frames:
                    result[start:i] = 1
                in_speech = False

        if in_speech and len(vad) - start >= min_frames:
            result[start:] = 1

        return result

    def _apply_hangover(
        self,
        vad: np.ndarray,
        hangover: int,
    ) -> np.ndarray:
        """Extend speech regions by hangover frames."""
        result = vad.copy()
        count = 0

        for i in range(len(vad)):
            if vad[i]:
                count = hangover
            elif count > 0:
                result[i] = 1
                count -= 1

        return result
