"""
GatedEnhancer: Conditionally skips enhancement based on VAD, SSL, and
tonal interference detection.

Solves two V1/V2 bottlenecks:
1. Enhancement DEGRADES WER in multi-speaker scenarios (25% → 39.3%)
   by treating interfering speech as noise.
2. Enhancement DEGRADES WER when tonal interference is present
   (spectral subtraction artifacts are worse than the original tone).

Decision matrix:
    tonal_interference=True      → SKIP (enhancement artifacts > tone)
    n_speakers=1 + speech=True   → APPLY enhancement (safe, single speaker)
    n_speakers=1 + speech=False  → SKIP (nothing to enhance)
    n_speakers>1 + speech=True   → SKIP (enhancer destroys target)
    n_speakers>1 + speech=False  → SKIP (nothing to enhance)
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from ..base import BaseModule

logger = logging.getLogger(__name__)


def detect_tonal_interference(
    audio: np.ndarray,
    sr: int,
    peak_threshold_db: float = 15.0,
    min_freq: float = 50.0,
    max_freq: float = 7500.0,
) -> List[Tuple[float, float]]:
    """Detect tonal interference in audio using spectral peak detection.

    Returns list of (frequency_hz, prominence_db) for detected tones.
    A tone is a spectral peak that exceeds the local median floor by
    > peak_threshold_db. This detects sine waves, hums, and machine tones.

    Args:
        audio: Audio signal [n_samples].
        sr: Sample rate.
        peak_threshold_db: Detection threshold above local floor.
        min_freq: Minimum frequency to check (Hz).
        max_freq: Maximum frequency to check (Hz).

    Returns:
        List of (freq_hz, prominence_db) for each detected tone.
    """
    from scipy.signal import welch
    from scipy.ndimage import median_filter

    freqs, psd = welch(audio, fs=sr, nperseg=min(2048, len(audio)), noverlap=min(1024, len(audio) // 2))
    psd_db = 10 * np.log10(psd + 1e-20)

    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    valid_indices = np.where(freq_mask)[0]
    if len(valid_indices) == 0:
        return []

    floor_db = median_filter(psd_db, size=50)
    prominence = psd_db - floor_db

    detected = []
    for idx in valid_indices:
        if prominence[idx] > peak_threshold_db:
            if 0 < idx < len(psd_db) - 1:
                if psd_db[idx] >= psd_db[idx - 1] and psd_db[idx] >= psd_db[idx + 1]:
                    if not any(abs(freqs[idx] - f) < 20 for f, _ in detected):
                        detected.append((freqs[idx], prominence[idx]))

    return detected


class GatedEnhancer(BaseModule):
    """Wraps an enhancement module with a conditional gate.

    Skips enhancement when it would hurt ASR quality:
    - Multi-speaker: enhancement treats interfering speech as noise
    - Tonal interference: spectral subtraction artifacts > original tone
    - No speech: nothing to enhance

    Args:
        enhancer: The actual enhancement module to gate.
        detect_tones: Whether to check for tonal interference. Default True.
        tone_threshold_db: Peak prominence threshold for tone detection.
    """

    def __init__(
        self,
        enhancer: BaseModule,
        detect_tones: bool = True,
        tone_threshold_db: float = 15.0,
    ):
        super().__init__(
            name=f"{enhancer.name}-gated",
            sample_rate=enhancer.sample_rate,
        )
        self.enhancer = enhancer
        self.detect_tones = detect_tones
        self.tone_threshold_db = tone_threshold_db

    def _get_input_audio(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract the best available mono audio for analysis."""
        if "beamformed_audio" in data:
            audio = data["beamformed_audio"]
        elif "multichannel_audio" in data:
            audio = data["multichannel_audio"][0]
        else:
            return None
        if audio.ndim > 1:
            audio = audio[0]
        return audio

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        n_speakers = data.get("n_detected_sources", 1)
        has_speech = any(data.get("vad_is_speech", [True]))

        # Check for tonal interference before deciding
        tones_detected = []
        if self.detect_tones:
            audio = self._get_input_audio(data)
            if audio is not None:
                sr = data.get("sample_rate", self.sample_rate)
                tones_detected = detect_tonal_interference(
                    audio, sr, self.tone_threshold_db,
                )
                if tones_detected:
                    data["detected_tones"] = [
                        {"freq_hz": f, "prominence_db": p}
                        for f, p in tones_detected
                    ]

        # Decision logic
        if tones_detected:
            reason = (
                f"tonal_interference ({len(tones_detected)} tone(s): "
                + ", ".join(f"{f:.0f}Hz" for f, _ in tones_detected) + ")"
            )
            skip = True
        elif n_speakers > 1:
            reason = "multi_speaker"
            skip = True
        elif not has_speech:
            reason = "no_speech"
            skip = True
        else:
            reason = "single_speaker"
            skip = False

        if not skip:
            data = self.enhancer.process(data)
            data["enhancement_applied"] = True
            data["enhancement_gate_reason"] = reason
            logger.info("Enhancement gate: APPLIED (single speaker)")
        else:
            audio = self._get_input_audio(data)
            if audio is None:
                logger.warning("No audio to pass through gate")
                return data

            data["enhanced_audio"] = audio.copy()
            data["enhancement_applied"] = False
            data["enhancement_method"] = f"{self.enhancer.name} (skipped)"
            data["enhancement_latency_ms"] = 0.0
            data["enhancement_gate_reason"] = reason
            logger.info(
                f"Enhancement gate: SKIPPED ({reason}, "
                f"n_speakers={n_speakers})"
            )

        return data

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gated": True,
            "inner_enhancer": self.enhancer.get_config(),
        }

    def count_parameters(self) -> int:
        return self.enhancer.count_parameters()

    def estimate_macs(self, data: Dict[str, Any]) -> int:
        return self.enhancer.estimate_macs(data)
