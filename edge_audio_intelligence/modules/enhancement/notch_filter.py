"""
NotchFilterEnhancer: Detects and removes tonal interference using notch filters.

Tonal interference (sine waves, hums, machine tones) cannot be handled by
spatial beamforming (DS) or broadband noise reduction (spectral subtraction).
A notch filter is the correct tool: it removes energy in a narrow frequency
band while preserving the rest of the spectrum.

Algorithm:
    1. Compute power spectrum of input
    2. Detect peaks that exceed the local spectral floor by > peak_threshold_db
    3. For each detected peak, apply a 2nd-order IIR notch filter
    4. Cascade multiple notches if multiple tones are detected

This is a spectral-domain approach: cheap, effective, no spatial distortion.

Ref: Parks & Burrus (1987), Digital Filter Design
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import iirnotch, sosfilt, welch

from .base import BaseEnhancer

logger = logging.getLogger(__name__)


class NotchFilterEnhancer(BaseEnhancer):
    """Detects tonal interference and removes it with notch filters.

    Args:
        peak_threshold_db: A spectral peak must exceed the local floor by this
            many dB to be classified as tonal interference. Default 15.
        notch_q: Quality factor of the notch filter. Higher Q = narrower notch.
            Default 30 (removes ~500Hz/Q ≈ 17Hz bandwidth at 500Hz).
        max_notches: Maximum number of notch filters to apply. Default 5.
        min_freq: Minimum frequency to consider (Hz). Default 50.
        max_freq: Maximum frequency to consider (Hz). Default 7500.
        floor_window: Number of frequency bins for local floor estimation.
            Default 50.
    """

    def __init__(
        self,
        peak_threshold_db: float = 15.0,
        notch_q: float = 30.0,
        max_notches: int = 5,
        min_freq: float = 50.0,
        max_freq: float = 7500.0,
        floor_window: int = 50,
        sample_rate: int = 16000,
    ):
        super().__init__(name="Notch-Filter", sample_rate=sample_rate)
        self.peak_threshold_db = peak_threshold_db
        self.notch_q = notch_q
        self.max_notches = max_notches
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.floor_window = floor_window

    def detect_tones(
        self, audio: np.ndarray, sr: int,
    ) -> List[Tuple[float, float]]:
        """Detect tonal interference frequencies in the audio.

        Returns list of (frequency_hz, peak_power_db) for detected tones.
        """
        freqs, psd = welch(audio, fs=sr, nperseg=2048, noverlap=1024)
        psd_db = 10 * np.log10(psd + 1e-20)

        # Restrict to [min_freq, max_freq]
        freq_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        valid_indices = np.where(freq_mask)[0]

        if len(valid_indices) == 0:
            return []

        # Compute local spectral floor using median filter
        # (median is robust to peaks)
        from scipy.ndimage import median_filter
        floor_db = median_filter(psd_db, size=self.floor_window)

        # Find peaks exceeding floor by threshold
        prominence = psd_db - floor_db
        detected = []

        for idx in valid_indices:
            if prominence[idx] > self.peak_threshold_db:
                # Check it's a local maximum (not just a shoulder)
                if idx > 0 and idx < len(psd_db) - 1:
                    if psd_db[idx] >= psd_db[idx - 1] and psd_db[idx] >= psd_db[idx + 1]:
                        detected.append((freqs[idx], prominence[idx]))

        # Merge nearby detections (within 20 Hz), keep strongest
        if detected:
            detected.sort(key=lambda x: -x[1])  # Sort by prominence
            merged = []
            for freq, prom in detected:
                # Check if close to an already-selected tone
                if not any(abs(freq - f) < 20 for f, _ in merged):
                    merged.append((freq, prom))
                    if len(merged) >= self.max_notches:
                        break
            detected = merged

        return detected

    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Detect tonal interference and apply notch filters.

        Args:
            audio: Input audio [n_samples], float32.
            sample_rate: Sample rate.

        Returns:
            Filtered audio with tonal interference removed.
        """
        tones = self.detect_tones(audio, sample_rate)

        if not tones:
            logger.info("Notch filter: no tonal interference detected")
            return audio.copy()

        logger.info(
            f"Notch filter: detected {len(tones)} tone(s): "
            + ", ".join(f"{f:.0f}Hz ({p:.1f}dB)" for f, p in tones)
        )

        # Apply cascaded notch filters
        filtered = audio.astype(np.float64)
        for freq, _ in tones:
            # Compute normalized frequency
            w0 = freq / (sample_rate / 2)
            if w0 >= 1.0 or w0 <= 0:
                continue

            # Design notch filter
            b, a = iirnotch(w0, self.notch_q)
            # Convert to SOS for numerical stability
            from scipy.signal import tf2sos
            sos = tf2sos(b, a)
            filtered = sosfilt(sos, filtered)

        return filtered.astype(np.float32)

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": "Notch-Filter",
            "peak_threshold_db": self.peak_threshold_db,
            "notch_q": self.notch_q,
            "max_notches": self.max_notches,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "sample_rate": self.sample_rate,
        }
