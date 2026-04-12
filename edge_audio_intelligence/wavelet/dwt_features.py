"""
DWT Feature Extraction: Decomposition and sub-band feature computation.

Provides frame-level DWT features for downstream modules.
Used by wavelet enhancement, wavelet VAD, and interpretability analysis.

Ref: CLAUDE.md Section 5.1
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DWTFeatureExtractor:
    """Extract DWT sub-band features from audio signals.

    Computes per-frame DWT decomposition and extracts energy,
    zero-crossing rate, and statistical features per sub-band.

    Args:
        wavelet: Wavelet name (pywt convention). Default 'bior2.2'.
        levels: Number of decomposition levels. Default 3.
        frame_length: Frame length in samples. Default 512.
        hop_length: Frame hop in samples. Default 256.
    """

    def __init__(
        self,
        wavelet: str = "bior2.2",
        levels: int = 3,
        frame_length: int = 512,
        hop_length: int = 256,
    ):
        self.wavelet = wavelet
        self.levels = levels
        self.frame_length = frame_length
        self.hop_length = hop_length

    def decompose(self, audio: np.ndarray) -> List[np.ndarray]:
        """Full-signal DWT decomposition.

        Args:
            audio: Audio signal [n_samples].

        Returns:
            List of coefficient arrays: [cA_J, cD_J, cD_{J-1}, ..., cD_1]
        """
        import pywt
        return pywt.wavedec(audio, self.wavelet, level=self.levels)

    def reconstruct(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """Reconstruct signal from DWT coefficients.

        Args:
            coeffs: List [cA_J, cD_J, ..., cD_1].

        Returns:
            Reconstructed signal [n_samples].
        """
        import pywt
        return pywt.waverec(coeffs, self.wavelet)

    def subband_energies(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute total energy per sub-band.

        Args:
            audio: Audio signal [n_samples].

        Returns:
            Dict {band_name: energy_value}.
        """
        coeffs = self.decompose(audio)
        energies = {}
        energies[f"cA_{self.levels}"] = float(np.sum(coeffs[0] ** 2))
        for j in range(1, len(coeffs)):
            level = self.levels - j + 1
            energies[f"cD_{level}"] = float(np.sum(coeffs[j] ** 2))
        return energies

    def frame_features(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> np.ndarray:
        """Extract per-frame DWT energy features.

        For each frame, computes DWT and returns the energy in each
        sub-band as a feature vector.

        Args:
            audio: Audio signal [n_samples].
            sr: Sample rate.

        Returns:
            Feature matrix [n_frames, n_bands].
            Bands: [E(cA_J), E(cD_J), ..., E(cD_1)]
        """
        n_samples = len(audio)
        n_frames = max(1, (n_samples - self.frame_length) // self.hop_length + 1)
        n_bands = self.levels + 1

        features = np.zeros((n_frames, n_bands), dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = audio[start:end]

            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)))

            coeffs = self.decompose(frame)
            features[i, 0] = np.sum(coeffs[0] ** 2)  # cA_J
            for j in range(1, len(coeffs)):
                features[i, j] = np.sum(coeffs[j] ** 2)  # cD levels

        return features

    def subband_frequency_ranges(self, sr: int = 16000) -> Dict[str, Tuple[float, float]]:
        """Return frequency range for each sub-band.

        For J-level decomposition at sample rate sr:
            cA_J: [0, sr / 2^{J+1}]
            cD_j: [sr / 2^{j+1}, sr / 2^j]

        Args:
            sr: Sample rate in Hz.

        Returns:
            Dict {band_name: (f_low, f_high)}.
        """
        ranges = {}
        nyquist = sr / 2

        # Approximation: [0, nyquist / 2^J]
        ranges[f"cA_{self.levels}"] = (0.0, nyquist / (2 ** self.levels))

        # Details
        for j in range(self.levels, 0, -1):
            f_low = nyquist / (2 ** j)
            f_high = nyquist / (2 ** (j - 1))
            ranges[f"cD_{j}"] = (f_low, f_high)

        return ranges
