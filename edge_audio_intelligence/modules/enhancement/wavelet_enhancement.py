"""
Wavelet-based Speech Enhancement: DWT sub-band denoising.

KEY WAVELET INTEGRATION POINT (CLAUDE.md Section 5.1).

Applies J-level DWT decomposition, then performs level-dependent
soft thresholding for denoising. Speech energy concentrates in
approximation coefficients (low freq), while broadband noise
distributes uniformly across all sub-bands.

Ref: PIPELINE_ALGORITHM.md Section 3.2
    Donoho & Johnstone (1994) - wavelet shrinkage

Algorithm:
    1. Decompose signal via J-level DWT
    2. Estimate noise sigma from finest detail coefficients (MAD estimator)
    3. Compute level-dependent threshold: T_j = sigma * sqrt(2 * log(N)) * scale_j
    4. Apply soft thresholding to detail coefficients at each level
    5. Reconstruct via inverse DWT

Wavelet choice: CDF 5/3 (bior2.2) for low compute cost and symmetry.
See CLAUDE.md Section 7 Key Decisions Log.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseEnhancer

logger = logging.getLogger(__name__)


class WaveletEnhancer(BaseEnhancer):
    """Wavelet-based speech enhancement via DWT sub-band thresholding.

    Args:
        wavelet: Wavelet name (pywt convention). Default 'bior2.2' (CDF 5/3).
        levels: Number of decomposition levels J. Default 3.
        threshold_mode: 'soft' or 'hard'. Default 'soft'.
        threshold_scale: Multiplier for the universal threshold.
            Values > 1.0 remove more noise (may over-smooth speech).
            Default 1.0.
        noise_estimation: Method for sigma estimation.
            'mad': Median Absolute Deviation of finest detail coeffs.
            'first_frames': Use initial frames (assumed noise-only).
            Default 'mad'.
    """

    def __init__(
        self,
        wavelet: str = "db4", # Research Change: Db4 replaces Bior2.2
        levels: int = 3,
        threshold_mode: str = "soft",
        threshold_scale: float = 1.0,
        noise_estimation: str = "mad",
        sample_rate: int = 16000,
    ):
        super().__init__(name="Wavelet-Enhancement", sample_rate=sample_rate)
        self.wavelet = wavelet
        self.levels = levels
        self.threshold_mode = threshold_mode
        self.threshold_scale = threshold_scale
        self.noise_estimation = noise_estimation

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": "Wavelet-Enhancement",
            "wavelet": self.wavelet,
            "levels": self.levels,
            "threshold_mode": self.threshold_mode,
            "threshold_scale": self.threshold_scale,
            "noise_estimation": self.noise_estimation,
            "sample_rate": self.sample_rate,
        }

    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply wavelet-based denoising to audio.

        Args:
            audio: Noisy single-channel audio [n_samples].
            sample_rate: Sample rate in Hz.

        Returns:
            Enhanced audio [n_samples].
        """
        import pywt

        n_samples = len(audio)

        # Research Change 2: Wavelet Packet Decomposition (WPD)
        # Using periodic boundary mode to maintain O(N) complexity
        wp = pywt.WaveletPacket(data=audio, wavelet=self.wavelet, mode='periodization', maxlevel=self.levels)
        
        # Extract terminal nodes ordered by frequency map
        nodes = wp.get_level(self.levels, order='freq')

        for node in nodes:
            coeffs = node.data
            
            # MAD noise scaling dynamically mapped specifically for this octave terminal
            sigma = self._estimate_noise_sigma(coeffs)
            
            # Universal adaptive threshold formula
            n_coeff = len(coeffs)
            threshold = (
                sigma 
                * np.sqrt(2 * np.log(max(n_coeff, 2))) 
                * self.threshold_scale
            )
            
            if self.threshold_mode == "soft":
                node.data = self._soft_threshold(coeffs, threshold)
            elif self.threshold_mode == "hard":
                node.data = self._hard_threshold(coeffs, threshold)
            else:
                node.data = self._soft_threshold(coeffs, threshold)

        # Reconstruct the WP tree
        enhanced = wp.reconstruct(update=False)[:n_samples].astype(np.float32)

        return enhanced

    def _estimate_noise_sigma(self, detail_coeffs: np.ndarray) -> float:
        """Estimate noise standard deviation using MAD estimator.

        sigma_hat = median(|d|) / 0.6745

        The constant 0.6745 makes this consistent with the standard
        deviation for Gaussian noise.

        Args:
            detail_coeffs: Finest-level detail coefficients.

        Returns:
            Estimated noise sigma.
        """
        return float(np.median(np.abs(detail_coeffs)) / 0.6745)

    def _soft_threshold(
        self,
        coeffs: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Soft thresholding: sign(x) * max(|x| - T, 0).

        Shrinks coefficients toward zero, reducing noise while
        preserving signal structure.
        """
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)

    def _hard_threshold(
        self,
        coeffs: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Hard thresholding: x * 1(|x| > T).

        Preserves large coefficients exactly, zeroes small ones.
        Can create artifacts at threshold boundary.
        """
        return coeffs * (np.abs(coeffs) > threshold)

    def get_subband_energies(
        self,
        audio: np.ndarray,
    ) -> Dict[str, float]:
        """Compute energy in each DWT sub-band.

        Useful for interpretability analysis (CLAUDE.md Section 5.4).

        Args:
            audio: Audio signal [n_samples].

        Returns:
            Dict mapping sub-band name to energy.
        """
        import pywt

        coeffs = pywt.wavedec(audio, self.wavelet, level=self.levels)

        energies = {}
        energies[f"cA_{self.levels}"] = float(np.sum(coeffs[0] ** 2))
        for j, detail in enumerate(coeffs[1:], 1):
            level = self.levels - j + 1
            energies[f"cD_{level}"] = float(np.sum(detail ** 2))

        return energies

    def estimate_flops(self, data: Dict[str, Any]) -> int:
        """Estimate FLOPs for wavelet enhancement.

        DWT: O(N) per level (filter + downsample)
        Thresholding: O(N) total
        IDWT: O(N) per level
        """
        if "beamformed_audio" in data:
            n_samples = len(data["beamformed_audio"])
        else:
            n_samples = data["multichannel_audio"].shape[1]

        # DWT: ~4*N FLOPs per level (convolution with short filter)
        dwt_flops = int(4 * n_samples * self.levels)
        # Thresholding: ~3*N
        thresh_flops = int(3 * n_samples)
        # IDWT: same as DWT
        idwt_flops = dwt_flops

        return dwt_flops + thresh_flops + idwt_flops
