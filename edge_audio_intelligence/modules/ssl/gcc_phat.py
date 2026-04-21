"""
GCC-PHAT: Generalized Cross-Correlation with Phase Transform.

The foundational TDOA estimation method. Computes cross-correlation
between mic pairs in frequency domain with phase whitening.

Ref: PIPELINE_ALGORITHM.md Section 1.1, Eq. 1.1-1.3
    Knapp & Carter (1976)

Eq. 1.1: R_xy(f) = X(f) * Y*(f) / |X(f) * Y*(f)|
Eq. 1.2: r_xy(tau) = IFFT{ R_xy(f) }
Eq. 1.3: tau_hat = argmax_tau r_xy(tau)

DOA is then computed from TDOA:
    theta = arccos(tau * c / d)
where d = microphone spacing, c = speed of sound.
"""

import logging
from typing import Any, Dict

import numpy as np
from scipy.signal import fftconvolve

from .base import BaseSSL

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0  # m/s at 20 deg C


class GccPhatSSL(BaseSSL):
    """GCC-PHAT based Sound Source Localization.

    Estimates TDOA between mic pairs, then converts to DOA.
    For multi-mic arrays, uses the pair with largest baseline.

    Args:
        n_fft: FFT size for spectral analysis. Default 1024.
        max_tau_samples: Maximum lag in samples to search. Default None (auto).
    """

    def __init__(
        self,
        n_fft: int = 1024,
        max_tau_samples: int = None,
        sample_rate: int = 16000,
    ):
        super().__init__(name="GCC-PHAT", sample_rate=sample_rate)
        self.n_fft = n_fft
        self.max_tau_samples = max_tau_samples

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": "GCC-PHAT",
            "n_fft": self.n_fft,
            "max_tau_samples": self.max_tau_samples,
            "sample_rate": self.sample_rate,
        }

    def estimate_doa(
        self,
        multichannel_audio: np.ndarray,
        mic_positions: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """Estimate DOA using GCC-PHAT on the widest mic pair.

        Args:
            multichannel_audio: [n_mics, n_samples]
            mic_positions: [n_mics, 3]
            sample_rate: Sample rate in Hz.

        Returns:
            Dict with estimated_doa (azimuth in degrees), confidence, spectrum.
        """
        n_mics = multichannel_audio.shape[0]
        if n_mics < 2:
            raise ValueError(f"GCC-PHAT requires at least 2 microphones, got {n_mics}")

        # Find the widest-baseline mic pair for best resolution
        best_pair = (0, 1)
        max_dist = 0.0
        for i in range(n_mics):
            for j in range(i + 1, n_mics):
                dist = np.linalg.norm(mic_positions[i] - mic_positions[j])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (i, j)

        m1, m2 = best_pair
        mic_dist = max_dist
        if mic_dist < 1e-6:
            raise ValueError("Microphone pair distance is zero — identical positions")

        # Physics-based max lag: max possible TDOA for this mic pair
        max_tau = int(mic_dist / SPEED_OF_SOUND * sample_rate) + 2

        # Compute GCC-PHAT
        gcc, lags = self._gcc_phat(
            multichannel_audio[m1],
            multichannel_audio[m2],
            sample_rate,
            max_lag_override=max_tau,
        )

        # Find peak with parabolic interpolation for sub-sample precision
        peak_idx = np.argmax(gcc)
        tau_samples, confidence = self._parabolic_interpolation(gcc, lags, peak_idx)
        tau_seconds = tau_samples / sample_rate

        # Convert TDOA to DOA (azimuth)
        # theta = arccos(tau * c / d), clamped to [-1, 1]
        cos_theta = np.clip(tau_seconds * SPEED_OF_SOUND / mic_dist, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)

        # Determine the axis of the mic pair for correct azimuth
        mic_vec = mic_positions[m2] - mic_positions[m1]
        pair_angle = np.arctan2(mic_vec[1], mic_vec[0])
        azimuth = np.degrees(pair_angle + theta_rad) % 360.0

        return {
            "estimated_doa": float(azimuth),
            "doa_confidence": confidence,
            "n_detected_sources": 1,
            "tdoa_samples": int(tau_samples),
            "tdoa_seconds": float(tau_seconds),
            "spatial_spectrum": gcc,
            "mic_pair_used": best_pair,
        }

    def _wavelet_denoise_channel(self, signal: np.ndarray, wavelet='db4', level=1) -> np.ndarray:
        import pywt
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)[:len(signal)]

    def _gcc_phat(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
        fs: int,
        max_lag_override: int = None,
    ) -> tuple:
        """Compute GCC-PHAT between two signals with Wavelet pre-denoising."""
        n = len(sig1) + len(sig2) - 1
        n_fft = max(self.n_fft, int(2 ** np.ceil(np.log2(n))))

        # Research Improvement 1: Pre-denoise channel before FFT
        clean_sig1 = self._wavelet_denoise_channel(sig1)
        clean_sig2 = self._wavelet_denoise_channel(sig2)

        # Eq. 1.1: Cross-spectrum with PHAT weighting
        X1 = np.fft.rfft(clean_sig1, n=n_fft)
        X2 = np.fft.rfft(clean_sig2, n=n_fft)
        cross_spec = X1 * np.conj(X2)

        # PHAT: normalize by magnitude
        magnitude = np.abs(cross_spec)
        magnitude[magnitude < 1e-10] = 1e-10
        cross_spec_phat = cross_spec / magnitude

        # Eq. 1.2: Inverse FFT to get GCC
        # Upsample by 16x via zero-padding for sub-sample TDOA resolution.
        # Critical for mm-scale mic arrays where max TDOA < 3 samples.
        upsample_factor = 16
        gcc = np.fft.irfft(cross_spec_phat, n=n_fft * upsample_factor)

        # Determine max lag (physics-based preferred)
        if self.max_tau_samples is not None:
            max_lag = self.max_tau_samples
        elif max_lag_override is not None:
            max_lag = max_lag_override
        else:
            max_lag = n_fft // 2

        # Clamp to valid FFT range
        if max_lag > n_fft // 2:
            self._logger.warning(
                f"Max delay {max_lag} samples exceeds n_fft//2={n_fft//2}. "
                f"Consider increasing n_fft from {n_fft} to at least {max_lag*2}."
            )
            max_lag = n_fft // 2

        # Account for upsampling: lag indices are now at upsample_factor resolution
        max_lag_up = max_lag * upsample_factor
        n_gcc = len(gcc)

        # Rearrange to center zero-lag (upsampled)
        gcc = np.concatenate([gcc[-max_lag_up:], gcc[:max_lag_up + 1]])
        # Lags in original sample units (fractional)
        lags = np.arange(-max_lag_up, max_lag_up + 1) / upsample_factor

        return gcc, lags

    def _parabolic_interpolation(
        self,
        gcc: np.ndarray,
        lags: np.ndarray,
        peak_idx: int,
    ) -> tuple:
        """Refine peak location using 3-point parabolic interpolation.

        Fits a parabola through the peak and its two neighbors to get
        sub-sample TDOA precision.

        Args:
            gcc: GCC values.
            lags: Lag indices.
            peak_idx: Index of the argmax peak.

        Returns:
            Tuple of (interpolated_tau_samples, peak_value).
        """
        if peak_idx <= 0 or peak_idx >= len(gcc) - 1:
            return float(lags[peak_idx]), float(gcc[peak_idx])

        alpha = gcc[peak_idx - 1]
        beta = gcc[peak_idx]
        gamma = gcc[peak_idx + 1]

        # Parabolic interpolation: offset = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        denominator = alpha - 2 * beta + gamma
        if abs(denominator) < 1e-10:
            return float(lags[peak_idx]), float(beta)

        offset = 0.5 * (alpha - gamma) / denominator
        # offset is in units of lag spacing; convert to same units as lags
        lag_step = float(lags[1] - lags[0]) if len(lags) > 1 else 1.0
        interpolated_tau = float(lags[peak_idx]) + offset * lag_step
        interpolated_value = float(beta - 0.25 * (alpha - gamma) * offset)

        return interpolated_tau, interpolated_value

    def estimate_macs(self, data: Dict[str, Any]) -> int:
        """Estimate MACs for GCC-PHAT.

        Two FFTs + element-wise ops + one IFFT.
        FFT of length N: ~(N/2)*log2(N)*4 MACs (complex butterfly operations)
        """
        from ...utils.profiling import estimate_macs_for_fft
        n_samples = data["multichannel_audio"].shape[1]
        n_fft = max(self.n_fft, int(2 ** np.ceil(np.log2(2 * n_samples))))
        return 3 * estimate_macs_for_fft(n_fft)  # 2 forward FFTs + 1 IFFT
