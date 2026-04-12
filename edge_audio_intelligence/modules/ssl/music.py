"""
MUSIC: MUltiple SIgnal Classification.

Eigendecomposition-based DOA estimation. Decomposes the spatial
covariance matrix into signal and noise subspaces, then scans
for directions orthogonal to the noise subspace.

Ref: PIPELINE_ALGORITHM.md Section 1.3, Eq. 1.9-1.12
    Schmidt (1986)

Eq. 1.9:  R_xx = (1/L) * sum_l X(l) * X(l)^H
Eq. 1.10: R_xx = U_s * Lambda_s * U_s^H + U_n * Lambda_n * U_n^H
Eq. 1.11: P_MUSIC(theta) = 1 / (a(theta)^H * U_n * U_n^H * a(theta))
Eq. 1.12: theta_hat = argmax P_MUSIC(theta)

Requires knowledge of the number of sources (n_sources).
Uses pyroomacoustics.doa.MUSIC for the core computation.
"""

import logging
from typing import Any, Dict

import numpy as np

from .base import BaseSSL

logger = logging.getLogger(__name__)


class MusicSSL(BaseSSL):
    """MUSIC-based Sound Source Localization.

    Provides super-resolution DOA estimation but requires
    the number of sources to be known or estimated.

    Args:
        n_fft: FFT size. Default 1024.
        n_sources: Expected number of sources. Default 1.
        grid_resolution: Number of azimuth grid points. Default 360.
        freq_range: [f_low, f_high] frequency range in Hz. Default [200, 4000].
    """

    def __init__(
        self,
        n_fft: int = 1024,
        n_sources: int = 1,
        grid_resolution: int = 360,
        freq_range: list = None,
        sample_rate: int = 16000,
    ):
        super().__init__(name="MUSIC", sample_rate=sample_rate)
        self.n_fft = n_fft
        if n_sources < 1:
            raise ValueError("n_sources must be >= 1")
        self.n_sources = n_sources
        self.grid_resolution = grid_resolution
        self.freq_range = freq_range or [200, 4000]

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": "MUSIC",
            "n_fft": self.n_fft,
            "n_sources": self.n_sources,
            "grid_resolution": self.grid_resolution,
            "freq_range": self.freq_range,
            "sample_rate": self.sample_rate,
        }

    def estimate_doa(
        self,
        multichannel_audio: np.ndarray,
        mic_positions: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """Estimate DOA using MUSIC algorithm via pyroomacoustics.

        Args:
            multichannel_audio: [n_mics, n_samples]
            mic_positions: [n_mics, 3]
            sample_rate: Sample rate in Hz.

        Returns:
            Dict with estimated_doa(s), spatial_spectrum, confidence.
        """
        import pyroomacoustics as pra

        mic_array = mic_positions.T  # [3, n_mics]
        n_mics = multichannel_audio.shape[0]
        hop = self.n_fft // 2

        # MUSIC requires n_sources < n_mics for valid eigendecomposition
        n_src = self.n_sources
        if n_src >= n_mics:
            self._logger.warning(
                f"n_sources ({n_src}) >= n_mics ({n_mics}), "
                f"reducing to {n_mics - 1} for valid eigendecomposition"
            )
            n_src = n_mics - 1

        # Build MUSIC DOA estimator
        doa = pra.doa.MUSIC(
            L=mic_array,
            fs=sample_rate,
            nfft=self.n_fft,
            num_src=n_src,
            azimuth=np.deg2rad(np.linspace(0, 360, self.grid_resolution, endpoint=False)),
        )

        # Compute STFT for each channel
        X = np.array([
            pra.transform.stft.analysis(
                multichannel_audio[m], self.n_fft, hop
            ).T
            for m in range(multichannel_audio.shape[0])
        ])

        # Run DOA estimation
        doa.locate_sources(X, freq_range=self.freq_range)

        # Extract results with safety guards
        try:
            estimated_azimuths = [
                float(np.degrees(az)) % 360.0
                for az in doa.azimuth_recon
            ]
        except (IndexError, AttributeError):
            logger.warning("MUSIC: locate_sources returned no results, defaulting to 0")
            estimated_azimuths = [0.0] * self.n_sources

        try:
            spatial_spectrum = doa.grid.values if hasattr(doa, "grid") and doa.grid is not None else np.array([])
        except AttributeError:
            spatial_spectrum = np.array([])

        # For single source, return scalar; for multi-source, return list
        if self.n_sources == 1:
            estimated_doa = estimated_azimuths[0]
        else:
            estimated_doa = estimated_azimuths

        return {
            "estimated_doa": estimated_doa,
            "doa_confidence": float(np.max(spatial_spectrum)) if len(spatial_spectrum) > 0 else 1.0,
            "n_detected_sources": self.n_sources,
            "spatial_spectrum": spatial_spectrum,
        }

    def estimate_flops(self, data: Dict[str, Any]) -> int:
        """Estimate FLOPs for MUSIC.

        STFTs + covariance matrix + eigendecomposition + spectrum scan.
        """
        n_mics = data["multichannel_audio"].shape[0]
        n_samples = data["multichannel_audio"].shape[1]
        n_freq = self.n_fft // 2 + 1
        n_frames = n_samples // (self.n_fft // 2)

        fft_flops = int(n_mics * 5 * self.n_fft * np.log2(self.n_fft) * n_frames)
        # Covariance: n_freq * n_frames * n_mics^2
        cov_flops = int(n_freq * n_frames * n_mics ** 2)
        # Eigendecomposition: ~O(n_mics^3) per frequency bin
        eig_flops = int(n_freq * n_mics ** 3 * 10)
        # Spectrum scan: grid_resolution * n_freq * n_mics
        scan_flops = int(self.grid_resolution * n_freq * n_mics * 4)

        return fft_flops + cov_flops + eig_flops + scan_flops
