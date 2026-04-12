"""
SRP-PHAT: Steered Response Power with Phase Transform.

Scans a grid of candidate directions, computing the steered power
at each point using GCC-PHAT weights. The peak indicates the DOA.

Ref: PIPELINE_ALGORITHM.md Section 1.2, Eq. 1.4-1.8
    DiBiase (2000)

Eq. 1.4: P_SRP(theta) = sum_{pairs} GCC-PHAT(tau(theta))
Eq. 1.5: tau_ij(theta) = (d_i - d_j) . u(theta) / c

Uses pyroomacoustics.doa.SRP for the core computation.
"""

import logging
from typing import Any, Dict

import numpy as np

from .base import BaseSSL

logger = logging.getLogger(__name__)


class SrpPhatSSL(BaseSSL):
    """SRP-PHAT based Sound Source Localization.

    Wraps pyroomacoustics SRP-PHAT with standardized interface.

    Args:
        n_fft: FFT size. Default 1024.
        grid_resolution: Number of azimuth grid points. Default 360.
        freq_range: [f_low, f_high] frequency range in Hz. Default [200, 4000].
    """

    def __init__(
        self,
        n_fft: int = 1024,
        grid_resolution: int = 360,
        freq_range: list = None,
        sample_rate: int = 16000,
    ):
        super().__init__(name="SRP-PHAT", sample_rate=sample_rate)
        self.n_fft = n_fft
        self.grid_resolution = grid_resolution
        self.freq_range = freq_range or [200, 4000]

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": "SRP-PHAT",
            "n_fft": self.n_fft,
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
        """Estimate DOA using SRP-PHAT via pyroomacoustics.

        Args:
            multichannel_audio: [n_mics, n_samples]
            mic_positions: [n_mics, 3]
            sample_rate: Sample rate in Hz.

        Returns:
            Dict with estimated_doa, spatial_spectrum, confidence.
        """
        import pyroomacoustics as pra

        # pyroomacoustics expects mic positions as [n_dims, n_mics]
        mic_array = mic_positions.T  # [3, n_mics]

        # Compute STFT
        n_samples = multichannel_audio.shape[1]
        hop = self.n_fft // 2

        # Build SRP-PHAT DOA estimator
        doa = pra.doa.SRP(
            L=mic_array,
            fs=sample_rate,
            nfft=self.n_fft,
            num_src=1,
            azimuth=np.deg2rad(np.linspace(0, 360, self.grid_resolution, endpoint=False)),
        )

        # Compute STFT for each channel
        X = np.array([
            pra.transform.stft.analysis(
                multichannel_audio[m], self.n_fft, hop
            ).T
            for m in range(multichannel_audio.shape[0])
        ])
        # X shape: [n_mics, n_freq, n_frames]

        # Run DOA estimation
        doa.locate_sources(X, freq_range=self.freq_range)

        # Extract results with safety guards
        try:
            estimated_azimuth = float(np.degrees(doa.azimuth_recon[0])) % 360.0
        except (IndexError, AttributeError):
            logger.warning("SRP-PHAT: locate_sources returned no results")
            estimated_azimuth = float("nan")

        # Get spatial spectrum
        try:
            spatial_spectrum = doa.grid.values if hasattr(doa, "grid") and doa.grid is not None else np.array([])
        except AttributeError:
            spatial_spectrum = np.array([])

        return {
            "estimated_doa": estimated_azimuth,
            "doa_confidence": float(np.max(spatial_spectrum)) if len(spatial_spectrum) > 0 else 1.0,
            "n_detected_sources": 1,
            "spatial_spectrum": spatial_spectrum,
        }

    def estimate_flops(self, data: Dict[str, Any]) -> int:
        """Estimate FLOPs for SRP-PHAT.

        N_mics STFTs + grid_resolution * n_pairs * n_freq operations.
        """
        n_mics = data["multichannel_audio"].shape[0]
        n_samples = data["multichannel_audio"].shape[1]
        n_freq = self.n_fft // 2 + 1
        n_frames = n_samples // (self.n_fft // 2)
        n_pairs = n_mics * (n_mics - 1) // 2

        fft_flops = int(n_mics * 5 * self.n_fft * np.log2(self.n_fft) * n_frames)
        grid_flops = int(self.grid_resolution * n_pairs * n_freq * 6)

        return fft_flops + grid_flops
