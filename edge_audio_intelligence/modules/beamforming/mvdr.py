"""
MVDR (Minimum Variance Distortionless Response) Beamformer.

Also known as Capon beamformer. Minimizes output power while
maintaining unity gain in the look direction.

Ref: PIPELINE_ALGORITHM.md Section 2.2, Eq. 2.4-2.8

Eq. 2.4: min_w  w^H * R_nn * w
          s.t.   w^H * a(theta) = 1

Eq. 2.5: w_MVDR = R_nn^{-1} * a(theta) / (a(theta)^H * R_nn^{-1} * a(theta))

Eq. 2.6: Y(f) = w_MVDR(f)^H * X(f)

R_nn is the noise covariance matrix. When the noise covariance is
unknown, the full covariance R_xx is used as an approximation.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseBeamformer

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0


class MVDRBeamformer(BaseBeamformer):
    """MVDR (Capon) Beamformer.

    Frequency-domain implementation: computes MVDR weights per
    frequency bin using the spatial covariance matrix.

    Args:
        n_fft: FFT size for STFT. Default 512.
        hop_length: STFT hop size. Default n_fft // 2.
        diagonal_loading: Regularization factor (scaled by trace(R)/n_mics).
            Default 0.01.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        diagonal_loading: float = 0.01,
        sample_rate: int = 16000,
    ):
        super().__init__(name="MVDR", sample_rate=sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.diagonal_loading = diagonal_loading

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": "MVDR",
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "diagonal_loading": self.diagonal_loading,
            "sample_rate": self.sample_rate,
        }

    def beamform(
        self,
        multichannel_audio: np.ndarray,
        mic_positions: np.ndarray,
        doa_azimuth: float,
        sample_rate: int,
        vad_is_noise: list = None,
        vad_frame_duration_ms: float = 16.0,
    ) -> np.ndarray:
        """Apply MVDR beamforming in frequency domain.

        When vad_is_noise is provided, the noise covariance matrix Phi_nn
        is computed ONLY from frames labeled as noise by VAD. This prevents
        speech contamination of Phi_nn, which would cause MVDR to suppress
        the target speaker. (VAD_IMPLEMENTATION.md Section 4.2, Eq. 2.4)

        Args:
            multichannel_audio: [n_mics, n_samples]
            mic_positions: [n_mics, 3]
            doa_azimuth: Target direction azimuth in degrees.
            sample_rate: Sample rate in Hz.
            vad_is_noise: Per-frame noise labels from VAD (True=noise).
                If None, uses all frames (original behavior).

        Returns:
            Beamformed signal [n_samples].
        """
        n_mics, n_samples = multichannel_audio.shape
        n_freq = self.n_fft // 2 + 1

        # ---- STFT ----
        from scipy.signal import stft as scipy_stft, istft as scipy_istft

        _, _, X = scipy_stft(
            multichannel_audio,
            fs=sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            axis=-1,
        )
        # X shape: [n_mics, n_freq, n_frames]
        n_frames = X.shape[2]

        # ---- Map VAD frame labels to STFT frame indices ----
        # VAD operates at hop_length samples (e.g. 256 = 16ms)
        # STFT operates at self.hop_length (e.g. 256 = 16ms)
        # Map by time alignment
        noise_frame_mask = None
        if vad_is_noise is not None and len(vad_is_noise) > 0:
            vad_hop_samples = int(vad_frame_duration_ms * sample_rate / 1000.0)
            noise_frame_mask = np.zeros(n_frames, dtype=bool)
            for stft_idx in range(n_frames):
                # Center time of this STFT frame
                stft_center_sample = stft_idx * self.hop_length + self.n_fft // 2
                # Corresponding VAD frame
                vad_idx = stft_center_sample // vad_hop_samples
                if vad_idx < len(vad_is_noise):
                    noise_frame_mask[stft_idx] = vad_is_noise[vad_idx]
                else:
                    noise_frame_mask[stft_idx] = True  # Default to noise for safety

            n_noise_frames = np.sum(noise_frame_mask)
            logger.info(
                f"MVDR Phi_nn gating: {n_noise_frames}/{n_frames} frames "
                f"({100*n_noise_frames/max(n_frames,1):.0f}%) marked as noise"
            )

        # ---- Steering vector ----
        # a_m(f) = exp(-j * 2 * pi * f * tau_m)
        theta_rad = np.radians(doa_azimuth)
        u = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])

        mic_center = np.mean(mic_positions, axis=0)
        relative_positions = mic_positions - mic_center
        delays_s = relative_positions @ u / SPEED_OF_SOUND  # [n_mics]

        freqs = np.linspace(0, sample_rate / 2, n_freq)  # [n_freq]

        # a[freq, mic] = exp(-j * 2 * pi * f * tau_m)
        steering = np.exp(
            -1j * 2 * np.pi * freqs[:, np.newaxis] * delays_s[np.newaxis, :]
        )  # [n_freq, n_mics]

        # ---- Compute covariance and MVDR weights per frequency ----
        Y = np.zeros((n_freq, n_frames), dtype=np.complex128)

        for f_idx in range(n_freq):
            X_f = X[:, f_idx, :]  # [n_mics, n_frames]

            # Eq. 2.4: Phi_nn = (1/T_noise) * sum_{noise frames} X*X^H
            if noise_frame_mask is not None and np.sum(noise_frame_mask) >= n_mics:
                # Use only noise-labeled frames for covariance
                X_noise = X_f[:, noise_frame_mask]  # [n_mics, n_noise_frames]
                R = (X_noise @ X_noise.conj().T) / X_noise.shape[1]
            else:
                # Fallback: use all frames (no VAD or too few noise frames)
                R = (X_f @ X_f.conj().T) / n_frames

            # Adaptive diagonal loading for stability
            loading = self.diagonal_loading * np.real(np.trace(R)) / n_mics
            loading = max(loading, 1e-10)
            R += loading * np.eye(n_mics)

            # Steering vector for this frequency
            a = steering[f_idx, :]  # [n_mics]

            # Eq. 2.5: w = R^{-1} a / (a^H R^{-1} a)
            try:
                R_inv = np.linalg.inv(R)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R)

            R_inv_a = R_inv @ a
            denominator = a.conj() @ R_inv_a

            if np.abs(denominator) < 1e-10:
                w = a / n_mics
            else:
                w = R_inv_a / denominator  # [n_mics]

            # Eq. 2.6: Y(f,t) = w^H * X(f,t)
            Y[f_idx, :] = w.conj() @ X_f

        # ---- ISTFT ----
        _, beamformed = scipy_istft(
            Y,
            fs=sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )

        # Trim to original length
        beamformed = beamformed[:n_samples].real.astype(np.float32)

        return beamformed

    def estimate_flops(self, data: Dict[str, Any]) -> int:
        """Estimate FLOPs for MVDR.

        STFT + covariance per freq + matrix inverse + weight computation + ISTFT.
        """
        n_mics = data["multichannel_audio"].shape[0]
        n_samples = data["multichannel_audio"].shape[1]
        n_freq = self.n_fft // 2 + 1
        n_frames = n_samples // self.hop_length

        stft_flops = int(n_mics * 5 * self.n_fft * np.log2(self.n_fft) * n_frames)
        cov_flops = int(n_freq * n_frames * n_mics ** 2)
        inv_flops = int(n_freq * n_mics ** 3 * 10)
        weight_flops = int(n_freq * n_mics * 4)
        apply_flops = int(n_freq * n_frames * n_mics * 2)
        istft_flops = int(5 * self.n_fft * np.log2(self.n_fft) * n_frames)

        return stft_flops + cov_flops + inv_flops + weight_flops + apply_flops + istft_flops
