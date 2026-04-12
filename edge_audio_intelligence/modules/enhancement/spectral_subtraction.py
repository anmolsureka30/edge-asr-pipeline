"""
Spectral Subtraction: Classic noise reduction method.

Estimates noise spectrum from non-speech segments (or initial frames),
then subtracts it from the noisy spectrum. Uses oversubtraction and
spectral flooring to reduce musical noise.

Ref: PIPELINE_ALGORITHM.md Section 3.1, Eq. 3.1-3.4
    Boll (1979)

Eq. 3.1: |S_hat(f)|^2 = |X(f)|^2 - alpha * |N_hat(f)|^2
Eq. 3.2: |S_hat(f)|^2 = max(|S_hat(f)|^2, beta * |X(f)|^2)  (spectral floor)
Eq. 3.3: S_hat(f) = |S_hat(f)| * exp(j * angle(X(f)))   (phase reuse)
Eq. 3.4: N_hat updated via exponential moving average of noise frames
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

from .base import BaseEnhancer

logger = logging.getLogger(__name__)


class SpectralSubtractionEnhancer(BaseEnhancer):
    """Spectral Subtraction speech enhancer.

    Args:
        n_fft: FFT size. Default 512.
        hop_length: STFT hop size. Default 256.
        alpha: Oversubtraction factor. Default 2.0.
            Higher values remove more noise but may distort speech.
        beta: Spectral floor factor. Default 0.01.
            Prevents negative power estimates (reduces musical noise).
        noise_frames: Number of initial frames assumed to be noise-only.
            Default 10 (~160ms at 16kHz with 256 hop).
        noise_update_rate: Exponential averaging rate for noise update.
            Default 0.98.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        alpha: float = 2.0,
        beta: float = 0.01,
        noise_frames: int = 10,
        noise_update_rate: float = 0.98,
        sample_rate: int = 16000,
    ):
        super().__init__(name="Spectral-Subtraction", sample_rate=sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        self.beta = beta
        self.noise_frames = noise_frames
        self.noise_update_rate = noise_update_rate

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": "Spectral-Subtraction",
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "alpha": self.alpha,
            "beta": self.beta,
            "noise_frames": self.noise_frames,
            "noise_update_rate": self.noise_update_rate,
            "sample_rate": self.sample_rate,
        }

    def process(self, data):
        """Override to pass VAD noise labels to enhance()."""
        self._vad_is_noise = data.get("vad_is_noise", None)
        return super().process(data)

    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply spectral subtraction to noisy audio.

        Args:
            audio: Noisy single-channel audio [n_samples].
            sample_rate: Sample rate in Hz.

        Returns:
            Enhanced audio [n_samples].
        """
        from scipy.signal import stft as scipy_stft, istft as scipy_istft

        # ---- STFT ----
        _, _, X = scipy_stft(
            audio,
            fs=sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        # X shape: [n_freq, n_frames]

        magnitude = np.abs(X)
        phase = np.angle(X)
        power = magnitude ** 2

        n_freq, n_frames = X.shape

        # ---- Estimate noise power (Eq. 3.4) ----
        # If VAD noise labels are available, use them for better noise estimation.
        # Otherwise fall back to first N frames (assumes initial silence).
        vad_noise_labels = getattr(self, '_vad_is_noise', None)
        if vad_noise_labels is not None and len(vad_noise_labels) > 0:
            # Map VAD frames to STFT frames and select noise-only frames
            vad_hop = 256  # VAD hop in samples
            noise_frame_indices = []
            for t in range(n_frames):
                stft_center = t * self.hop_length + self.n_fft // 2
                vad_idx = stft_center // vad_hop
                if vad_idx < len(vad_noise_labels) and vad_noise_labels[vad_idx]:
                    noise_frame_indices.append(t)

            if len(noise_frame_indices) >= 3:
                noise_power = np.mean(power[:, noise_frame_indices], axis=1)
                logger.info(f"SpectSub: using {len(noise_frame_indices)}/{n_frames} VAD noise frames")
            else:
                noise_frames_count = min(self.noise_frames, n_frames)
                noise_power = np.mean(power[:, :noise_frames_count], axis=1)
                logger.info("SpectSub: too few VAD noise frames, using initial frames")
        else:
            noise_frames_count = min(self.noise_frames, n_frames)
            noise_power = np.mean(power[:, :noise_frames_count], axis=1)

        # ---- Spectral subtraction per frame ----
        enhanced_power = np.zeros_like(power)

        for t in range(n_frames):
            # Eq. 3.1: Subtract overestimated noise
            subtracted = power[:, t] - self.alpha * noise_power

            # Eq. 3.2: Spectral floor to prevent negative values
            floor = self.beta * power[:, t]
            enhanced_power[:, t] = np.maximum(subtracted, floor)

            # Eq. 3.4: Update noise estimate (simple approach)
            # If frame power is close to noise level, update noise estimate
            snr_local = power[:, t] / (noise_power + 1e-10)
            noise_mask = snr_local < 2.0  # frames likely noise-dominated
            if np.mean(noise_mask) > 0.5:
                noise_power = (
                    self.noise_update_rate * noise_power
                    + (1 - self.noise_update_rate) * power[:, t]
                )

        # Eq. 3.3: Reconstruct with original phase
        enhanced_magnitude = np.sqrt(enhanced_power)
        X_enhanced = enhanced_magnitude * np.exp(1j * phase)

        # ---- ISTFT ----
        _, enhanced = scipy_istft(
            X_enhanced,
            fs=sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )

        # Trim to original length
        enhanced = enhanced[:len(audio)].real.astype(np.float32)

        return enhanced

    def estimate_flops(self, data: Dict[str, Any]) -> int:
        """Estimate FLOPs for spectral subtraction."""
        if "beamformed_audio" in data:
            n_samples = len(data["beamformed_audio"])
        else:
            n_samples = data["multichannel_audio"].shape[1]

        n_frames = n_samples // self.hop_length
        n_freq = self.n_fft // 2 + 1

        stft_flops = int(5 * self.n_fft * np.log2(self.n_fft) * n_frames)
        subtraction_flops = int(n_freq * n_frames * 5)  # subtract + max + phase
        istft_flops = stft_flops

        return stft_flops + subtraction_flops + istft_flops
