"""
Delay-and-Sum Beamformer.

The simplest beamforming method: delays each mic signal to align the
target direction, then averages. Provides array gain = n_mics for
spatially white noise.

Ref: PIPELINE_ALGORITHM.md Section 2.1, Eq. 2.1-2.3

Eq. 2.1: tau_m(theta) = (p_m . u(theta)) / c
Eq. 2.2: y(t) = (1/M) * sum_m x_m(t - tau_m)
Eq. 2.3: W_DS(f, theta) = (1/M) * sum_m exp(-j*2*pi*f*tau_m)

Array gain for M mics: G = M (in spatially white noise)
"""

import logging
from typing import Any, Dict

import numpy as np

from .base import BaseBeamformer

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0


class DelayAndSumBeamformer(BaseBeamformer):
    """Delay-and-Sum (DS) Beamformer.

    Time-domain implementation: compute per-mic delays from DOA,
    apply fractional delays via sinc interpolation, then average.

    Args:
        use_fractional_delay: If True, use sinc interpolation for
            sub-sample delays. If False, use nearest-sample rounding.
    """

    def __init__(
        self,
        use_fractional_delay: bool = True,
        sample_rate: int = 16000,
    ):
        super().__init__(name="Delay-and-Sum", sample_rate=sample_rate)
        self.use_fractional_delay = use_fractional_delay

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": "Delay-and-Sum",
            "use_fractional_delay": self.use_fractional_delay,
            "sample_rate": self.sample_rate,
        }

    def beamform(
        self,
        multichannel_audio: np.ndarray,
        mic_positions: np.ndarray,
        doa_azimuth: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply delay-and-sum beamforming.

        Args:
            multichannel_audio: [n_mics, n_samples]
            mic_positions: [n_mics, 3]
            doa_azimuth: Target direction azimuth in degrees.
            sample_rate: Sample rate in Hz.

        Returns:
            Beamformed signal [n_samples].
        """
        n_mics, n_samples = multichannel_audio.shape

        # Eq. 2.1: Compute per-mic delays
        # Unit vector toward source
        theta_rad = np.radians(doa_azimuth)
        u = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])

        # Delays relative to array center
        mic_center = np.mean(mic_positions, axis=0)
        relative_positions = mic_positions - mic_center

        # tau_m = (p_m . u) / c  (in seconds)
        delays_s = relative_positions @ u / SPEED_OF_SOUND
        delays_samples = delays_s * sample_rate

        # Normalize delays so minimum is 0 (all delays are non-negative)
        delays_samples -= np.min(delays_samples)

        # Eq. 2.2: Delay and sum
        output = np.zeros(n_samples, dtype=np.float64)

        for m in range(n_mics):
            delay = delays_samples[m]

            if self.use_fractional_delay:
                aligned = self._fractional_delay(
                    multichannel_audio[m], delay
                )
            else:
                # Integer delay: shift by rounding
                shift = int(np.round(delay))
                aligned = np.zeros(n_samples)
                if shift >= 0 and shift < n_samples:
                    aligned[shift:] = multichannel_audio[m, :n_samples - shift]

            output += aligned[:n_samples]

        # Average (Eq. 2.2)
        output /= n_mics

        return output.astype(np.float32)

    def _fractional_delay(
        self,
        signal: np.ndarray,
        delay_samples: float,
    ) -> np.ndarray:
        """Apply fractional delay using frequency-domain phase shift.

        More accurate than sinc interpolation for arbitrary delays.

        Args:
            signal: Input signal [n_samples].
            delay_samples: Delay in samples (can be fractional).

        Returns:
            Delayed signal [n_samples].
        """
        n = len(signal)
        n_fft = int(2 ** np.ceil(np.log2(n)))

        # FFT
        X = np.fft.rfft(signal, n=n_fft)

        # Phase shift to advance signal by delay_samples:
        # H(k) = exp(+j * 2π * k * d / N) shifts signal LEFT by d samples
        # This aligns the delayed mic signal with the reference mic.
        freqs = np.arange(len(X))
        phase_shift = np.exp(1j * 2 * np.pi * freqs * delay_samples / n_fft)
        X_delayed = X * phase_shift

        # IFFT
        delayed = np.fft.irfft(X_delayed, n=n_fft)
        return delayed[:n].real

    def estimate_flops(self, data: Dict[str, Any]) -> int:
        """Estimate FLOPs for Delay-and-Sum.

        M FFTs + M phase shifts + M IFFTs + M additions.
        """
        n_mics = data["multichannel_audio"].shape[0]
        n_samples = data["multichannel_audio"].shape[1]
        n_fft = int(2 ** np.ceil(np.log2(n_samples)))

        if self.use_fractional_delay:
            fft_flops = int(n_mics * 2 * 5 * n_fft * np.log2(n_fft))  # FFT + IFFT
            phase_flops = int(n_mics * n_fft * 6)  # complex multiply
            sum_flops = int(n_mics * n_samples)
            return fft_flops + phase_flops + sum_flops
        else:
            return int(n_mics * n_samples * 2)  # shift + add
