"""
Signal Mixing: Combine sources through RIRs and add noise.

Mathematical model (PIPELINE_ALGORITHM.md Eq. 0.1):

    x_m(t) = sum_s [ h_ms(t) * s_s(t) ] + n_m(t)

    where:
        x_m(t)  = observed signal at microphone m
        h_ms(t) = RIR from source s to microphone m
        s_s(t)  = clean source signal s
        n_m(t)  = noise at microphone m
        *       = convolution

Noise scaling for target SNR:

    SNR = 10 * log10(P_signal / P_noise)
    => noise_scaled = noise * sqrt(P_signal / (P_noise * 10^(SNR/10)))
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


def mix_sources_through_rirs(
    sources: List[np.ndarray],
    rirs: List[List[np.ndarray]],
    target_length: Optional[int] = None,
) -> np.ndarray:
    """Mix multiple sources through their respective RIRs.

    Implements: x_m(t) = sum_s [ h_ms(t) * s_s(t) ]

    Args:
        sources: List of source signals, each [n_samples].
        rirs: rirs[m][s] = RIR from source s to mic m (pyroomacoustics format).
              OR rirs[s][m] — see note below.
        target_length: Trim/pad output to this length.

    Returns:
        Multichannel mixture [n_mics, target_length].
    """
    n_sources = len(sources)
    n_mics = len(rirs)

    # Determine output length
    max_len = 0
    for m in range(n_mics):
        for s in range(n_sources):
            conv_len = len(sources[s]) + len(rirs[m][s]) - 1
            max_len = max(max_len, conv_len)

    if target_length is None:
        target_length = max_len

    mixture = np.zeros((n_mics, target_length), dtype=np.float64)

    for s in range(n_sources):
        for m in range(n_mics):
            convolved = fftconvolve(sources[s], rirs[m][s], mode="full")
            length = min(len(convolved), target_length)
            mixture[m, :length] += convolved[:length]

    return mixture.astype(np.float32)


def add_noise_at_snr(
    clean_signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Add noise to a signal at a specified SNR level.

    noise_scaled = noise * sqrt(P_signal / (P_noise * 10^(SNR/10)))
    output = clean + noise_scaled

    Args:
        clean_signal: Clean signal [n_mics, n_samples] or [n_samples].
        noise: Noise signal (same shape as clean_signal).
        snr_db: Target SNR in dB.

    Returns:
        Noisy signal (same shape as input).
    """
    p_signal = np.mean(clean_signal.astype(np.float64) ** 2)
    p_noise = np.mean(noise.astype(np.float64) ** 2)

    if p_noise < 1e-10:
        return clean_signal.copy()

    scale = np.sqrt(p_signal / (p_noise * 10 ** (snr_db / 10)))
    return (clean_signal + scale * noise).astype(np.float32)


def generate_noise(
    shape: tuple,
    noise_type: str = "white",
    seed: int = 42,
) -> np.ndarray:
    """Generate noise of a given type.

    Args:
        shape: Output shape, e.g. (n_mics, n_samples) or (n_samples,).
        noise_type: 'white' for Gaussian white noise.
        seed: Random seed.

    Returns:
        Noise array of the given shape.
    """
    rng = np.random.default_rng(seed)

    if noise_type == "white":
        return rng.standard_normal(shape).astype(np.float32)
    else:
        return rng.standard_normal(shape).astype(np.float32)
