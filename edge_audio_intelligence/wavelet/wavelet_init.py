"""
Wavelet-Initialized Convolution Filters for CNN-based SSL.

KEY WAVELET INTEGRATION POINT (CLAUDE.md Section 5.2).

Initializes parallel convolution kernels using Daubechies wavelet
filter coefficients:
    kernel_size=3 -> db2 (4 coeffs, padded/trimmed to 3)
    kernel_size=5 -> db3 (6 coeffs, padded/trimmed to 5)
    kernel_size=7 -> db4 (8 coeffs, padded/trimmed to 7)

These kernels capture features at specific scales, providing
well-conditioned multi-scale analysis in the multi-stream CNN.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Mapping from kernel size to Daubechies wavelet order
KERNEL_WAVELET_MAP = {
    3: "db2",   # 4 filter coefficients
    5: "db3",   # 6 filter coefficients
    7: "db4",   # 8 filter coefficients
    9: "db5",   # 10 filter coefficients
    11: "db6",  # 12 filter coefficients
}


def wavelet_init_kernel(
    kernel_size: int,
    n_output_channels: int = 1,
    n_input_channels: int = 1,
    wavelet: Optional[str] = None,
    use_highpass: bool = False,
) -> np.ndarray:
    """Create convolution kernel initialized with wavelet filter coefficients.

    Args:
        kernel_size: Size of the convolution kernel (3, 5, 7, etc.).
        n_output_channels: Number of output channels.
        n_input_channels: Number of input channels.
        wavelet: Override wavelet name. If None, uses KERNEL_WAVELET_MAP.
        use_highpass: If True, use highpass (detail) filter. Default: lowpass.

    Returns:
        Kernel array [n_output_channels, n_input_channels, kernel_size].
    """
    import pywt

    if wavelet is None:
        if kernel_size not in KERNEL_WAVELET_MAP:
            raise ValueError(
                f"No default wavelet for kernel_size={kernel_size}. "
                f"Supported sizes: {list(KERNEL_WAVELET_MAP.keys())}. "
                f"Or provide a wavelet name explicitly."
            )
        wavelet = KERNEL_WAVELET_MAP[kernel_size]

    # Get wavelet filter coefficients
    w = pywt.Wavelet(wavelet)
    if use_highpass:
        coeffs = np.array(w.dec_hi, dtype=np.float32)
    else:
        coeffs = np.array(w.dec_lo, dtype=np.float32)

    # Adapt coefficients to kernel size
    filter_len = len(coeffs)
    if filter_len == kernel_size:
        kernel_1d = coeffs
    elif filter_len > kernel_size:
        # Center-crop to kernel_size
        start = (filter_len - kernel_size) // 2
        kernel_1d = coeffs[start:start + kernel_size]
    else:
        # Zero-pad to kernel_size
        pad_total = kernel_size - filter_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        kernel_1d = np.pad(coeffs, (pad_left, pad_right))

    # Normalize
    kernel_1d = kernel_1d / (np.linalg.norm(kernel_1d) + 1e-8)

    # Expand to multi-channel
    kernel = np.zeros(
        (n_output_channels, n_input_channels, kernel_size), dtype=np.float32
    )
    for o in range(n_output_channels):
        for i in range(n_input_channels):
            kernel[o, i, :] = kernel_1d

    return kernel


def create_multistream_kernels(
    kernel_sizes: List[int] = None,
    n_channels: int = 16,
) -> Dict[int, np.ndarray]:
    """Create a set of wavelet-initialized kernels for multi-stream CNN.

    As described in CLAUDE.md Section 5.2:
        k=3 -> db2, k=5 -> db3, k=7 -> db4

    Each kernel set contains both lowpass and highpass variants.

    Args:
        kernel_sizes: List of kernel sizes. Default [3, 5, 7].
        n_channels: Number of output channels per stream.

    Returns:
        Dict {kernel_size: kernel_array [2*n_channels, 1, kernel_size]}.
        First n_channels are lowpass-initialized, rest are highpass.
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 5, 7]

    kernels = {}
    for ks in kernel_sizes:
        lp = wavelet_init_kernel(ks, n_channels, 1, use_highpass=False)
        hp = wavelet_init_kernel(ks, n_channels, 1, use_highpass=True)

        # Add small random perturbation to break symmetry
        rng = np.random.default_rng(42)
        lp += rng.standard_normal(lp.shape).astype(np.float32) * 0.01
        hp += rng.standard_normal(hp.shape).astype(np.float32) * 0.01

        kernels[ks] = np.concatenate([lp, hp], axis=0)
        logger.info(
            f"Created multi-stream kernel k={ks}: shape={kernels[ks].shape}, "
            f"wavelet={KERNEL_WAVELET_MAP.get(ks, 'custom')}"
        )

    return kernels
