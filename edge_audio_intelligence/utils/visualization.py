"""
Visualization utilities for pipeline analysis.

Provides plotting functions for signals, spectra, metrics tables,
and wavelet decompositions. Uses matplotlib.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def plot_waveform(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = "Waveform",
    save_path: Optional[str] = None,
) -> None:
    """Plot audio waveform."""
    import matplotlib.pyplot as plt

    t = np.arange(len(audio)) / sr
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, audio, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved waveform plot to {save_path}")
    plt.close(fig)


def plot_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = "Spectrogram",
    save_path: Optional[str] = None,
) -> None:
    """Plot spectrogram of audio signal."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(audio, Fs=sr, NFFT=512, noverlap=256, cmap="viridis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_spectrum(
    angles: np.ndarray,
    spectrum: np.ndarray,
    true_doa: Optional[float] = None,
    title: str = "Spatial Spectrum",
    save_path: Optional[str] = None,
) -> None:
    """Plot spatial spectrum (e.g., from SRP-PHAT or MUSIC).

    Args:
        angles: Array of angles in degrees.
        spectrum: Power/pseudo-spectrum values.
        true_doa: Ground truth DOA for reference line.
        title: Plot title.
        save_path: Optional path to save figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(angles, spectrum, "b-", linewidth=1.5)

    if true_doa is not None:
        ax.axvline(true_doa, color="r", linestyle="--", label=f"True DOA={true_doa:.1f} deg")
        ax.legend()

    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing a metric across multiple methods.

    Args:
        results: {method_name: {metric_name: value, ...}, ...}
        metric_name: Which metric to plot.
        title: Plot title.
        save_path: Optional save path.
    """
    import matplotlib.pyplot as plt

    methods = list(results.keys())
    values = [results[m].get(metric_name, 0) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(methods, values, color=["#3A7EE8", "#E8913A", "#4CAF50", "#E53935"][:len(methods)])
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} Comparison")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_snr_rt60_grid(
    grid_results: Dict[Tuple[float, float], float],
    metric_name: str,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Heatmap of metric values across SNR x RT60 grid.

    Args:
        grid_results: {(snr, rt60): metric_value, ...}
        metric_name: Name of the metric being plotted.
        title: Plot title.
        save_path: Optional save path.
    """
    import matplotlib.pyplot as plt

    snrs = sorted(set(k[0] for k in grid_results.keys()))
    rt60s = sorted(set(k[1] for k in grid_results.keys()))

    grid = np.zeros((len(snrs), len(rt60s)))
    for i, snr in enumerate(snrs):
        for j, rt60 in enumerate(rt60s):
            grid[i, j] = grid_results.get((snr, rt60), float("nan"))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(rt60s)))
    ax.set_xticklabels([f"{r:.1f}" for r in rt60s])
    ax.set_yticks(range(len(snrs)))
    ax.set_yticklabels([f"{s}" for s in snrs])
    ax.set_xlabel("RT60 (s)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title(title or f"{metric_name} across SNR x RT60")

    # Annotate cells
    for i in range(len(snrs)):
        for j in range(len(rt60s)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, label=metric_name)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
