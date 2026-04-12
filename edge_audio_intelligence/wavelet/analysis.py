"""
Wavelet Interpretability Analysis.

KEY WAVELET INTEGRATION POINT (CLAUDE.md Section 5.4).

At each pipeline stage, computes and visualizes the wavelet scalogram.
Shows how information flows through sub-bands at each processing step,
providing the interpretability argument for the paper.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .dwt_features import DWTFeatureExtractor

logger = logging.getLogger(__name__)


class WaveletAnalyzer:
    """Wavelet-based signal analysis and interpretability tool.

    Computes scalograms, sub-band energy distributions, and
    comparative visualizations across pipeline stages.

    Args:
        wavelet: Wavelet name. Default 'bior2.2'.
        levels: DWT decomposition levels. Default 4.
    """

    def __init__(self, wavelet: str = "bior2.2", levels: int = 4):
        self.wavelet = wavelet
        self.levels = levels
        self.feature_extractor = DWTFeatureExtractor(
            wavelet=wavelet, levels=levels
        )

    def compute_scalogram(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> Dict[str, np.ndarray]:
        """Compute DWT-based scalogram (time-scale representation).

        Args:
            audio: Audio signal [n_samples].
            sr: Sample rate.

        Returns:
            Dict with:
                'coefficients': list of sub-band coefficient arrays
                'energies': [n_frames, n_bands] frame-level energies
                'band_names': list of band names
                'freq_ranges': dict of frequency ranges per band
        """
        coeffs = self.feature_extractor.decompose(audio)
        frame_energies = self.feature_extractor.frame_features(audio, sr)
        freq_ranges = self.feature_extractor.subband_frequency_ranges(sr)

        band_names = [f"cA_{self.levels}"]
        for j in range(self.levels, 0, -1):
            band_names.append(f"cD_{j}")

        return {
            "coefficients": coeffs,
            "energies": frame_energies,
            "band_names": band_names,
            "freq_ranges": freq_ranges,
        }

    def compare_pipeline_stages(
        self,
        signals: Dict[str, np.ndarray],
        sr: int = 16000,
    ) -> Dict[str, Dict[str, float]]:
        """Compare sub-band energy distributions across pipeline stages.

        Args:
            signals: {stage_name: audio_array, ...}
                e.g., {"input": noisy, "beamformed": bf, "enhanced": enh}
            sr: Sample rate.

        Returns:
            {stage_name: {band_name: energy, ...}, ...}
        """
        results = {}
        for stage_name, audio in signals.items():
            if audio.ndim > 1:
                audio = audio[0]  # Use first channel
            results[stage_name] = self.feature_extractor.subband_energies(audio)

        return results

    def plot_scalogram(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        title: str = "Wavelet Scalogram",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot wavelet scalogram of a signal.

        Args:
            audio: Audio signal [n_samples].
            sr: Sample rate.
            title: Plot title.
            save_path: Optional path to save figure.
        """
        import matplotlib.pyplot as plt

        scalogram = self.compute_scalogram(audio, sr)
        energies = scalogram["energies"]
        band_names = scalogram["band_names"]

        fig, axes = plt.subplots(
            len(band_names) + 1, 1,
            figsize=(12, 2 * (len(band_names) + 1)),
            sharex=True,
        )

        # Plot original waveform
        t = np.arange(len(audio)) / sr
        axes[0].plot(t, audio, linewidth=0.5, color="black")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(title)

        # Plot each sub-band's coefficients
        for i, (name, coeffs) in enumerate(
            zip(band_names, scalogram["coefficients"])
        ):
            freq_range = scalogram["freq_ranges"].get(name, (0, 0))
            t_band = np.linspace(0, len(audio) / sr, len(coeffs))
            axes[i + 1].plot(t_band, coeffs, linewidth=0.5)
            axes[i + 1].set_ylabel(f"{name}\n{freq_range[0]:.0f}-{freq_range[1]:.0f}Hz")

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved scalogram to {save_path}")
        plt.close(fig)

    def plot_stage_comparison(
        self,
        signals: Dict[str, np.ndarray],
        sr: int = 16000,
        title: str = "Sub-band Energy Comparison",
        save_path: Optional[str] = None,
    ) -> None:
        """Bar chart comparing sub-band energies across pipeline stages.

        Args:
            signals: {stage_name: audio_array, ...}
            sr: Sample rate.
            title: Plot title.
            save_path: Optional save path.
        """
        import matplotlib.pyplot as plt

        stage_energies = self.compare_pipeline_stages(signals, sr)
        stages = list(stage_energies.keys())
        bands = list(stage_energies[stages[0]].keys())

        x = np.arange(len(bands))
        width = 0.8 / len(stages)
        colors = ["#3A7EE8", "#E8913A", "#4CAF50", "#E53935", "#9C27B0"]

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, stage in enumerate(stages):
            energies = [stage_energies[stage][b] for b in bands]
            # Log scale for better visualization
            energies_log = [np.log10(e + 1e-10) for e in energies]
            ax.bar(
                x + i * width, energies_log, width,
                label=stage, color=colors[i % len(colors)]
            )

        ax.set_xlabel("Sub-band")
        ax.set_ylabel("log10(Energy)")
        ax.set_title(title)
        ax.set_xticks(x + width * (len(stages) - 1) / 2)
        ax.set_xticklabels(bands, rotation=45)
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
