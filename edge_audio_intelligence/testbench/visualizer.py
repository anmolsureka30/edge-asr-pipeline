"""
Testbench visualization: scene layout, pipeline signals, results dashboards.

Ref: CLAUDE.md Section 3.1, ACOUSTIC_LAB.md Section 5

Re-exports general plotting from utils/visualization.py and adds
testbench-specific scene visualization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_spatial_spectrum,
    plot_metrics_comparison,
    plot_snr_rt60_grid,
)

logger = logging.getLogger(__name__)


def plot_scene_layout(
    scene: Any,
    title: str = "Acoustic Scene Layout",
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D room layout with sources, mics, and DOA arrows.

    Args:
        scene: AcousticScene object.
        title: Plot title.
        save_path: Optional path to save figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(8, 6))

    room = scene.config.room_dim
    rect = patches.Rectangle(
        (0, 0), room[0], room[1],
        linewidth=2, edgecolor="black", facecolor="#f0f0f0",
    )
    ax.add_patch(rect)

    # Plot microphones
    if scene.mic_positions is not None:
        mics = scene.mic_positions
        ax.scatter(
            mics[:, 0], mics[:, 1],
            marker="^", s=100, c="#3A7EE8", zorder=5, label="Microphones",
        )
        for i, pos in enumerate(mics):
            ax.annotate(f"M{i}", (pos[0], pos[1]), fontsize=8,
                        textcoords="offset points", xytext=(5, 5))

    # Plot sources
    for i, pos in enumerate(scene.source_positions):
        ax.scatter(
            pos[0], pos[1],
            marker="o", s=150, c="#E8913A", zorder=5,
            label="Sources" if i == 0 else None,
        )
        ax.annotate(f"S{i}", (pos[0], pos[1]), fontsize=8,
                    textcoords="offset points", xytext=(5, 5))

    # Plot DOA arrows from mic center to source directions
    if scene.mic_positions is not None and len(scene.true_doas) > 0:
        mic_center = np.mean(scene.mic_positions, axis=0)
        for i, doa in enumerate(scene.true_doas):
            dx = 0.5 * np.cos(np.radians(doa))
            dy = 0.5 * np.sin(np.radians(doa))
            ax.annotate(
                "", xy=(mic_center[0] + dx, mic_center[1] + dy),
                xytext=(mic_center[0], mic_center[1]),
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=2),
            )

    ax.set_xlim(-0.3, room[0] + 0.3)
    ax.set_ylim(-0.3, room[1] + 0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    info_text = (
        f"Room: {room[0]}x{room[1]}x{room[2]}m\n"
        f"RT60: {scene.config.rt60}s\n"
        f"SNR: {scene.config.snr_db}dB\n"
        f"Sources: {scene.n_sources}"
    )
    ax.text(
        0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved scene layout to {save_path}")
    plt.close(fig)


def plot_pipeline_signals(
    scene: Any,
    data: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:
    """Plot waveforms at each pipeline stage for visual inspection.

    Args:
        scene: AcousticScene object.
        data: Pipeline output data dict.
        save_path: Optional path to save figure.
    """
    import matplotlib.pyplot as plt

    stages = []
    if scene.multichannel_audio is not None:
        stages.append(("Input (mic 0)", scene.multichannel_audio[0]))
    if "beamformed_audio" in data:
        bf = data["beamformed_audio"]
        stages.append(("Beamformed", bf[0] if bf.ndim > 1 else bf))
    if "enhanced_audio" in data:
        enh = data["enhanced_audio"]
        stages.append(("Enhanced", enh[0] if enh.ndim > 1 else enh))
    if len(scene.clean_sources) > 0:
        stages.append(("Clean reference", scene.clean_sources[0]))

    if not stages:
        return

    sr = scene.sample_rate
    n = len(stages)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, sig) in zip(axes, stages):
        t = np.arange(len(sig)) / sr
        ax.plot(t, sig, linewidth=0.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Pipeline Signal Flow", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
