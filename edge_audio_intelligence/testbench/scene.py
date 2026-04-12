"""
AcousticScene: Defines room geometry, sources, microphones, and ground truth.

This is the fundamental data structure for all simulation and evaluation.
See ACOUSTIC_LAB.md Section 2 for scene generation philosophy.

The scene encapsulates everything needed to reproduce an experiment:
room dimensions, wall absorption, source positions and signals,
microphone array geometry, and all ground truth labels.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Configuration for a single sound source.

    Attributes:
        position: [x, y, z] position in meters.
        signal_type: 'speech', 'noise', 'sine', 'chirp', 'white_noise'.
        audio_path: Path to audio file (for speech sources).
        frequency: Frequency in Hz (for sine/chirp sources).
        amplitude: Signal amplitude multiplier.
        label: Speaker ID or source label.
        transcription: Ground truth text (for speech sources).
        onset_s: When this source starts (seconds from sim start). Default 0.0.
        offset_s: When this source stops (-1 = end of simulation). Default -1.
    """
    position: List[float]
    signal_type: str = "speech"
    audio_path: Optional[str] = None
    frequency: float = 440.0
    amplitude: float = 1.0
    label: str = "S0"
    transcription: Optional[str] = None
    onset_s: float = 0.0
    offset_s: float = -1.0


@dataclass
class MicArrayConfig:
    """Configuration for the microphone array.

    Attributes:
        positions: [n_mics, 3] microphone positions in meters.
        array_type: 'linear', 'circular', 'custom'.
        center: [x, y, z] array center position.
    """
    positions: List[List[float]] = field(default_factory=list)
    array_type: str = "linear"
    center: Optional[List[float]] = None

    @staticmethod
    def linear_array(
        n_mics: int = 4,
        spacing: float = 0.015,
        center: List[float] = None,
        height: float = 1.2,
    ) -> "MicArrayConfig":
        """Create a uniform linear array (ULA).

        Args:
            n_mics: Number of microphones.
            spacing: Inter-element spacing in meters. Default 15mm (edge device).
            center: [x, y] center of the array.
            height: Array height in meters.

        Returns:
            MicArrayConfig with computed positions.
        """
        if center is None:
            center = [3.0, 3.0]

        positions = []
        total_length = (n_mics - 1) * spacing
        start_x = center[0] - total_length / 2

        for i in range(n_mics):
            positions.append([start_x + i * spacing, center[1], height])

        return MicArrayConfig(
            positions=positions,
            array_type="linear",
            center=[center[0], center[1], height],
        )

    @staticmethod
    def circular_array(
        n_mics: int = 4,
        radius: float = 0.035,
        center: List[float] = None,
        height: float = 1.2,
    ) -> "MicArrayConfig":
        """Create a uniform circular array (UCA).

        Args:
            n_mics: Number of microphones.
            radius: Array radius in meters.
            center: [x, y] center of the array.
            height: Array height in meters.

        Returns:
            MicArrayConfig with computed positions.
        """
        if center is None:
            center = [3.0, 3.0]

        positions = []
        for i in range(n_mics):
            angle = 2 * np.pi * i / n_mics
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            positions.append([x, y, height])

        return MicArrayConfig(
            positions=positions,
            array_type="circular",
            center=[center[0], center[1], height],
        )

    @staticmethod
    def phone_2mic(
        center: List[float] = None,
        height: float = 1.2,
    ) -> "MicArrayConfig":
        """Smartphone-like 2-mic array, 14mm spacing.

        Real phone mic spacing: ~14mm between bottom microphones.
        """
        return MicArrayConfig.linear_array(
            n_mics=2, spacing=0.014, center=center, height=height,
        )

    @staticmethod
    def smart_speaker_4mic(
        center: List[float] = None,
        height: float = 1.2,
    ) -> "MicArrayConfig":
        """Smart-speaker circular 4-mic array, 29mm radius (58mm diameter).

        Matches ReSpeaker Lite 2-Mic / small circular array form factor.
        """
        return MicArrayConfig.circular_array(
            n_mics=4, radius=0.029, center=center, height=height,
        )


@dataclass
class SceneConfig:
    """Complete configuration for an acoustic scene.

    This is the top-level config that defines everything for a simulation run.
    All parameters are logged for reproducibility.

    Attributes:
        room_dim: [length, width, height] in meters.
        rt60: Reverberation time in seconds (0.0 for anechoic).
        snr_db: Signal-to-noise ratio in dB.
        sources: List of source configurations.
        mic_array: Microphone array configuration.
        duration_s: Simulation duration in seconds.
        fs: Sample rate in Hz.
        noise_type: Type of additive noise ('white', 'babble', 'from_file').
        noise_path: Path to noise audio file (for 'from_file' type).
        seed: Random seed for reproducibility.
    """
    room_dim: List[float] = field(default_factory=lambda: [6.0, 5.0, 3.0])
    rt60: float = 0.3
    snr_db: float = 15.0
    sources: List[SourceConfig] = field(default_factory=list)
    mic_array: MicArrayConfig = field(default_factory=MicArrayConfig)
    duration_s: float = 3.0
    fs: int = 16000
    noise_type: str = "white"
    noise_path: Optional[str] = None
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging."""
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class AcousticScene:
    """A fully realized acoustic scene with signals and ground truth.

    Created by RIRSimulator from a SceneConfig. Contains all the data
    needed to run and evaluate the pipeline.
    """
    config: SceneConfig

    # Signals
    multichannel_audio: np.ndarray = field(default=None)  # [n_mics, n_samples]
    clean_sources: List[np.ndarray] = field(default_factory=list)  # per-source dry clean
    reverberant_sources: Optional[np.ndarray] = None  # [n_sources, n_mics, n_samples]
    noise_signal: Optional[np.ndarray] = None  # [n_samples]

    # Ground truth
    source_positions: List[np.ndarray] = field(default_factory=list)  # [x,y,z]
    true_doas: List[float] = field(default_factory=list)  # azimuth in degrees
    transcriptions: List[str] = field(default_factory=list)
    speaker_labels: List[Tuple[float, float, str]] = field(default_factory=list)
    n_sources: int = 0

    # Metadata
    rir: Optional[np.ndarray] = None  # [n_mics, n_sources, rir_length]
    sample_rate: int = 16000
    mic_positions: Optional[np.ndarray] = None  # [n_mics, 3]

    def to_pipeline_dict(self) -> Dict[str, Any]:
        """Convert to the standardized pipeline data dictionary.

        See CLAUDE.md Section 3.3 for the SCENE INPUT specification.
        """
        return {
            "multichannel_audio": self.multichannel_audio,
            "sample_rate": self.sample_rate,
            "mic_positions": self.mic_positions,
            "ground_truth": {
                "doa_per_source": [np.array([doa]) for doa in self.true_doas],
                "clean_sources": self.clean_sources,
                "transcriptions": self.transcriptions,
                "speaker_labels": self.speaker_labels,
                "n_sources": self.n_sources,
            },
        }

    def __repr__(self) -> str:
        room = self.config.room_dim
        return (
            f"AcousticScene(room={room[0]}x{room[1]}x{room[2]}m, "
            f"RT60={self.config.rt60}s, SNR={self.config.snr_db}dB, "
            f"sources={self.n_sources}, mics={len(self.config.mic_array.positions)})"
        )
