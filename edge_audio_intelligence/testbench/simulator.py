"""
RIRSimulator: Generates complete acoustic scenes for pipeline evaluation.

Orchestrates the algorithms (RIR generation, signal mixing, DOA computation)
to produce AcousticScene objects with multichannel audio and ground truth.

The actual math lives in the algorithms/ package:
    - algorithms/rir.py: RIR generation (Image Source Method)
    - algorithms/signal_mixing.py: Convolution and noise addition
    - algorithms/doa.py: DOA computation

This file is the GLUE — it connects those algorithms to the scene/config
data structures used by the testbench and pipeline.

"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .scene import AcousticScene, MicArrayConfig, SceneConfig, SourceConfig
from ..algorithms.rir import RIRGenerator, convolve_rir
from ..algorithms.signal_mixing import add_noise_at_snr, generate_noise
from ..algorithms.doa import compute_doa_azimuth

logger = logging.getLogger(__name__)


class RIRSimulator:
    """Generates acoustic scenes using the algorithms package.

    Handles:
    1. Source signal generation/loading
    2. RIR computation (via algorithms.rir.RIRGenerator)
    3. Signal convolution (via algorithms.signal_mixing)
    4. Noise addition at target SNR
    5. Ground truth DOA computation (via algorithms.doa)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.rir_generator = RIRGenerator()

    def generate_scene(self, config: SceneConfig) -> AcousticScene:
        """Generate a complete acoustic scene from configuration.

        Args:
            config: Scene configuration specifying room, sources, mics.

        Returns:
            AcousticScene with multichannel audio and ground truth.
        """
        self.rng = np.random.default_rng(config.seed)
        n_samples = int(config.duration_s * config.fs)

        # ---- Generate source signals ----
        clean_sources = []
        source_positions = []
        transcriptions = []

        for src_cfg in config.sources:
            signal = self._generate_source_signal(src_cfg, n_samples, config.fs)
            clean_sources.append(signal)
            source_positions.append(np.array(src_cfg.position))
            transcriptions.append(src_cfg.transcription or "")

        # ---- Generate RIRs and simulate (via algorithms.rir) ----
        mic_positions = config.mic_array.positions

        multichannel, rirs, reverberant_sources = (
            self.rir_generator.generate_with_simulation(
                room_dim=config.room_dim,
                source_signals=clean_sources,
                source_positions=[s.position for s in config.sources],
                mic_positions=mic_positions,
                rt60=config.rt60,
                fs=config.fs,
            )
        )

        # Trim/pad to target length
        if multichannel.shape[1] > n_samples:
            multichannel = multichannel[:, :n_samples]
        elif multichannel.shape[1] < n_samples:
            pad_len = n_samples - multichannel.shape[1]
            multichannel = np.pad(multichannel, ((0, 0), (0, pad_len)))

        # NOTE: Onset/offset timing is applied BEFORE RIR convolution
        # (in _generate_source_signal). No post-RIR timing adjustment needed.
        # The source signals already have silence outside their active windows.

        # ---- Add noise (via algorithms.signal_mixing) ----
        noise = self._generate_noise(
            config.noise_type, multichannel.shape, config.fs, config.noise_path
        )
        multichannel_noisy = add_noise_at_snr(multichannel, noise, config.snr_db)

        # ---- Compute true DOAs (via algorithms.doa) ----
        mic_center = np.mean(np.array(mic_positions), axis=0)
        true_doas = [
            compute_doa_azimuth(mic_center, pos) for pos in source_positions
        ]

        # ---- Build scene ----
        scene = AcousticScene(
            config=config,
            multichannel_audio=multichannel_noisy.astype(np.float32),
            clean_sources=[s.astype(np.float32) for s in clean_sources],
            reverberant_sources=reverberant_sources,
            noise_signal=noise[0].astype(np.float32) if noise is not None else None,
            source_positions=source_positions,
            true_doas=true_doas,
            transcriptions=transcriptions,
            n_sources=len(config.sources),
            rir=rirs,
            sample_rate=config.fs,
            mic_positions=np.array(mic_positions),
        )

        logger.info(f"Generated scene: {scene}")
        return scene

    def generate_grid_scenes(
        self,
        base_config: SceneConfig,
        snr_levels: List[float] = None,
        rt60_levels: List[float] = None,
    ) -> List[AcousticScene]:
        """Generate scenes across the standardized SNR x RT60 grid.

        See CLAUDE.md Section 4.2 for the evaluation grid specification.
        """
        if snr_levels is None:
            snr_levels = [5, 15, 30]
        if rt60_levels is None:
            rt60_levels = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]

        import dataclasses
        scenes = []
        for snr in snr_levels:
            for rt60 in rt60_levels:
                cfg = dataclasses.replace(base_config, snr_db=snr, rt60=rt60)
                scenes.append(self.generate_scene(cfg))

        logger.info(f"Generated {len(scenes)} grid scenes")
        return scenes

    def generate_corner_cases(
        self,
        base_config: SceneConfig,
    ) -> List[AcousticScene]:
        """Generate the 4 corner-case scenes.

        LL: SNR=30dB, RT60=0.0s (easy)
        HL: SNR=5dB,  RT60=0.0s (noisy)
        LH: SNR=30dB, RT60=1.5s (reverberant)
        HH: SNR=5dB,  RT60=1.5s (hard)
        """
        import dataclasses
        corners = [
            (30.0, 0.0, "LL"), (5.0, 0.0, "HL"),
            (30.0, 1.5, "LH"), (5.0, 1.5, "HH"),
        ]
        scenes = []
        for snr, rt60, label in corners:
            cfg = dataclasses.replace(base_config, snr_db=snr, rt60=rt60)
            scenes.append(self.generate_scene(cfg))
            logger.info(f"Corner case {label}: SNR={snr}dB, RT60={rt60}s")
        return scenes

    def _generate_source_signal(
        self, src_cfg: SourceConfig, n_samples: int, fs: int,
    ) -> np.ndarray:
        """Generate or load a source signal, placed at the correct onset time.

        If onset_s > 0, the audio is placed starting at onset_s within a
        silence buffer of total duration n_samples. This ensures staggered
        speakers are correctly positioned in time BEFORE RIR convolution.
        """
        # First, generate/load the raw audio content
        raw_signal = self._generate_raw_signal(src_cfg, n_samples, fs)

        # Apply onset/offset timing: place audio at the correct time position
        onset_sample = int(getattr(src_cfg, 'onset_s', 0.0) * fs)
        offset_s = getattr(src_cfg, 'offset_s', -1.0)
        offset_sample = int(offset_s * fs) if offset_s > 0 else n_samples

        onset_sample = max(0, min(onset_sample, n_samples))
        offset_sample = max(onset_sample, min(offset_sample, n_samples))
        active_len = offset_sample - onset_sample

        if onset_sample == 0 and offset_sample >= n_samples:
            # No timing adjustment needed — use full signal
            return raw_signal.astype(np.float64)

        # Create silence buffer and place audio at onset position
        output = np.zeros(n_samples, dtype=np.float64)
        audio_len = min(len(raw_signal), active_len)

        # Strip any leading silence from raw_signal for cleaner placement
        # (LibriSpeech files may have short silence at start)
        output[onset_sample : onset_sample + audio_len] = raw_signal[:audio_len]

        logger.info(
            f"Source {src_cfg.label}: placed {audio_len/fs:.1f}s of audio "
            f"at {onset_sample/fs:.1f}s-{(onset_sample+audio_len)/fs:.1f}s"
        )

        return output

    def _generate_raw_signal(
        self, src_cfg: SourceConfig, n_samples: int, fs: int,
    ) -> np.ndarray:
        """Generate or load raw audio content (without timing placement)."""
        t = np.arange(n_samples) / fs

        if src_cfg.signal_type == "sine":
            signal = src_cfg.amplitude * np.sin(2 * np.pi * src_cfg.frequency * t)

        elif src_cfg.signal_type == "white_noise":
            signal = src_cfg.amplitude * self.rng.standard_normal(n_samples)

        elif src_cfg.signal_type == "chirp":
            from scipy.signal import chirp as scipy_chirp
            signal = src_cfg.amplitude * scipy_chirp(
                t, f0=200, f1=4000, t1=t[-1], method="linear"
            )

        elif src_cfg.signal_type == "speech":
            if src_cfg.audio_path and Path(src_cfg.audio_path).exists():
                from ..utils.audio_io import load_audio
                signal, _ = load_audio(src_cfg.audio_path, sr=fs, mono=True)
                # Don't pad to n_samples — return actual audio length
                # Timing placement handles the padding
                signal = signal * src_cfg.amplitude
            else:
                logger.warning(f"Speech file not found: {src_cfg.audio_path}. Using filtered noise.")
                from scipy.signal import butter, lfilter
                raw = self.rng.standard_normal(n_samples)
                b, a = butter(4, [300, 3400], btype="band", fs=fs)
                signal = src_cfg.amplitude * lfilter(b, a, raw).astype(np.float32)

        else:
            logger.warning(f"Unknown signal type: {src_cfg.signal_type}, using white noise")
            signal = src_cfg.amplitude * self.rng.standard_normal(n_samples)

        return signal.astype(np.float64)

    def _generate_noise(
        self, noise_type: str, shape: tuple, fs: int, noise_path: Optional[str] = None,
    ) -> np.ndarray:
        """Generate noise signal."""
        n_mics, n_samples = shape

        if noise_type == "white":
            return self.rng.standard_normal(shape).astype(np.float32)

        elif noise_type == "from_file" and noise_path:
            from ..utils.audio_io import load_audio
            noise_mono, _ = load_audio(noise_path, sr=fs, mono=True)
            if len(noise_mono) < n_samples:
                noise_mono = np.tile(noise_mono, (n_samples // len(noise_mono)) + 1)
            noise_mono = noise_mono[:n_samples]
            return np.tile(noise_mono, (n_mics, 1)).astype(np.float32)

        else:
            return self.rng.standard_normal(shape).astype(np.float32)
