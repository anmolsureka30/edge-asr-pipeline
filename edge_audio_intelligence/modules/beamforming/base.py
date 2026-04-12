"""
BaseBeamformer: Abstract base class for beamforming modules.

Beamformers take multichannel audio + DOA estimate and produce
a single-channel (or per-source) enhanced signal by spatial filtering.

Ref: PIPELINE_ALGORITHM.md Section 2 (Beamforming)
"""

import time
from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from ..base import BaseModule


class BaseBeamformer(BaseModule):
    """Base class for all beamforming algorithms.

    Input keys required:
        - multichannel_audio: [n_mics, n_samples]
        - mic_positions: [n_mics, 3]
        - estimated_doa: float (azimuth in degrees) from SSL
        - sample_rate: int

    Output keys added:
        - beamformed_audio: [n_samples] or [n_sources, n_samples]
        - bf_method: str
        - bf_latency_ms: float
    """

    def __init__(self, name: str, sample_rate: int = 16000):
        super().__init__(name=name, sample_rate=sample_rate)

    @abstractmethod
    def beamform(
        self,
        multichannel_audio: np.ndarray,
        mic_positions: np.ndarray,
        doa_azimuth: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Core beamforming method.

        Args:
            multichannel_audio: [n_mics, n_samples]
            mic_positions: [n_mics, 3]
            doa_azimuth: Estimated DOA azimuth in degrees.
            sample_rate: Sample rate in Hz.

        Returns:
            Beamformed signal [n_samples].
        """
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process pipeline data dictionary.

        Extracts VAD noise labels from data dict (if present) and
        passes them to beamform() for Phi_nn gating.
        """
        multichannel_audio = data["multichannel_audio"]
        mic_positions = data["mic_positions"]
        doa = data.get("estimated_doa", 0.0)
        sample_rate = data.get("sample_rate", self.sample_rate)

        # Handle multi-source DOA
        if isinstance(doa, (list, np.ndarray)):
            doa = float(np.array(doa).flat[0])

        # Extract VAD noise labels for Phi_nn gating (if available)
        vad_is_noise = data.get("vad_is_noise", None)

        t0 = time.perf_counter()

        # Pass VAD data to beamform if the method supports it
        import inspect
        bf_params = inspect.signature(self.beamform).parameters
        if "vad_is_noise" in bf_params and vad_is_noise is not None:
            vad_frame_ms = data.get("vad_frame_duration_ms", 16.0)
            beamformed = self.beamform(
                multichannel_audio, mic_positions, doa, sample_rate,
                vad_is_noise=vad_is_noise,
                vad_frame_duration_ms=vad_frame_ms,
            )
        else:
            beamformed = self.beamform(
                multichannel_audio, mic_positions, doa, sample_rate,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        data["beamformed_audio"] = beamformed
        data["bf_method"] = self.name
        data["bf_latency_ms"] = latency_ms

        self._logger.info(f"Beamformed: shape={beamformed.shape}, latency={latency_ms:.1f}ms")
        return data
