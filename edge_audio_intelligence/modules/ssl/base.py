"""
BaseSSL: Abstract base class for Sound Source Localization modules.

All SSL modules take multichannel audio and microphone positions,
and output estimated DOA(s) with confidence.

Ref: PIPELINE_ALGORITHM.md Section 1 (Sound Source Localization)
"""

from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from ..base import BaseModule


class BaseSSL(BaseModule):
    """Base class for all SSL algorithms.

    Input keys required:
        - multichannel_audio: [n_mics, n_samples]
        - sample_rate: int
        - mic_positions: [n_mics, 3]

    Output keys added:
        - estimated_doa: float or [n_frames, n_sources, 2]
        - doa_confidence: float or [n_frames, n_sources]
        - n_detected_sources: int
        - ssl_method: str
        - ssl_latency_ms: float
        - spatial_spectrum: [n_directions] (optional)
    """

    def __init__(self, name: str, sample_rate: int = 16000):
        super().__init__(name=name, sample_rate=sample_rate)

    @abstractmethod
    def estimate_doa(
        self,
        multichannel_audio: np.ndarray,
        mic_positions: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """Core DOA estimation method.

        Args:
            multichannel_audio: [n_mics, n_samples]
            mic_positions: [n_mics, 3]
            sample_rate: Sample rate in Hz.

        Returns:
            Dict with at minimum 'estimated_doa' key.
        """
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process pipeline data dictionary.

        Extracts required inputs, runs DOA estimation, adds results.
        """
        import time

        multichannel_audio = data["multichannel_audio"]
        mic_positions = data["mic_positions"]
        sample_rate = data.get("sample_rate", self.sample_rate)

        t0 = time.perf_counter()
        result = self.estimate_doa(multichannel_audio, mic_positions, sample_rate)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Add standard keys
        data["estimated_doa"] = result["estimated_doa"]
        data["doa_confidence"] = result.get("doa_confidence", 1.0)
        data["n_detected_sources"] = result.get("n_detected_sources", 1)
        data["ssl_method"] = self.name
        data["ssl_latency_ms"] = latency_ms

        if "spatial_spectrum" in result:
            data["spatial_spectrum"] = result["spatial_spectrum"]

        self._logger.info(
            f"DOA estimate: {data['estimated_doa']}, "
            f"latency: {latency_ms:.1f}ms"
        )
        return data
