"""
BaseSeparator: Abstract base class for speaker separation modules.

V2 scope: Conv-TasNet, Spatial Conv-TasNet.
"""

import time
from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np

from ..base import BaseModule


class BaseSeparator(BaseModule):
    """Base class for all separation algorithms.

    Input keys required:
        - enhanced_audio or beamformed_audio or multichannel_audio
        - sample_rate: int
        - n_detected_sources: int (from SSL)

    Output keys added:
        - separated_sources: list of [n_samples] arrays
        - separation_method: str
        - separation_latency_ms: float
    """

    def __init__(self, name: str, sample_rate: int = 16000):
        super().__init__(name=name, sample_rate=sample_rate)

    @abstractmethod
    def separate(
        self,
        audio: np.ndarray,
        n_sources: int,
        sample_rate: int,
    ) -> List[np.ndarray]:
        """Separate mixed audio into individual sources.

        Args:
            audio: Mixed audio [n_samples] or [n_mics, n_samples].
            n_sources: Number of sources to separate.
            sample_rate: Sample rate.

        Returns:
            List of separated source signals.
        """
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "enhanced_audio" in data:
            audio = data["enhanced_audio"]
        elif "beamformed_audio" in data:
            audio = data["beamformed_audio"]
        else:
            audio = data["multichannel_audio"]

        n_sources = data.get("n_detected_sources", 1)
        sample_rate = data.get("sample_rate", self.sample_rate)

        t0 = time.perf_counter()
        sources = self.separate(audio, n_sources, sample_rate)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        data["separated_sources"] = sources
        data["separation_method"] = self.name
        data["separation_latency_ms"] = latency_ms
        return data
