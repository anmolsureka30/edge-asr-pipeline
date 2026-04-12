"""
BaseEnhancer: Abstract base class for speech enhancement modules.

Enhancement modules take a (beamformed or raw) single-channel signal
and remove noise/artifacts, outputting a cleaner signal.

Ref: PIPELINE_ALGORITHM.md Section 3 (Enhancement)
"""

import time
from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from ..base import BaseModule


class BaseEnhancer(BaseModule):
    """Base class for all speech enhancement algorithms.

    Input keys required:
        - beamformed_audio: [n_samples] (preferred)
          OR multichannel_audio: [n_mics, n_samples] (uses first channel)
        - sample_rate: int

    Output keys added:
        - enhanced_audio: [n_samples]
        - enhancement_method: str
        - enhancement_latency_ms: float
    """

    def __init__(self, name: str, sample_rate: int = 16000):
        super().__init__(name=name, sample_rate=sample_rate)

    @abstractmethod
    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Core enhancement method.

        Args:
            audio: Single-channel audio [n_samples].
            sample_rate: Sample rate in Hz.

        Returns:
            Enhanced audio [n_samples].
        """
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process pipeline data dictionary."""
        # Use beamformed audio if available, else first channel
        if "beamformed_audio" in data:
            audio = data["beamformed_audio"]
            if audio.ndim > 1:
                audio = audio[0]
        elif "multichannel_audio" in data:
            audio = data["multichannel_audio"][0]
        else:
            raise KeyError("No audio found in data (need beamformed_audio or multichannel_audio)")

        sample_rate = data.get("sample_rate", self.sample_rate)

        t0 = time.perf_counter()
        enhanced = self.enhance(audio, sample_rate)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        data["enhanced_audio"] = enhanced
        data["enhancement_method"] = self.name
        data["enhancement_latency_ms"] = latency_ms

        self._logger.info(f"Enhanced: shape={enhanced.shape}, latency={latency_ms:.1f}ms")
        return data
