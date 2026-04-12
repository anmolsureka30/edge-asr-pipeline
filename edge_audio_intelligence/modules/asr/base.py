"""
BaseASR: Abstract base class for ASR modules.

ASR modules take enhanced/separated audio and produce transcriptions
with optional word-level timestamps.
"""

import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import BaseModule


class BaseASR(BaseModule):
    """Base class for all ASR algorithms.

    Input keys required:
        - enhanced_audio: [n_samples] or [n_sources, n_samples]
          OR beamformed_audio / multichannel_audio (fallback)
        - sample_rate: int

    Output keys added:
        - transcriptions: list of str
        - word_timestamps: list of list of (word, start, end) (optional)
        - asr_method: str
        - asr_latency_ms: float
    """

    def __init__(self, name: str, sample_rate: int = 16000):
        super().__init__(name=name, sample_rate=sample_rate)

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """Core transcription method.

        Args:
            audio: Single-channel audio [n_samples].
            sample_rate: Sample rate in Hz.

        Returns:
            Dict with 'text' and optionally 'word_timestamps'.
        """
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process pipeline data dictionary."""
        # Get audio from best available source
        if "enhanced_audio" in data:
            audio = data["enhanced_audio"]
        elif "beamformed_audio" in data:
            audio = data["beamformed_audio"]
        elif "multichannel_audio" in data:
            audio = data["multichannel_audio"][0]
        else:
            raise KeyError("No audio found for ASR")

        sample_rate = data.get("sample_rate", self.sample_rate)

        # Handle multi-source: transcribe each separately
        if audio.ndim == 2:
            transcriptions = []
            word_timestamps = []
            total_latency = 0.0

            for i in range(audio.shape[0]):
                t0 = time.perf_counter()
                result = self.transcribe(audio[i], sample_rate)
                total_latency += (time.perf_counter() - t0) * 1000.0

                transcriptions.append(result.get("text", ""))
                word_timestamps.append(result.get("word_timestamps", []))

            data["transcriptions"] = transcriptions
            data["word_timestamps"] = word_timestamps
            data["asr_latency_ms"] = total_latency
        else:
            t0 = time.perf_counter()
            result = self.transcribe(audio, sample_rate)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            data["transcriptions"] = [result.get("text", "")]
            data["word_timestamps"] = [result.get("word_timestamps", [])]
            data["asr_latency_ms"] = latency_ms

        data["asr_method"] = self.name
        self._logger.info(
            f"ASR: {len(data['transcriptions'])} transcription(s), "
            f"latency={data['asr_latency_ms']:.1f}ms"
        )
        return data
