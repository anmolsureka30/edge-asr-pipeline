"""
BaseDiarizer: Abstract base class for speaker diarization modules.

V2 scope: pyannote-based diarization.
"""

import time
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from ..base import BaseModule


class BaseDiarizer(BaseModule):
    """Base class for all diarization algorithms.

    Input keys required:
        - enhanced_audio or beamformed_audio or multichannel_audio
        - sample_rate: int
        - transcriptions: list of str (optional, from ASR)

    Output keys added:
        - speaker_segments: list of (start, end, speaker_id)
        - attributed_transcription: list of (speaker_id, text, start, end)
        - diarization_method: str
        - diarization_latency_ms: float
    """

    def __init__(self, name: str, sample_rate: int = 16000):
        super().__init__(name=name, sample_rate=sample_rate)

    @abstractmethod
    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float, str]]:
        """Identify speaker segments.

        Args:
            audio: Audio signal [n_samples].
            sample_rate: Sample rate.

        Returns:
            List of (start_time, end_time, speaker_id) tuples.
        """
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "enhanced_audio" in data:
            audio = data["enhanced_audio"]
        elif "beamformed_audio" in data:
            audio = data["beamformed_audio"]
        else:
            audio = data["multichannel_audio"][0]

        if audio.ndim > 1:
            audio = audio[0]

        sample_rate = data.get("sample_rate", self.sample_rate)

        t0 = time.perf_counter()
        segments = self.diarize(audio, sample_rate)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        data["speaker_segments"] = segments
        data["diarization_method"] = self.name
        data["diarization_latency_ms"] = latency_ms
        return data
