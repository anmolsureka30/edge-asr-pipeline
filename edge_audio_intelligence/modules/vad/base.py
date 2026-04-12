"""
BaseVAD: Abstract interface for all VAD modules.

Extends BaseModule with VAD-specific output keys.
All VAD modules produce the same standardized output dictionary keys
so they can be swapped in the pipeline without changing downstream code.

Output keys added to data dict:
    vad_frame_probs:     List[float]  — P(speech) per frame
    vad_is_speech:       List[bool]   — P > speech_threshold (default 0.5)
    vad_is_noise:        List[bool]   — P < noise_threshold (default 0.3)
    vad_speech_segments: List[Tuple]  — (start_s, end_s) speech regions
    vad_method:          str          — method name
    vad_latency_ms:      float        — processing time
"""

import time
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from ..base import BaseModule

logger = logging.getLogger(__name__)


class BaseVAD(BaseModule):
    """Abstract base class for Voice Activity Detection modules.

    Subclasses must implement _detect() which returns per-frame probabilities.
    The base class handles thresholding, segment extraction, and timing.

    Args:
        name: Module name for logging.
        speech_threshold: P(speech) above this → speech frame. Default 0.5.
        noise_threshold: P(speech) below this → noise frame. Default 0.3.
            Frames between thresholds are "uncertain" (excluded from Phi_nn).
        sample_rate: Expected audio sample rate.
    """

    def __init__(
        self,
        name: str,
        speech_threshold: float = 0.5,
        noise_threshold: float = 0.3,
        sample_rate: int = 16000,
    ):
        super().__init__(name=name, sample_rate=sample_rate)
        self.speech_threshold = speech_threshold
        self.noise_threshold = noise_threshold

    @abstractmethod
    def _detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Run VAD on audio and return per-frame speech probabilities.

        Args:
            audio: Mono audio signal [n_samples], float32, normalized.
            sr: Sample rate.

        Returns:
            Array of P(speech) values, one per frame. Shape [n_frames].
        """
        pass

    @abstractmethod
    def get_frame_duration_ms(self) -> float:
        """Return the duration of each VAD frame in milliseconds."""
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run VAD and add standardized keys to data dict.

        Expects either 'beamformed_audio' or 'enhanced_audio' or
        first channel of 'multichannel_audio' as input.
        """
        sr = data.get("sample_rate", self.sample_rate)

        # Select best available mono audio
        if "enhanced_audio" in data:
            audio = data["enhanced_audio"]
        elif "beamformed_audio" in data:
            audio = data["beamformed_audio"]
        elif "multichannel_audio" in data:
            audio = data["multichannel_audio"][0]
        else:
            logger.warning("No audio found for VAD")
            return data

        if audio.ndim > 1:
            audio = audio[0]

        audio = audio.astype(np.float32)

        t0 = time.perf_counter()
        frame_probs = self._detect(audio, sr)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Apply thresholds
        is_speech = [bool(p > self.speech_threshold) for p in frame_probs]
        is_noise = [bool(p < self.noise_threshold) for p in frame_probs]

        # Extract speech segments
        frame_dur_s = self.get_frame_duration_ms() / 1000.0
        segments = self._extract_segments(is_speech, frame_dur_s)

        # Add to data dict
        data["vad_frame_probs"] = list(frame_probs)
        data["vad_is_speech"] = is_speech
        data["vad_is_noise"] = is_noise
        data["vad_speech_segments"] = segments
        data["vad_method"] = self.name
        data["vad_latency_ms"] = latency_ms
        data["vad_frame_duration_ms"] = self.get_frame_duration_ms()

        n_speech = sum(is_speech)
        n_noise = sum(is_noise)
        n_uncertain = len(frame_probs) - n_speech - n_noise
        logger.info(
            f"{self.name}: {n_speech} speech, {n_noise} noise, "
            f"{n_uncertain} uncertain frames. "
            f"{len(segments)} segments. {latency_ms:.1f}ms"
        )

        return data

    def _extract_segments(
        self,
        is_speech: List[bool],
        frame_dur_s: float,
    ) -> List[Tuple[float, float]]:
        """Convert per-frame speech labels to (start_s, end_s) segments."""
        segments = []
        in_speech = False
        start_frame = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                segments.append((
                    start_frame * frame_dur_s,
                    i * frame_dur_s,
                ))
                in_speech = False

        if in_speech:
            segments.append((
                start_frame * frame_dur_s,
                len(is_speech) * frame_dur_s,
            ))

        return segments

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "speech_threshold": self.speech_threshold,
            "noise_threshold": self.noise_threshold,
            "sample_rate": self.sample_rate,
            "frame_duration_ms": self.get_frame_duration_ms(),
        }
