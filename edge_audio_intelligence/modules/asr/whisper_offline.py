"""
Whisper Offline ASR: OpenAI Whisper for batch transcription.

V1 baseline ASR using Whisper (tiny/base/small models).
Not suitable for real-time streaming but provides strong accuracy
baseline for evaluating upstream modules.

For V2, will be compared with Emformer streaming ASR.
"""

import logging
from typing import Any, Dict

import numpy as np

from .base import BaseASR

logger = logging.getLogger(__name__)


# Global cache: avoid reloading Whisper model on each pipeline run
_whisper_cache = {}


class WhisperOfflineASR(BaseASR):
    """Whisper-based offline ASR.

    Wraps OpenAI's Whisper model for transcription.

    Args:
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
            Default 'base' - good balance of accuracy and speed for testing.
        language: Language code. Default 'en'.
        device: 'cpu' or 'cuda'. Default 'cpu'.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        device: str = "cpu",
        sample_rate: int = 16000,
    ):
        super().__init__(name=f"Whisper-{model_size}", sample_rate=sample_rate)
        self.model_size = model_size
        self.language = language
        self.device = device
        self._model = None

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": f"Whisper-{self.model_size}",
            "model_size": self.model_size,
            "language": self.language,
            "device": self.device,
            "sample_rate": self.sample_rate,
        }

    def _load_model(self):
        """Lazy-load Whisper model on first use (cached globally)."""
        if self._model is not None:
            return

        cache_key = f"{self.model_size}_{self.device}"
        if cache_key in _whisper_cache:
            self._model = _whisper_cache[cache_key]
            self._logger.info(f"Whisper {self.model_size}: using cached model")
            return

        try:
            import whisper
            import torch
            self._logger.info(f"Loading Whisper {self.model_size} on {self.device}...")
            self._model = whisper.load_model(self.model_size, device=self.device)
            
            # Research Change 3: Int8 Dynamic Quantization of encoder
            self._logger.info(f"Quantizing Whisper {self.model_size} encoder to int8...")
            self._model.encoder = torch.quantization.quantize_dynamic(
                self._model.encoder, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            _whisper_cache[cache_key] = self._model
            self._logger.info("Whisper model loaded and cached.")
        except ImportError:
            raise ImportError(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """Transcribe audio using Whisper.

        Args:
            audio: Single-channel audio [n_samples] as float32.
            sample_rate: Sample rate (must be 16000 for Whisper).

        Returns:
            Dict with 'text' and 'word_timestamps'.
        """
        self._load_model()

        # Whisper expects float32, 16kHz
        audio = audio.astype(np.float32)

        if sample_rate != 16000:
            from ...utils.audio_io import resample
            audio = resample(audio, sample_rate, 16000)

        # Run transcription
        result = self._model.transcribe(
            audio,
            language=self.language,
            fp16=False,  # CPU-safe
            word_timestamps=True,
        )

        text = result.get("text", "").strip()

        # Extract word timestamps if available
        word_timestamps = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word_timestamps.append((
                    word_info.get("word", "").strip(),
                    word_info.get("start", 0.0),
                    word_info.get("end", 0.0),
                ))

        return {
            "text": text,
            "word_timestamps": word_timestamps,
            "segments": result.get("segments", []),
        }

    def count_parameters(self) -> int:
        """Return parameter count for the Whisper model."""
        param_counts = {
            "tiny": 39_000_000,
            "base": 74_000_000,
            "small": 244_000_000,
            "medium": 769_000_000,
            "large": 1_550_000_000,
        }
        return param_counts.get(self.model_size, 0)

    def estimate_macs(self, data: Dict[str, Any]) -> int:
        """Estimate MACs for Whisper inference.

        Based on published architecture analysis.
        Source: 'Quantization for OpenAI's Whisper Models' (2025)
        """
        from ...utils.profiling import estimate_macs_for_whisper
        sr = data.get("sample_rate", 16000)
        if "enhanced_audio" in data:
            audio = data["enhanced_audio"]
            dur = (audio.shape[-1] if audio.ndim == 1 else audio.shape[-1]) / sr
        elif "beamformed_audio" in data:
            dur = data["beamformed_audio"].shape[-1] / sr
        else:
            dur = data.get("multichannel_audio", np.zeros((1, sr))).shape[-1] / sr
        return estimate_macs_for_whisper(self.model_size, dur)
