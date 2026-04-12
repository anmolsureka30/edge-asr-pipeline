"""
AtomicVAD module: Ultra-compact always-on wake gate.

Loads the pre-trained AtomicVAD model (300 parameters, GGCU activation)
directly from the .keras checkpoint. Uses Sliding Window Inference (SWI)
for streaming operation.

Key properties:
- 300 trainable parameters (~103 KB model file)
- AUROC: 0.903 on AVA-Speech
- Input: 0.63s raw audio (10,080 samples at 16kHz)
- GGCU activation: f(x) = (w1*x + b1) * cos(w2*x + b2)
- SWI: 87.5% overlap → updates every ~80ms after initial 630ms

Reference: Soto-Vergel et al., "AtomicVAD: A Tiny Voice Activity
Detection Model for Efficient Inference in Intelligent IoT Systems,"
Internet of Things, Elsevier, 2025.
"""

import logging
import os
import sys
from typing import Any, Dict

import numpy as np

from .base import BaseVAD

logger = logging.getLogger(__name__)

# Paths
_MODULE_DIR = os.path.dirname(__file__)
_ATOMIC_SRC_DIR = os.path.join(_MODULE_DIR, "atomic_vad_src")
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(_MODULE_DIR)),  # edge_audio_intelligence/
    "models", "AtomicVADGGCU-chkpt0.keras"
)

# AtomicVAD audio parameters (from training/config.py)
SEGMENT_DURATION = 0.63  # seconds
SEGMENT_SAMPLES = 10080  # 16000 * 0.63
SWI_OVERLAP = 0.875      # 87.5% overlap for sliding window

# Global model cache to avoid reloading TF model on each pipeline run (~3.4s)
_cached_model = None
_cached_model_path = None


class AtomicVADModule(BaseVAD):
    """AtomicVAD wrapper implementing BaseVAD interface.

    Loads the pre-trained .keras model with custom layers (GGCU, Spectrogram,
    SpecAugment, SpecCutout). The model takes raw 16kHz audio and internally
    computes 64-coefficient MFCCs.

    Args:
        model_path: Path to .keras checkpoint file.
        speech_threshold: P(speech) threshold for speech decision. Default 0.7.
        noise_threshold: P(speech) threshold for noise decision. Default 0.3.
        use_swi: Use Sliding Window Inference for finer granularity. Default True.
        swi_overlap: Overlap fraction for SWI. Default 0.875.
        aggregation: SWI aggregation method: 'median' or 'mean'. Default 'median'.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        speech_threshold: float = 0.7,
        noise_threshold: float = 0.3,
        use_swi: bool = True,
        swi_overlap: float = SWI_OVERLAP,
        aggregation: str = "median",
        sample_rate: int = 16000,
    ):
        super().__init__(
            name="AtomicVAD",
            speech_threshold=speech_threshold,
            noise_threshold=noise_threshold,
            sample_rate=sample_rate,
        )
        self.model_path = model_path
        self.use_swi = use_swi
        self.swi_overlap = swi_overlap
        self.aggregation = aggregation
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the pre-trained AtomicVAD .keras model (cached globally)."""
        global _cached_model, _cached_model_path

        # Use cached model if available (saves ~3.4s per run)
        if _cached_model is not None and _cached_model_path == self.model_path:
            self._model = _cached_model
            logger.info("AtomicVAD: using cached model")
            return

        try:
            if _ATOMIC_SRC_DIR not in sys.path:
                sys.path.insert(0, _ATOMIC_SRC_DIR)

            import tensorflow as tf
            from layers import GGCU, Spectrogram, SpecCutout
            from spec_augment import SpecAugment

            self._model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    "GGCU": GGCU,
                    "Spectrogram": Spectrogram,
                    "SpecAugment": SpecAugment,
                    "SpecCutout": SpecCutout,
                },
                compile=False,
            )

            _cached_model = self._model
            _cached_model_path = self.model_path

            n_params = self._model.count_params()
            logger.info(
                f"AtomicVAD loaded: {n_params} parameters, "
                f"model={os.path.basename(self.model_path)}"
            )
        except ImportError:
            logger.error(
                "TensorFlow not installed. Install with: pip install tensorflow"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load AtomicVAD model: {e}")
            raise

    def _detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Run AtomicVAD inference on audio.

        If use_swi=True, creates overlapping 0.63s windows and aggregates.
        Otherwise, processes non-overlapping 0.63s segments.
        """
        if self._model is None:
            raise RuntimeError("AtomicVAD model not loaded")

        if self.use_swi:
            return self._detect_swi(audio)
        else:
            return self._detect_spi(audio)

    def _detect_spi(self, audio: np.ndarray) -> np.ndarray:
        """Single-Pass Inference: one prediction per 0.63s segment."""
        n_segments = len(audio) // SEGMENT_SAMPLES
        if n_segments == 0:
            # Pad short audio
            padded = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
            padded[:len(audio)] = audio
            batch = padded[np.newaxis, :]
            preds = self._model.predict(batch, verbose=0)
            return np.array([preds[0, 1]], dtype=np.float32)

        probs = np.zeros(n_segments, dtype=np.float32)
        for i in range(n_segments):
            segment = audio[i * SEGMENT_SAMPLES : (i + 1) * SEGMENT_SAMPLES]
            batch = segment[np.newaxis, :].astype(np.float32)
            preds = self._model.predict(batch, verbose=0)
            probs[i] = preds[0, 1]  # P(speech) is index 1

        return probs

    def _detect_swi(self, audio: np.ndarray) -> np.ndarray:
        """Sliding Window Inference: overlapping windows, finer granularity.

        Creates 0.63s windows with swi_overlap, runs model on each,
        then maps back to frame-level decisions.
        """
        step_samples = int(SEGMENT_SAMPLES * (1.0 - self.swi_overlap))
        if step_samples < 1:
            step_samples = 1

        # Create windows
        windows = []
        starts = []
        pos = 0
        while pos + SEGMENT_SAMPLES <= len(audio):
            windows.append(audio[pos : pos + SEGMENT_SAMPLES])
            starts.append(pos)
            pos += step_samples

        if len(windows) == 0:
            # Audio shorter than one window — pad
            padded = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
            padded[:len(audio)] = audio
            windows.append(padded)
            starts.append(0)

        # Batch inference
        batch = np.array(windows, dtype=np.float32)
        preds = self._model.predict(batch, verbose=0)
        window_probs = preds[:, 1]  # P(speech) for each window

        # Map window probabilities to per-frame output
        # Each frame corresponds to one step
        n_output_frames = len(window_probs)
        return window_probs.astype(np.float32)

    def get_frame_duration_ms(self) -> float:
        if self.use_swi:
            step_samples = int(SEGMENT_SAMPLES * (1.0 - self.swi_overlap))
            return (step_samples / self.sample_rate) * 1000.0
        else:
            return (SEGMENT_SAMPLES / self.sample_rate) * 1000.0

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "model_path": os.path.basename(self.model_path),
            "use_swi": self.use_swi,
            "swi_overlap": self.swi_overlap,
            "aggregation": self.aggregation,
            "segment_duration_s": SEGMENT_DURATION,
            "segment_samples": SEGMENT_SAMPLES,
        })
        return config

    def count_parameters(self) -> int:
        if self._model is not None:
            return self._model.count_params()
        return 300  # Known from paper

    def estimate_macs(self, data: Dict[str, Any]) -> int:
        """Estimate MACs for AtomicVAD.

        Architecture: 2 depthwise conv blocks + skip + dense.
        Per 0.63s segment: ~500 MACs (extremely compact).
        """
        sr = data.get("sample_rate", 16000)
        n_samples = 0
        for key in ["enhanced_audio", "beamformed_audio", "multichannel_audio"]:
            if key in data:
                n_samples = data[key].shape[-1]
                break

        if self.use_swi:
            step_samples = int(SEGMENT_SAMPLES * (1.0 - self.swi_overlap))
            n_windows = max(1, (n_samples - SEGMENT_SAMPLES) // step_samples + 1)
        else:
            n_windows = max(1, n_samples // SEGMENT_SAMPLES)

        # MFCC: ~50K MACs per segment (FFT + mel filterbank)
        # Model: ~500 MACs per segment (300 params, 2 conv blocks)
        macs_per_window = 50500
        return n_windows * macs_per_window
