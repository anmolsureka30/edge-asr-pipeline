"""
Voice Activity Detection modules.

TWO-PURPOSE ARCHITECTURE (VAD_IMPLEMENTATION.md):

Purpose 1 — Always-On Wake Gate (<5mW, 24/7):
    AtomicVAD (300 params, GGCU activation, 103KB .keras model)
    Decides WHETHER to wake the full pipeline.
    False negative = delayed transcription (acceptable).
    False positive = wasted energy (not catastrophic).
    Input: 0.63s segments with 87.5% SWI overlap.

Purpose 2 — Precision Timestamps for MVDR + Enhancement Gate:
    TEN VAD (prebuilt C library, RNN-based, 16ms frames)
    Provides per-frame noise/speech labels that feed into:
      a) MVDR Phi_nn gating: only accumulate noise covariance from
         confirmed noise frames (vad_is_noise). If speech contaminates
         Phi_nn, MVDR suppresses the target speaker.
      b) Enhancement gate: skip enhancement when n_speakers > 1
         (prevents 14pp WER degradation from V1).
    Input: int16 PCM, 256 samples per frame.

Baseline — Wavelet Energy VAD (EE678 course requirement):
    WaveletVADModule (0 params, DWT ratio E(cA3)/E(cD1))
    Serves as ablation baseline against neural methods.
    Provides interpretable sub-band energy visualization.
    Input: 512-sample frames, bior2.2 wavelet, 3 levels.

Pipeline data flow:
    1. VAD runs on raw mic 0 audio BEFORE beamforming
    2. vad_is_noise → MVDR Phi_nn (noise covariance from noise-only frames)
    3. SSL → Beamforming (MVDR uses VAD-gated Phi_nn) → Enhancement (gated)
    4. vad_is_speech + n_detected_sources → Enhancement gate decision

All three methods produce standardized output keys via BaseVAD:
    vad_frame_probs, vad_is_speech, vad_is_noise,
    vad_speech_segments, vad_method, vad_latency_ms
"""

from .ten_vad_module import TENVadModule
from .wavelet_vad_module import WaveletVADModule

# AtomicVAD requires TensorFlow — import conditionally
try:
    from .atomic_vad_module import AtomicVADModule
except ImportError:
    AtomicVADModule = None
