"""
Wavelet utilities for course-specific multi-resolution analysis.

This package provides wavelet-based tools used across the pipeline:
- DWT decomposition and feature extraction
- Wavelet energy VAD (always-on listener)
- Wavelet-initialized convolution filters (for CNN-based SSL)
- Interpretability analysis and visualization
"""

from .dwt_features import DWTFeatureExtractor
from .wavelet_vad import WaveletVAD
from .wavelet_init import wavelet_init_kernel
from .analysis import WaveletAnalyzer
