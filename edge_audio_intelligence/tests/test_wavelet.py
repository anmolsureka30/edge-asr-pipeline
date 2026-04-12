"""Tests for wavelet utilities."""

import numpy as np
import pytest

from edge_audio_intelligence.wavelet.dwt_features import DWTFeatureExtractor
from edge_audio_intelligence.wavelet.wavelet_vad import WaveletVAD
from edge_audio_intelligence.wavelet.wavelet_init import wavelet_init_kernel, create_multistream_kernels
from edge_audio_intelligence.wavelet.analysis import WaveletAnalyzer


class TestDWTFeatures:
    """DWT feature extraction tests."""

    def test_decompose_reconstruct(self):
        """DWT decompose then reconstruct should give back original."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(1024).astype(np.float64)

        extractor = DWTFeatureExtractor(wavelet="db4", levels=3)
        coeffs = extractor.decompose(signal)
        reconstructed = extractor.reconstruct(coeffs)

        np.testing.assert_allclose(
            signal, reconstructed[:len(signal)], atol=1e-10
        )

    def test_subband_energies_positive(self):
        """All sub-band energies should be non-negative."""
        signal = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
        extractor = DWTFeatureExtractor()
        energies = extractor.subband_energies(signal)

        assert all(e >= 0 for e in energies.values())
        assert len(energies) == 4  # cA_3, cD_3, cD_2, cD_1

    def test_frame_features_shape(self):
        """Frame features should have correct shape."""
        signal = np.random.randn(16000).astype(np.float32)
        extractor = DWTFeatureExtractor(
            frame_length=512, hop_length=256, levels=3
        )
        features = extractor.frame_features(signal)

        expected_frames = (16000 - 512) // 256 + 1
        assert features.shape == (expected_frames, 4)

    def test_frequency_ranges(self):
        """Frequency ranges should span 0 to Nyquist."""
        extractor = DWTFeatureExtractor(levels=3)
        ranges = extractor.subband_frequency_ranges(sr=16000)

        # Should cover 0 to 8000 Hz (Nyquist)
        assert ranges["cA_3"][0] == 0.0
        assert ranges["cD_1"][1] == 8000.0


class TestWaveletVAD:
    """Wavelet VAD tests."""

    def test_detect_speech_in_sine(self):
        """Sine wave (tonal) should be detected as speech-like."""
        # Sine at 500Hz has energy concentrated in low frequency bands
        t = np.arange(16000) / 16000
        speech_like = np.sin(2 * np.pi * 500 * t).astype(np.float32)

        vad = WaveletVAD(threshold=1.5)
        mask = vad.detect(speech_like)

        # Most frames should be detected
        assert np.mean(mask) > 0.3

    def test_silence_not_detected(self):
        """Very low amplitude noise should not be detected."""
        rng = np.random.default_rng(42)
        silence = (rng.standard_normal(16000) * 0.001).astype(np.float32)

        vad = WaveletVAD(threshold=5.0)
        mask = vad.detect(silence)

        # Very few or no frames detected
        assert np.mean(mask) < 0.3

    def test_detect_segments_returns_tuples(self):
        """Segments should be (start, end) time tuples."""
        signal = np.sin(2 * np.pi * 500 * np.arange(16000) / 16000).astype(np.float32)

        vad = WaveletVAD(threshold=1.5)
        segments = vad.detect_segments(signal)

        for start, end in segments:
            assert start < end
            assert start >= 0

    def test_flops_estimate(self):
        """FLOPs should be reasonable for edge deployment."""
        vad = WaveletVAD()
        flops = vad.estimate_flops_per_frame()
        assert flops > 0
        assert flops < 100000  # Should be very low compute


class TestWaveletInit:
    """Wavelet-initialized kernels tests."""

    def test_kernel_shape(self):
        """Kernel should have correct shape."""
        kernel = wavelet_init_kernel(5, n_output_channels=16, n_input_channels=1)
        assert kernel.shape == (16, 1, 5)

    def test_kernel_normalized(self):
        """Kernel should be approximately unit norm."""
        kernel = wavelet_init_kernel(5, n_output_channels=1, n_input_channels=1)
        norm = np.linalg.norm(kernel[0, 0])
        assert abs(norm - 1.0) < 0.01

    def test_multistream_kernels(self):
        """Multi-stream should create kernels for all sizes."""
        kernels = create_multistream_kernels([3, 5, 7], n_channels=8)
        assert 3 in kernels
        assert 5 in kernels
        assert 7 in kernels
        assert kernels[3].shape == (16, 1, 3)  # 8 lp + 8 hp
        assert kernels[5].shape == (16, 1, 5)
        assert kernels[7].shape == (16, 1, 7)

    def test_lowpass_highpass_different(self):
        """Lowpass and highpass kernels should be different."""
        lp = wavelet_init_kernel(5, use_highpass=False)
        hp = wavelet_init_kernel(5, use_highpass=True)
        assert not np.allclose(lp, hp)


class TestWaveletAnalyzer:
    """Wavelet analysis tests."""

    def test_scalogram_keys(self):
        """Scalogram should return expected keys."""
        signal = np.random.randn(16000).astype(np.float32)
        analyzer = WaveletAnalyzer(levels=3)
        result = analyzer.compute_scalogram(signal)

        assert "coefficients" in result
        assert "energies" in result
        assert "band_names" in result
        assert "freq_ranges" in result

    def test_compare_stages(self):
        """Stage comparison should return energy dicts."""
        rng = np.random.default_rng(42)
        signals = {
            "noisy": rng.standard_normal(16000).astype(np.float32),
            "clean": np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32),
        }

        analyzer = WaveletAnalyzer(levels=3)
        result = analyzer.compare_pipeline_stages(signals)

        assert "noisy" in result
        assert "clean" in result
        assert "cA_4" in result["noisy"] or "cA_3" in result["noisy"]
