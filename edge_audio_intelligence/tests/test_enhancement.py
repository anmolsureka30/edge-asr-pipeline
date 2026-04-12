"""Tests for enhancement modules."""

import numpy as np
import pytest

from edge_audio_intelligence.modules.enhancement import (
    SpectralSubtractionEnhancer,
    WaveletEnhancer,
)


class TestSpectralSubtraction:
    """Spectral subtraction enhancer tests."""

    def test_process_returns_required_keys(self, pipeline_data):
        from edge_audio_intelligence.modules.ssl import GccPhatSSL
        from edge_audio_intelligence.modules.beamforming import DelayAndSumBeamformer

        data = GccPhatSSL().process(pipeline_data)
        data = DelayAndSumBeamformer().process(data)

        enh = SpectralSubtractionEnhancer()
        result = enh.process(data)

        assert "enhanced_audio" in result
        assert "enhancement_method" in result
        assert result["enhancement_method"] == "Spectral-Subtraction"

    def test_output_length_matches(self, pipeline_data):
        from edge_audio_intelligence.modules.ssl import GccPhatSSL
        from edge_audio_intelligence.modules.beamforming import DelayAndSumBeamformer

        data = GccPhatSSL().process(pipeline_data)
        data = DelayAndSumBeamformer().process(data)
        n_samples = len(data["beamformed_audio"])

        data = SpectralSubtractionEnhancer().process(data)
        assert len(data["enhanced_audio"]) == n_samples

    def test_pure_noise_reduction(self):
        """Enhancement should reduce energy of pure noise."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(16000).astype(np.float32)

        enh = SpectralSubtractionEnhancer(alpha=3.0)
        enhanced = enh.enhance(noise, 16000)

        assert np.mean(enhanced ** 2) < np.mean(noise ** 2)


class TestWaveletEnhancer:
    """Wavelet enhancement tests."""

    def test_process_returns_required_keys(self, pipeline_data):
        from edge_audio_intelligence.modules.ssl import GccPhatSSL
        from edge_audio_intelligence.modules.beamforming import DelayAndSumBeamformer

        data = GccPhatSSL().process(pipeline_data)
        data = DelayAndSumBeamformer().process(data)

        enh = WaveletEnhancer()
        result = enh.process(data)

        assert "enhanced_audio" in result
        assert result["enhancement_method"] == "Wavelet-Enhancement"

    def test_subband_energies(self):
        """Should compute energy per sub-band."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(16000).astype(np.float32)

        enh = WaveletEnhancer()
        energies = enh.get_subband_energies(signal)

        assert "cA_3" in energies
        assert "cD_1" in energies
        assert all(e >= 0 for e in energies.values())

    def test_denoising_reduces_noise(self):
        """Wavelet denoising should reduce noise energy."""
        rng = np.random.default_rng(42)
        clean = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        noise = 0.3 * rng.standard_normal(16000).astype(np.float32)
        noisy = clean + noise

        enh = WaveletEnhancer(threshold_scale=1.0)
        enhanced = enh.enhance(noisy, 16000)

        # Enhanced should be closer to clean than noisy is
        noisy_error = np.mean((noisy - clean) ** 2)
        enhanced_error = np.mean((enhanced - clean) ** 2)
        assert enhanced_error < noisy_error
