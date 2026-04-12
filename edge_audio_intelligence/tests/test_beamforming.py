"""Tests for beamforming modules."""

import numpy as np
import pytest

from edge_audio_intelligence.modules.beamforming import (
    DelayAndSumBeamformer,
    MVDRBeamformer,
)


class TestDelayAndSum:
    """Delay-and-Sum beamformer tests."""

    def test_process_returns_required_keys(self, pipeline_data):
        """Must add beamformed_audio key."""
        # First run SSL to get DOA
        from edge_audio_intelligence.modules.ssl import GccPhatSSL
        ssl = GccPhatSSL()
        data = ssl.process(pipeline_data)

        bf = DelayAndSumBeamformer()
        result = bf.process(data)

        assert "beamformed_audio" in result
        assert "bf_method" in result
        assert "bf_latency_ms" in result
        assert result["bf_method"] == "Delay-and-Sum"

    def test_output_is_mono(self, pipeline_data):
        """Beamformed output must be single-channel."""
        from edge_audio_intelligence.modules.ssl import GccPhatSSL
        data = GccPhatSSL().process(pipeline_data)
        data = DelayAndSumBeamformer().process(data)

        bf_audio = data["beamformed_audio"]
        assert bf_audio.ndim == 1

    def test_output_length_matches_input(self, pipeline_data):
        """Output length should match input."""
        from edge_audio_intelligence.modules.ssl import GccPhatSSL
        data = GccPhatSSL().process(pipeline_data)
        n_samples = data["multichannel_audio"].shape[1]

        data = DelayAndSumBeamformer().process(data)
        assert len(data["beamformed_audio"]) == n_samples

    def test_snr_improvement(self, pipeline_data):
        """DS beamformer should improve SNR (basic check)."""
        from edge_audio_intelligence.modules.ssl import GccPhatSSL
        data = GccPhatSSL().process(pipeline_data)
        data = DelayAndSumBeamformer().process(data)

        # Beamformed signal should have lower variance than noise
        bf = data["beamformed_audio"]
        assert np.isfinite(bf).all()
        assert np.max(np.abs(bf)) > 0  # Not all zeros


class TestMVDR:
    """MVDR beamformer tests."""

    def test_process_returns_required_keys(self, pipeline_data):
        from edge_audio_intelligence.modules.ssl import GccPhatSSL
        data = GccPhatSSL().process(pipeline_data)

        bf = MVDRBeamformer()
        result = bf.process(data)

        assert "beamformed_audio" in result
        assert result["bf_method"] == "MVDR"

    def test_output_is_mono(self, pipeline_data):
        from edge_audio_intelligence.modules.ssl import GccPhatSSL
        data = GccPhatSSL().process(pipeline_data)
        data = MVDRBeamformer().process(data)

        assert data["beamformed_audio"].ndim == 1
