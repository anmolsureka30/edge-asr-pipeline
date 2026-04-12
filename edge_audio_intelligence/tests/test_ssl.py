"""Tests for SSL modules."""

import numpy as np
import pytest

from edge_audio_intelligence.modules.ssl import GccPhatSSL, SrpPhatSSL, MusicSSL


class TestGccPhat:
    """GCC-PHAT tests with known-answer cases."""

    def test_process_returns_required_keys(self, pipeline_data):
        """SSL process() must add all required output keys."""
        ssl = GccPhatSSL()
        result = ssl.process(pipeline_data)

        assert "estimated_doa" in result
        assert "doa_confidence" in result
        assert "n_detected_sources" in result
        assert "ssl_method" in result
        assert "ssl_latency_ms" in result

    def test_doa_is_finite(self, pipeline_data):
        """Estimated DOA must be a valid number."""
        ssl = GccPhatSSL()
        result = ssl.process(pipeline_data)
        doa = result["estimated_doa"]
        assert np.isfinite(doa)
        assert 0 <= doa < 360

    def test_known_delay_detection(self, synthetic_multichannel):
        """GCC-PHAT should detect the correct TDOA for a synthetic signal."""
        ssl = GccPhatSSL(n_fft=2048)
        data = {
            "multichannel_audio": synthetic_multichannel["multichannel_audio"],
            "mic_positions": synthetic_multichannel["mic_positions"],
            "sample_rate": synthetic_multichannel["sample_rate"],
        }
        result = ssl.process(data)

        # The TDOA should be close to the true value
        true_tdoa = synthetic_multichannel["true_tdoa"]
        est_tdoa = result.get("tdoa_seconds", 0)
        # Allow some tolerance (1 sample at 16kHz = 62.5us)
        assert abs(est_tdoa - true_tdoa) < 2.0 / 16000

    def test_get_config(self):
        ssl = GccPhatSSL(n_fft=2048)
        config = ssl.get_config()
        assert config["method"] == "GCC-PHAT"
        assert config["n_fft"] == 2048


class TestSrpPhat:
    """SRP-PHAT tests."""

    def test_process_returns_required_keys(self, pipeline_data):
        ssl = SrpPhatSSL()
        result = ssl.process(pipeline_data)

        assert "estimated_doa" in result
        assert "ssl_method" in result
        assert result["ssl_method"] == "SRP-PHAT"

    def test_doa_is_finite(self, pipeline_data):
        ssl = SrpPhatSSL()
        result = ssl.process(pipeline_data)
        doa = result["estimated_doa"]
        assert np.isfinite(doa)
        assert 0 <= doa < 360


class TestMusic:
    """MUSIC tests."""

    def test_process_returns_required_keys(self, pipeline_data):
        ssl = MusicSSL(n_sources=1)
        result = ssl.process(pipeline_data)

        assert "estimated_doa" in result
        assert "ssl_method" in result
        assert result["ssl_method"] == "MUSIC"

    def test_doa_is_finite(self, pipeline_data):
        ssl = MusicSSL(n_sources=1)
        result = ssl.process(pipeline_data)
        doa = result["estimated_doa"]
        assert np.isfinite(doa)
