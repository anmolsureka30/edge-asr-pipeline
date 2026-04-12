"""Tests for pipeline orchestration."""

import numpy as np
import pytest

from edge_audio_intelligence.modules.ssl import GccPhatSSL
from edge_audio_intelligence.modules.beamforming import DelayAndSumBeamformer
from edge_audio_intelligence.modules.enhancement import SpectralSubtractionEnhancer
from edge_audio_intelligence.pipeline.cascade import CascadePipeline


class TestCascadePipeline:
    """Cascade pipeline integration tests."""

    def test_three_stage_pipeline(self, pipeline_data):
        """SSL -> Beamforming -> Enhancement should run end-to-end."""
        pipeline = CascadePipeline(name="test")
        pipeline.add_module(GccPhatSSL())
        pipeline.add_module(DelayAndSumBeamformer())
        pipeline.add_module(SpectralSubtractionEnhancer())

        result = pipeline.run(pipeline_data)

        # All stage outputs should be present
        assert "estimated_doa" in result
        assert "beamformed_audio" in result
        assert "enhanced_audio" in result

        # Pipeline metadata
        assert "pipeline_total_ms" in result
        assert result["pipeline_total_ms"] > 0

    def test_pipeline_preserves_ground_truth(self, pipeline_data):
        """Pipeline should not modify ground_truth."""
        pipeline = CascadePipeline()
        pipeline.add_module(GccPhatSSL())
        pipeline.add_module(DelayAndSumBeamformer())

        gt_before = pipeline_data["ground_truth"].copy()
        result = pipeline.run(pipeline_data)

        assert result["ground_truth"]["n_sources"] == gt_before["n_sources"]

    def test_get_config(self):
        """Pipeline config should list all modules."""
        pipeline = CascadePipeline(name="test_config")
        pipeline.add_module(GccPhatSSL())
        pipeline.add_module(DelayAndSumBeamformer())

        config = pipeline.get_config()
        assert config["pipeline_type"] == "cascade"
        assert len(config["modules"]) == 2
        assert config["modules"][0]["name"] == "GCC-PHAT"

    def test_chaining(self):
        """add_module should support method chaining."""
        pipeline = (
            CascadePipeline()
            .add_module(GccPhatSSL())
            .add_module(DelayAndSumBeamformer())
        )
        assert len(pipeline.modules) == 2


class TestWaveletPipeline:
    """Pipeline with wavelet enhancement."""

    def test_wavelet_enhancement_pipeline(self, pipeline_data):
        """SSL -> DS -> Wavelet Enhancement should run."""
        from edge_audio_intelligence.modules.enhancement import WaveletEnhancer

        pipeline = CascadePipeline(name="wavelet_test")
        pipeline.add_module(GccPhatSSL())
        pipeline.add_module(DelayAndSumBeamformer())
        pipeline.add_module(WaveletEnhancer())

        result = pipeline.run(pipeline_data)

        assert "enhanced_audio" in result
        assert result["enhancement_method"] == "Wavelet-Enhancement"
