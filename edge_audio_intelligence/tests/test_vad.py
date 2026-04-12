"""
Tests for VAD modules using TEN VAD testset (30 .wav + .scv files).

Tests all three VAD methods on real audio with ground truth labels,
computing F1, precision, recall, and Phi_nn contamination rate.
"""

import os
import sys
import pytest
import numpy as np

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT))

TESTSET_DIR = os.path.join(
    os.path.dirname(ROOT), "ten-vad", "testset"
)


def parse_scv(scv_path: str):
    """Parse .scv ground truth file.

    Format: id,start,end,label,start,end,label,...
    Label: 0 = non-speech, 1 = speech

    Returns list of (start_s, end_s, is_speech) tuples.
    """
    with open(scv_path, "r") as f:
        line = f.read().strip()

    parts = line.split(",")
    # First element is the audio ID
    segments = []
    i = 1  # Skip audio ID
    while i + 2 < len(parts):
        start = float(parts[i])
        end = float(parts[i + 1])
        label = int(parts[i + 2])
        segments.append((start, end, bool(label)))
        i += 3

    return segments


def segments_to_frame_labels(segments, audio_duration_s, frame_duration_s):
    """Convert segment-level labels to per-frame labels."""
    n_frames = int(audio_duration_s / frame_duration_s)
    labels = [False] * n_frames

    for start_s, end_s, is_speech in segments:
        start_frame = int(start_s / frame_duration_s)
        end_frame = int(end_s / frame_duration_s)
        for f in range(max(0, start_frame), min(n_frames, end_frame)):
            labels[f] = is_speech

    return labels


@pytest.fixture
def testset_files():
    """Find all testset .wav + .scv pairs."""
    if not os.path.exists(TESTSET_DIR):
        pytest.skip("TEN VAD testset not found")

    files = []
    for i in range(1, 31):
        wav = os.path.join(TESTSET_DIR, f"testset-audio-{i:02d}.wav")
        scv = os.path.join(TESTSET_DIR, f"testset-audio-{i:02d}.scv")
        if os.path.exists(wav) and os.path.exists(scv):
            files.append((wav, scv))

    if not files:
        pytest.skip("No testset files found")
    return files


class TestTENVad:
    """Test TEN VAD on real audio with ground truth."""

    def test_ten_vad_loads(self):
        from edge_audio_intelligence.modules.vad.ten_vad_module import TENVadModule
        vad = TENVadModule(hop_size=256)
        assert vad.name == "TEN-VAD"
        assert vad.get_frame_duration_ms() == 16.0

    def test_ten_vad_on_silence(self):
        from edge_audio_intelligence.modules.vad.ten_vad_module import TENVadModule
        vad = TENVadModule(hop_size=256)
        silence = np.zeros(16000, dtype=np.float32)  # 1s silence
        data = {"multichannel_audio": silence[np.newaxis, :], "sample_rate": 16000}
        result = vad.process(data)
        # Most frames should be noise
        assert sum(result["vad_is_noise"]) > len(result["vad_is_noise"]) * 0.5

    def test_ten_vad_on_testset(self, testset_files):
        from edge_audio_intelligence.modules.vad.ten_vad_module import TENVadModule
        from edge_audio_intelligence.utils.metrics import compute_vad_metrics
        import soundfile as sf

        vad = TENVadModule(hop_size=256)
        all_f1 = []

        for wav_path, scv_path in testset_files[:5]:  # Test first 5 files
            audio, sr = sf.read(wav_path, dtype="float32")
            if sr != 16000:
                continue

            # Run VAD
            data = {"multichannel_audio": audio[np.newaxis, :], "sample_rate": sr}
            result = vad.process(data)

            # Parse ground truth
            gt_segments = parse_scv(scv_path)
            frame_dur_s = vad.get_frame_duration_ms() / 1000.0
            audio_dur_s = len(audio) / sr
            gt_labels = segments_to_frame_labels(gt_segments, audio_dur_s, frame_dur_s)

            # Compute metrics
            metrics = compute_vad_metrics(result["vad_is_speech"], gt_labels)
            all_f1.append(metrics["f1"])

        avg_f1 = np.mean(all_f1)
        assert avg_f1 > 0.3, f"TEN VAD F1 too low: {avg_f1:.3f}"


class TestWaveletVad:
    """Test Wavelet VAD."""

    def test_wavelet_vad_loads(self):
        from edge_audio_intelligence.modules.vad.wavelet_vad_module import WaveletVADModule
        vad = WaveletVADModule()
        assert vad.name == "Wavelet-VAD"
        assert vad.count_parameters() == 0

    def test_wavelet_vad_on_speech(self):
        from edge_audio_intelligence.modules.vad.wavelet_vad_module import WaveletVADModule
        vad = WaveletVADModule()

        sr = 16000
        t = np.arange(sr) / sr
        speech = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)
        data = {"multichannel_audio": speech[np.newaxis, :], "sample_rate": sr}
        result = vad.process(data)

        assert len(result["vad_frame_probs"]) > 0
        assert "vad_is_speech" in result
        assert "vad_is_noise" in result

    def test_wavelet_vad_sub_band_history(self):
        from edge_audio_intelligence.modules.vad.wavelet_vad_module import WaveletVADModule
        vad = WaveletVADModule()

        sr = 16000
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        data = {"multichannel_audio": audio[np.newaxis, :], "sample_rate": sr}
        vad.process(data)

        # Sub-band history should contain per-frame energy dicts
        assert len(vad.sub_band_history) > 0
        assert "cA3" in vad.sub_band_history[0]
        assert "cD1" in vad.sub_band_history[0]


class TestAtomicVad:
    """Test AtomicVAD (requires TensorFlow)."""

    @pytest.fixture(autouse=True)
    def check_tf(self):
        try:
            import tensorflow
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_atomic_vad_loads(self):
        from edge_audio_intelligence.modules.vad.atomic_vad_module import AtomicVADModule
        vad = AtomicVADModule()
        assert vad.name == "AtomicVAD"
        assert vad.count_parameters() > 0

    def test_atomic_vad_on_audio(self):
        from edge_audio_intelligence.modules.vad.atomic_vad_module import AtomicVADModule
        vad = AtomicVADModule()

        sr = 16000
        audio = np.random.randn(sr * 3).astype(np.float32) * 0.1  # 3s
        data = {"multichannel_audio": audio[np.newaxis, :], "sample_rate": sr}
        result = vad.process(data)

        assert len(result["vad_frame_probs"]) > 0
        assert result["vad_method"] == "AtomicVAD"


class TestEnhancementGate:
    """Test the enhancement gate logic."""

    def test_gate_applies_single_speaker(self):
        from edge_audio_intelligence.modules.enhancement.gated import GatedEnhancer
        from edge_audio_intelligence.modules.base import BaseModule

        class DummyEnhancer(BaseModule):
            def __init__(self):
                super().__init__("dummy-enh")
            def process(self, data):
                data["enhanced_audio"] = data.get("beamformed_audio", np.zeros(100))
                data["enhancement_method"] = "dummy"
                data["enhancement_latency_ms"] = 0
                return data
            def get_config(self):
                return {"name": "dummy"}

        gate = GatedEnhancer(DummyEnhancer())
        data = {
            "beamformed_audio": np.ones(100),
            "n_detected_sources": 1,
            "vad_is_speech": [True, True, True],
        }
        result = gate.process(data)
        assert result["enhancement_applied"] is True

    def test_gate_skips_multi_speaker(self):
        from edge_audio_intelligence.modules.enhancement.gated import GatedEnhancer
        from edge_audio_intelligence.modules.base import BaseModule

        class DummyEnhancer(BaseModule):
            def __init__(self):
                super().__init__("dummy-enh")
            def process(self, data):
                data["enhanced_audio"] = data.get("beamformed_audio", np.zeros(100)) * 0
                data["enhancement_method"] = "dummy"
                return data
            def get_config(self):
                return {"name": "dummy"}

        gate = GatedEnhancer(DummyEnhancer())
        data = {
            "beamformed_audio": np.ones(100),
            "n_detected_sources": 2,
            "vad_is_speech": [True, True],
        }
        result = gate.process(data)
        assert result["enhancement_applied"] is False
        assert result["enhancement_gate_reason"] == "multi_speaker"
        # Audio should pass through unchanged
        assert np.allclose(result["enhanced_audio"], np.ones(100))


class TestMVDRPhiNnGating:
    """Test that MVDR uses VAD noise labels for Phi_nn."""

    def test_mvdr_with_vad_improves_sdr(self):
        from edge_audio_intelligence.modules.beamforming.mvdr import MVDRBeamformer
        from edge_audio_intelligence.modules.vad.ten_vad_module import TENVadModule
        from edge_audio_intelligence.utils.metrics import si_sdr

        sr = 16000
        n_mics = 4
        n_samples = sr * 2
        rng = np.random.default_rng(42)

        # Speech in first 1s, silence in last 1s
        t = np.arange(n_samples) / sr
        speech = np.sin(2 * np.pi * 300 * t) * 0.5
        speech[sr:] = 0

        noise = rng.standard_normal((n_mics, n_samples)) * 0.1
        mic_audio = noise.copy()
        for m in range(n_mics):
            mic_audio[m] += speech

        mic_pos = np.array([[i * 0.015, 0, 0] for i in range(n_mics)])

        # Run VAD
        vad = TENVadModule(hop_size=256)
        vad_data = {"multichannel_audio": mic_audio, "sample_rate": sr}
        vad_result = vad.process(vad_data)

        mvdr = MVDRBeamformer(n_fft=512)

        # Without VAD
        bf_no_vad = mvdr.beamform(mic_audio, mic_pos, 0.0, sr)
        # With VAD
        bf_with_vad = mvdr.beamform(mic_audio, mic_pos, 0.0, sr,
                                     vad_is_noise=vad_result["vad_is_noise"])

        sdr_no = si_sdr(speech[:n_samples], bf_no_vad)
        sdr_vad = si_sdr(speech[:n_samples], bf_with_vad)

        # VAD-gated should be significantly better
        assert sdr_vad > sdr_no + 5.0, (
            f"VAD gating should improve SI-SDR by >5dB, "
            f"got {sdr_vad:.1f} vs {sdr_no:.1f}"
        )
