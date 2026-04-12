"""
Shared test fixtures for the edge audio intelligence pipeline.

Provides standard scene configurations, synthetic signals,
and helper functions for all test modules.
"""

import pytest
import numpy as np

from edge_audio_intelligence.testbench.scene import (
    AcousticScene,
    MicArrayConfig,
    SceneConfig,
    SourceConfig,
)
from edge_audio_intelligence.testbench.simulator import RIRSimulator


@pytest.fixture
def sample_rate():
    return 16000


@pytest.fixture
def duration_s():
    return 2.0


@pytest.fixture
def simple_scene_config():
    """Single source, 4-mic ULA, moderate conditions."""
    return SceneConfig(
        room_dim=[6.0, 5.0, 3.0],
        rt60=0.3,
        snr_db=20.0,
        sources=[
            SourceConfig(
                position=[2.0, 3.5, 1.5],
                signal_type="sine",
                frequency=1000.0,
                amplitude=1.0,
                label="S0",
            )
        ],
        mic_array=MicArrayConfig.linear_array(
            n_mics=4, spacing=0.05, center=[3.0, 2.5], height=1.2
        ),
        duration_s=2.0,
        fs=16000,
        seed=42,
    )


@pytest.fixture
def anechoic_scene_config():
    """Single source, anechoic, high SNR — easiest test case."""
    return SceneConfig(
        room_dim=[6.0, 5.0, 3.0],
        rt60=0.0,
        snr_db=30.0,
        sources=[
            SourceConfig(
                position=[2.0, 4.0, 1.2],
                signal_type="sine",
                frequency=1000.0,
                amplitude=1.0,
                label="S0",
            )
        ],
        mic_array=MicArrayConfig.linear_array(
            n_mics=4, spacing=0.05, center=[3.0, 2.5], height=1.2
        ),
        duration_s=1.0,
        fs=16000,
        seed=42,
    )


@pytest.fixture
def simple_scene(simple_scene_config):
    """Generate a simple scene for testing."""
    sim = RIRSimulator(seed=42)
    return sim.generate_scene(simple_scene_config)


@pytest.fixture
def anechoic_scene(anechoic_scene_config):
    """Generate an anechoic scene for testing."""
    sim = RIRSimulator(seed=42)
    return sim.generate_scene(anechoic_scene_config)


@pytest.fixture
def pipeline_data(simple_scene):
    """Pipeline data dict from simple scene."""
    return simple_scene.to_pipeline_dict()


@pytest.fixture
def synthetic_multichannel():
    """Synthetic multichannel signal with known TDOA.

    Two mics, 5cm apart. Source at 60 degrees.
    Expected TDOA: d*cos(theta)/c = 0.05*cos(60)/343 = 7.29e-5 s
    """
    fs = 16000
    n_samples = 16000  # 1 second
    d = 0.05  # mic spacing
    theta = 60.0  # degrees
    f0 = 1000.0  # Hz

    t = np.arange(n_samples) / fs
    signal = np.sin(2 * np.pi * f0 * t)

    # Compute delay
    tau = d * np.cos(np.radians(theta)) / 343.0
    delay_samples = tau * fs

    # Create delayed version
    sig1 = signal.copy()
    sig2 = np.zeros_like(signal)
    shift = int(np.round(delay_samples))
    if shift >= 0:
        sig2[shift:] = signal[:n_samples - shift]
    else:
        sig2[:n_samples + shift] = signal[-shift:]

    multichannel = np.stack([sig1, sig2])  # [2, n_samples]
    mic_positions = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])

    return {
        "multichannel_audio": multichannel,
        "mic_positions": mic_positions,
        "sample_rate": fs,
        "true_theta": theta,
        "true_tdoa": tau,
    }
