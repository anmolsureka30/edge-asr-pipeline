"""SceneService: Wraps RIRSimulator for the API layer."""

import logging

from edge_audio_intelligence.backend.models.requests import SceneRequest
from edge_audio_intelligence.testbench.scene import SceneConfig, SourceConfig, MicArrayConfig
from edge_audio_intelligence.testbench.simulator import RIRSimulator

logger = logging.getLogger(__name__)


def build_scene_config(req: SceneRequest) -> SceneConfig:
    """Convert a SceneRequest (Pydantic) to a SceneConfig (dataclass)."""
    sources = []
    for src in req.sources:
        sources.append(SourceConfig(
            position=src.position,
            signal_type=src.signal_type,
            audio_path=src.audio_path,
            frequency=src.frequency,
            amplitude=src.amplitude,
            label=src.label,
            transcription=src.transcription,
            onset_s=src.onset_s,
            offset_s=src.offset_s,
        ))

    mic = req.mic_array
    if mic.array_type == "linear":
        mic_cfg = MicArrayConfig.linear_array(
            n_mics=mic.n_mics, spacing=mic.spacing,
            center=mic.center, height=mic.height,
        )
    elif mic.array_type == "circular":
        mic_cfg = MicArrayConfig.circular_array(
            n_mics=mic.n_mics, radius=mic.spacing,
            center=mic.center, height=mic.height,
        )
    elif mic.array_type == "phone_2mic":
        mic_cfg = MicArrayConfig.phone_2mic(center=mic.center, height=mic.height)
    elif mic.array_type == "smart_speaker_4mic":
        mic_cfg = MicArrayConfig.smart_speaker_4mic(center=mic.center, height=mic.height)
    else:
        mic_cfg = MicArrayConfig.linear_array(
            n_mics=mic.n_mics, spacing=mic.spacing,
            center=mic.center, height=mic.height,
        )

    return SceneConfig(
        room_dim=req.room_dim,
        rt60=req.rt60,
        snr_db=req.snr_db,
        sources=sources,
        mic_array=mic_cfg,
        duration_s=req.duration_s,
        fs=req.fs,
        noise_type=req.noise_type,
        seed=req.seed,
    )


def generate_scene(req: SceneRequest):
    """Generate an AcousticScene from a SceneRequest.

    Returns the AcousticScene object.
    """
    config = build_scene_config(req)
    simulator = RIRSimulator(seed=config.seed)
    scene = simulator.generate_scene(config)
    logger.info(f"Generated scene: {scene}")
    return scene
