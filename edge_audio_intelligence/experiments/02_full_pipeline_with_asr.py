"""
Experiment 02: Full Pipeline with LibriSpeech + Whisper ASR

End-to-end: GCC-PHAT -> Delay-and-Sum -> Spectral Subtraction -> Whisper ASR

Uses real speech from LibriSpeech test-clean with ground truth transcriptions.
Runs on 4 corner cases and reports WER alongside all upstream metrics.
"""

import sys
import logging
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from edge_audio_intelligence.testbench.scene import (
    SceneConfig,
    SourceConfig,
    MicArrayConfig,
)
from edge_audio_intelligence.testbench.simulator import RIRSimulator
from edge_audio_intelligence.testbench.evaluator import PipelineEvaluator
from edge_audio_intelligence.modules.ssl import GccPhatSSL, SrpPhatSSL, MusicSSL
from edge_audio_intelligence.modules.beamforming import DelayAndSumBeamformer, MVDRBeamformer
from edge_audio_intelligence.modules.enhancement import (
    SpectralSubtractionEnhancer,
    WaveletEnhancer,
)
from edge_audio_intelligence.modules.asr import WhisperOfflineASR
from edge_audio_intelligence.pipeline.cascade import CascadePipeline
from edge_audio_intelligence.pipeline.runner import ExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
LIBRISPEECH_DIR = project_root / "data" / "librispeech" / "LibriSpeech" / "test-clean"
RESULTS_DIR = str(project_root / "edge_audio_intelligence" / "results")


def load_librispeech_utterances(n_utterances: int = 5, seed: int = 42):
    """Load random LibriSpeech utterances with transcriptions.

    Returns list of (flac_path, transcription) tuples.
    """
    trans_files = sorted(LIBRISPEECH_DIR.rglob("*.trans.txt"))
    if not trans_files:
        raise FileNotFoundError(
            f"No transcription files found in {LIBRISPEECH_DIR}. "
            "Download LibriSpeech test-clean first."
        )

    # Collect all utterances
    all_utterances = []
    for tf in trans_files:
        with open(tf) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    flac_path = tf.parent / f"{utt_id}.flac"
                    if flac_path.exists():
                        all_utterances.append((str(flac_path), text))

    # Sample randomly
    rng = random.Random(seed)
    selected = rng.sample(all_utterances, min(n_utterances, len(all_utterances)))

    logger.info(f"Loaded {len(selected)} LibriSpeech utterances from {len(all_utterances)} total")
    for path, text in selected:
        logger.info(f"  {Path(path).stem}: {text[:60]}...")

    return selected


def create_speech_config(
    audio_path: str,
    transcription: str,
    snr_db: float = 15.0,
    rt60: float = 0.3,
) -> SceneConfig:
    """Create a scene config with a real speech source."""
    return SceneConfig(
        room_dim=[6.0, 5.0, 3.0],
        rt60=rt60,
        snr_db=snr_db,
        sources=[
            SourceConfig(
                position=[2.0, 3.5, 1.5],
                signal_type="speech",
                audio_path=audio_path,
                amplitude=1.0,
                label="S0",
                transcription=transcription,
            ),
        ],
        mic_array=MicArrayConfig.linear_array(
            n_mics=4, spacing=0.05, center=[3.0, 2.5], height=1.2
        ),
        duration_s=5.0,  # Longer for speech
        fs=16000,
        noise_type="white",
        seed=42,
    )


def run_full_pipeline():
    """Run full pipeline: SSL -> BF -> Enhancement -> Whisper ASR."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 02: Full Pipeline with LibriSpeech + Whisper")
    logger.info("=" * 60)

    # Load utterances
    utterances = load_librispeech_utterances(n_utterances=4, seed=42)

    # Build pipeline
    pipeline = CascadePipeline(name="full_v1")
    pipeline.add_module(GccPhatSSL(n_fft=1024))
    pipeline.add_module(DelayAndSumBeamformer(use_fractional_delay=True))
    pipeline.add_module(SpectralSubtractionEnhancer(alpha=2.0, beta=0.01))
    pipeline.add_module(WhisperOfflineASR(model_size="base", device="cpu"))

    logger.info(f"Pipeline: {pipeline}")

    # Run on corner cases with each utterance
    simulator = RIRSimulator(seed=42)
    evaluator = PipelineEvaluator(results_dir=RESULTS_DIR)

    corners = [
        (30.0, 0.0, "LL"),  # Easy
        (5.0,  0.0, "HL"),  # Noisy
        (30.0, 0.6, "LM"),  # Moderate reverb
        (5.0,  0.6, "HM"),  # Noisy + reverb
    ]

    all_results = []
    for i, ((audio_path, transcription), (snr, rt60, label)) in enumerate(
        zip(utterances, corners)
    ):
        logger.info(f"\n--- Scene {i+1}/{len(utterances)}: {label} (SNR={snr}dB, RT60={rt60}s) ---")
        logger.info(f"Utterance: {transcription[:80]}...")

        config = create_speech_config(audio_path, transcription, snr_db=snr, rt60=rt60)
        scene = simulator.generate_scene(config)

        # Run pipeline
        data = scene.to_pipeline_dict()
        data = pipeline.run(data)

        # Evaluate
        result = evaluator.run_full_evaluation(data, scene, scene_id=f"full_{label}")
        all_results.append(result)

        # Print ASR result
        if "transcriptions" in data:
            logger.info(f"  Reference: {transcription[:80]}")
            logger.info(f"  Hypothesis: {data['transcriptions'][0][:80]}")
            if result.end_to_end_wer is not None:
                logger.info(f"  WER: {result.end_to_end_wer:.3f}")

    # Save and display
    evaluator.save_results("02_full_pipeline_results.json")
    summary = evaluator.print_summary_table()

    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE RESULTS")
    logger.info("=" * 60)
    print(summary)

    return all_results


if __name__ == "__main__":
    run_full_pipeline()
