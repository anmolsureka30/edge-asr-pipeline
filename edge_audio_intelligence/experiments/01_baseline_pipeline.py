"""
Experiment 01: Baseline Pipeline Evaluation

V1 baseline: GCC-PHAT -> Delay-and-Sum -> Spectral Subtraction
(No ASR in this run to avoid Whisper dependency; tests SSL+BF+Enhancement)

Runs on 4 corner cases:
    LL: SNR=30dB, RT60=0.0s (easy, sanity check)
    HL: SNR=5dB,  RT60=0.0s (noisy, no reverb)
    LH: SNR=30dB, RT60=1.5s (clean, high reverb)
    HH: SNR=5dB,  RT60=1.5s (hard, stress test)

Measures: SSL angular error, beamforming SI-SDR, enhancement metrics.
"""

import sys
import logging
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
from edge_audio_intelligence.pipeline.cascade import CascadePipeline
from edge_audio_intelligence.pipeline.runner import ExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_base_config() -> SceneConfig:
    """Create the standard single-source test configuration."""
    return SceneConfig(
        room_dim=[6.0, 5.0, 3.0],
        rt60=0.3,  # Will be overridden by corner cases
        snr_db=15.0,  # Will be overridden by corner cases
        sources=[
            SourceConfig(
                position=[2.0, 3.5, 1.5],
                signal_type="sine",
                frequency=1000.0,
                amplitude=1.0,
                label="S0",
            ),
        ],
        mic_array=MicArrayConfig.linear_array(
            n_mics=4, spacing=0.05, center=[3.0, 2.5], height=1.2
        ),
        duration_s=2.0,
        fs=16000,
        noise_type="white",
        seed=42,
    )


def run_baseline_experiment():
    """Run V1 baseline: GCC-PHAT -> DS Beamformer -> Spectral Subtraction."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 01: Baseline Pipeline (V1)")
    logger.info("=" * 60)

    # Build pipeline
    pipeline = CascadePipeline(name="v1_baseline")
    pipeline.add_module(GccPhatSSL(n_fft=1024))
    pipeline.add_module(DelayAndSumBeamformer(use_fractional_delay=True))
    pipeline.add_module(SpectralSubtractionEnhancer(alpha=2.0, beta=0.01))

    logger.info(f"Pipeline: {pipeline}")

    # Create experiment runner
    results_dir = str(project_root / "edge_audio_intelligence" / "results")
    runner = ExperimentRunner(
        name="01_baseline",
        pipeline=pipeline,
        results_dir=results_dir,
        seed=42,
    )

    # Run on corner cases
    base_config = create_base_config()
    results = runner.run_on_config(base_config, mode="corner_cases")

    # Save and display
    runner.save_results()
    summary = runner.print_summary()

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    print(summary)

    return results


def run_ssl_comparison():
    """Compare SSL algorithms on corner cases."""
    logger.info("\n" + "=" * 60)
    logger.info("SSL COMPARISON: GCC-PHAT vs SRP-PHAT vs MUSIC")
    logger.info("=" * 60)

    base_config = create_base_config()
    simulator = RIRSimulator(seed=42)
    scenes = simulator.generate_corner_cases(base_config)

    ssl_methods = {
        "GCC-PHAT": GccPhatSSL(n_fft=1024),
        "SRP-PHAT": SrpPhatSSL(n_fft=1024, grid_resolution=360),
        "MUSIC": MusicSSL(n_fft=1024, n_sources=1, grid_resolution=360),
    }

    results_dir = str(project_root / "edge_audio_intelligence" / "results")

    for method_name, ssl in ssl_methods.items():
        logger.info(f"\n--- {method_name} ---")
        pipeline = CascadePipeline(name=f"ssl_{method_name}")
        pipeline.add_module(ssl)
        pipeline.add_module(DelayAndSumBeamformer())
        pipeline.add_module(SpectralSubtractionEnhancer())

        runner = ExperimentRunner(
            name=f"02_ssl_{method_name}",
            pipeline=pipeline,
            results_dir=results_dir,
        )
        runner.run_on_scenes(scenes)
        runner.save_results()
        print(f"\n{method_name}:")
        print(runner.print_summary())


def run_enhancement_comparison():
    """Compare Spectral Subtraction vs Wavelet Enhancement."""
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCEMENT COMPARISON: Spectral Sub vs Wavelet")
    logger.info("=" * 60)

    base_config = create_base_config()
    simulator = RIRSimulator(seed=42)
    scenes = simulator.generate_corner_cases(base_config)

    enhancers = {
        "Spectral-Subtraction": SpectralSubtractionEnhancer(alpha=2.0),
        "Wavelet-Enhancement": WaveletEnhancer(
            wavelet="bior2.2", levels=3, threshold_scale=1.0
        ),
    }

    results_dir = str(project_root / "edge_audio_intelligence" / "results")

    for enh_name, enhancer in enhancers.items():
        logger.info(f"\n--- {enh_name} ---")
        pipeline = CascadePipeline(name=f"enh_{enh_name}")
        pipeline.add_module(GccPhatSSL())
        pipeline.add_module(DelayAndSumBeamformer())
        pipeline.add_module(enhancer)

        runner = ExperimentRunner(
            name=f"04_enh_{enh_name}",
            pipeline=pipeline,
            results_dir=results_dir,
        )
        runner.run_on_scenes(scenes)
        runner.save_results()
        print(f"\n{enh_name}:")
        print(runner.print_summary())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Pipeline Experiment")
    parser.add_argument(
        "--mode",
        choices=["baseline", "ssl", "enhancement", "all"],
        default="baseline",
        help="Which experiment to run",
    )
    args = parser.parse_args()

    if args.mode == "baseline" or args.mode == "all":
        run_baseline_experiment()

    if args.mode == "ssl" or args.mode == "all":
        run_ssl_comparison()

    if args.mode == "enhancement" or args.mode == "all":
        run_enhancement_comparison()
