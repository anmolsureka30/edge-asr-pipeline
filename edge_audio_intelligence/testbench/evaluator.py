"""
PipelineEvaluator: Computes all metrics at every probe point in the pipeline.

Evaluates each module's output against ground truth and logs results.
See DATASETS_AND_METRICS.md for metric definitions.
See CLAUDE.md Section 4.1 for the experiment workflow.

Probe points (measured after each module):
    1. SSL: angular error, RMSAE, position error
    2. Beamforming: SI-SDR improvement, output SNR
    3. Enhancement: PESQ, STOI, SI-SDR
    4. Separation: SI-SDRi, SDR, SIR, SAR
    5. ASR: WER, CER
    6. Diarization: DER, JER
    7. End-to-end: WER (primary), latency, RTF
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.metrics import (
    angular_error,
    rmsae,
    position_error,
    pesq_score,
    stoi_score,
    si_sdr,
    word_error_rate,
    real_time_factor,
)

logger = logging.getLogger(__name__)


@dataclass
class ModuleResult:
    """Metrics from a single module evaluation."""
    module_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete evaluation results for one scene through the pipeline."""
    scene_id: str
    snr_db: float
    rt60: float
    n_sources: int
    module_results: List[ModuleResult] = field(default_factory=list)
    end_to_end_wer: Optional[float] = None
    total_latency_ms: float = 0.0
    total_rtf: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Scene {self.scene_id}: SNR={self.snr_db}dB, RT60={self.rt60}s, "
            f"sources={self.n_sources}"
        ]
        for mr in self.module_results:
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in mr.metrics.items())
            lines.append(f"  {mr.module_name}: {metrics_str} ({mr.latency_ms:.1f}ms)")
        if self.end_to_end_wer is not None:
            lines.append(f"  End-to-end WER: {self.end_to_end_wer:.3f}")
        lines.append(f"  Total latency: {self.total_latency_ms:.1f}ms, RTF: {self.total_rtf:.3f}")
        return "\n".join(lines)


class PipelineEvaluator:
    """Evaluates pipeline outputs against ground truth at every stage.

    Usage:
        evaluator = PipelineEvaluator(results_dir="results/")
        scene = simulator.generate_scene(config)
        data = scene.to_pipeline_dict()

        data = ssl_module.process(data)
        evaluator.evaluate_ssl(data, scene)

        data = beamformer.process(data)
        evaluator.evaluate_beamforming(data, scene)
        ...
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "tables").mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)

        self.all_results: List[PipelineResult] = []

    def evaluate_ssl(
        self,
        data: Dict[str, Any],
        scene: Any,
    ) -> ModuleResult:
        """Evaluate SSL module output.

        Metrics: angular_error, rmsae, position_error (if available).

        Args:
            data: Pipeline data dict with 'estimated_doa' key.
            scene: AcousticScene with ground truth DOAs.

        Returns:
            ModuleResult with SSL metrics.
        """
        metrics = {}

        if "estimated_doa" in data and len(scene.true_doas) > 0:
            est_doa = data["estimated_doa"]

            # Handle single-frame or multi-frame DOA
            if isinstance(est_doa, np.ndarray) and est_doa.ndim >= 2:
                # Multi-frame: [n_frames, n_sources, 2] or [n_frames, n_sources]
                # Use mean across frames for single metric
                est_azimuth = float(np.mean(est_doa[..., 0])) if est_doa.shape[-1] >= 2 else float(np.mean(est_doa))
            elif isinstance(est_doa, (float, int)):
                est_azimuth = float(est_doa)
            else:
                est_azimuth = float(np.mean(est_doa))

            true_azimuth = scene.true_doas[0]  # First source
            metrics["angular_error_deg"] = angular_error(
                np.array([est_azimuth]), np.array([true_azimuth])
            )

        latency = data.get("ssl_latency_ms", 0.0)

        result = ModuleResult(
            module_name=data.get("ssl_method", "ssl"),
            metrics=metrics,
            latency_ms=latency,
        )
        logger.info(f"SSL eval: {metrics}")
        return result

    def evaluate_beamforming(
        self,
        data: Dict[str, Any],
        scene: Any,
    ) -> ModuleResult:
        """Evaluate beamforming output.

        Metrics: si_sdr (vs reverberant clean at ref mic), si_sdr_improvement.
        Uses reverberant_sources[0, 0] as reference when available (the target
        source at the first microphone, after room convolution). Falls back to
        dry clean_sources[0] if reverberant ground truth is not available.
        Ref: ACOUSTIC_LAB.md Section 2.4
        """
        metrics = {}

        if "beamformed_audio" in data and len(scene.clean_sources) > 0:
            bf_audio = data["beamformed_audio"]
            if bf_audio.ndim > 1:
                bf_audio = bf_audio[0]  # First source/beam

            # Use reverberant source at reference mic as ground truth.
            # This is the target signal AFTER room convolution (same acoustic domain
            # as the beamformed output). Using dry clean_sources would give deeply
            # negative SI-SDR due to propagation delay and room coloration mismatch.
            sr = scene.sample_rate
            if scene.reverberant_sources is not None:
                rev = scene.reverberant_sources
                if isinstance(rev, np.ndarray) and rev.ndim >= 2:
                    ref = rev[0, 0] if rev.ndim == 3 else rev[0]
                elif isinstance(rev, list) and len(rev) > 0:
                    r0 = rev[0]
                    ref = r0[0] if r0.ndim >= 2 else r0
                else:
                    ref = scene.clean_sources[0]
            else:
                ref = scene.clean_sources[0]

            # Determine active region from source timing
            onset = 0
            offset = len(ref)
            if hasattr(scene.config, 'sources') and len(scene.config.sources) > 0:
                src_cfg = scene.config.sources[0]
                onset = int(getattr(src_cfg, 'onset_s', 0.0) * sr)
                off_s = getattr(src_cfg, 'offset_s', -1.0)
                if off_s > 0:
                    offset = int(off_s * sr)

            # Compare only in the active region
            min_len = min(len(bf_audio), len(ref), offset)
            start = min(onset, min_len)
            ref_active = ref[start:min_len]
            bf_active = bf_audio[start:min_len]

            if len(ref_active) > sr // 2:
                metrics["si_sdr_db"] = si_sdr(ref_active, bf_active)

                # SI-SDR improvement vs raw noisy mic 0
                noisy_ref = scene.multichannel_audio[0][start:min_len]
                noisy_si_sdr = si_sdr(ref_active, noisy_ref)
                metrics["si_sdr_improvement_db"] = metrics["si_sdr_db"] - noisy_si_sdr

        latency = data.get("bf_latency_ms", 0.0)
        return ModuleResult(
            module_name=data.get("bf_method", "beamforming"),
            metrics=metrics,
            latency_ms=latency,
        )

    def evaluate_enhancement(
        self,
        data: Dict[str, Any],
        scene: Any,
    ) -> ModuleResult:
        """Evaluate enhancement output.

        Metrics: PESQ, STOI, SI-SDR.
        """
        metrics = {}

        if "enhanced_audio" in data and len(scene.clean_sources) > 0:
            enhanced = data["enhanced_audio"]
            if enhanced.ndim > 1:
                enhanced = enhanced[0]

            sr = data.get("sample_rate", 16000)

            # Use reverberant reference for SI-SDR (same domain as enhanced output)
            # Use clean source for PESQ/STOI (these metrics are designed for clean ref)
            if scene.reverberant_sources is not None:
                rev = scene.reverberant_sources
                if isinstance(rev, np.ndarray) and rev.ndim >= 2:
                    ref_rev = rev[0, 0] if rev.ndim == 3 else rev[0]
                elif isinstance(rev, list) and len(rev) > 0:
                    r0 = rev[0]
                    ref_rev = r0[0] if r0.ndim >= 2 else r0
                else:
                    ref_rev = scene.clean_sources[0]
            else:
                ref_rev = scene.clean_sources[0]

            ref_clean = scene.clean_sources[0]
            min_len = min(len(enhanced), len(ref_clean), len(ref_rev))

            metrics["pesq"] = pesq_score(ref_clean[:min_len], enhanced[:min_len], sr=sr)
            metrics["stoi"] = stoi_score(ref_clean[:min_len], enhanced[:min_len], sr=sr)
            metrics["si_sdr_db"] = si_sdr(ref_rev[:min_len], enhanced[:min_len])

        latency = data.get("enhancement_latency_ms", 0.0)
        return ModuleResult(
            module_name=data.get("enhancement_method", "enhancement"),
            metrics=metrics,
            latency_ms=latency,
        )

    def evaluate_asr(
        self,
        data: Dict[str, Any],
        scene: Any,
    ) -> ModuleResult:
        """Evaluate ASR output.

        Metrics: WER per source, average WER.
        """
        metrics = {}

        if "transcriptions" in data and len(scene.transcriptions) > 0:
            hyp_transcriptions = data["transcriptions"]
            ref_transcriptions = scene.transcriptions

            wers = []
            for i, (ref, hyp) in enumerate(
                zip(ref_transcriptions, hyp_transcriptions)
            ):
                if ref:  # Skip empty references
                    wer = word_error_rate(ref, hyp)
                    metrics[f"wer_source_{i}"] = wer
                    wers.append(wer)

            if wers:
                metrics["wer_avg"] = float(np.mean(wers))

        latency = data.get("asr_latency_ms", 0.0)
        return ModuleResult(
            module_name=data.get("asr_method", "asr"),
            metrics=metrics,
            latency_ms=latency,
        )

    def run_full_evaluation(
        self,
        data: Dict[str, Any],
        scene: Any,
        scene_id: str = "scene_0",
    ) -> PipelineResult:
        """Run evaluation at all available probe points.

        Checks which module outputs are present in the data dict
        and evaluates each one.

        Args:
            data: Pipeline data dict (after all modules have processed).
            scene: AcousticScene with ground truth.
            scene_id: Identifier for this scene.

        Returns:
            PipelineResult with all module metrics.
        """
        result = PipelineResult(
            scene_id=scene_id,
            snr_db=scene.config.snr_db,
            rt60=scene.config.rt60,
            n_sources=scene.n_sources,
        )

        # Evaluate each stage if output keys are present
        if "estimated_doa" in data:
            result.module_results.append(self.evaluate_ssl(data, scene))

        if "beamformed_audio" in data:
            result.module_results.append(self.evaluate_beamforming(data, scene))

        if "enhanced_audio" in data:
            result.module_results.append(self.evaluate_enhancement(data, scene))

        if "transcriptions" in data and data.get("asr_method"):
            result.module_results.append(self.evaluate_asr(data, scene))
            if result.module_results[-1].metrics.get("wer_avg") is not None:
                result.end_to_end_wer = result.module_results[-1].metrics["wer_avg"]

        # Total latency
        result.total_latency_ms = sum(mr.latency_ms for mr in result.module_results)

        audio_dur = scene.multichannel_audio.shape[-1] / scene.sample_rate
        result.total_rtf = real_time_factor(
            result.total_latency_ms / 1000.0, audio_dur
        )

        self.all_results.append(result)
        logger.info(f"Evaluation complete:\n{result.summary()}")
        return result

    def save_results(self, filename: str = "pipeline_results.json") -> Path:
        """Save all results to JSON file.

        Args:
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        path = self.results_dir / "tables" / filename
        results_data = [r.to_dict() for r in self.all_results]

        with open(path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"Saved {len(self.all_results)} results to {path}")
        return path

    def print_summary_table(self) -> str:
        """Print a summary table of all results.

        Returns:
            Formatted table string.
        """
        if not self.all_results:
            return "No results to display."

        lines = [
            f"{'Scene':<12} {'SNR':>5} {'RT60':>5} "
            f"{'SSL err':>8} {'BF SDR':>8} {'PESQ':>6} {'WER':>6} "
            f"{'Latency':>8} {'RTF':>6}"
        ]
        lines.append("-" * 75)

        for r in self.all_results:
            ssl_err = "--"
            bf_sdr = "--"
            pesq_val = "--"
            wer = "--"

            for mr in r.module_results:
                if "angular_error_deg" in mr.metrics:
                    ssl_err = f"{mr.metrics['angular_error_deg']:.1f}"
                if "si_sdr_db" in mr.metrics and "si_sdr_improvement_db" in mr.metrics:
                    bf_sdr = f"{mr.metrics['si_sdr_db']:.1f}"
                if "pesq" in mr.metrics:
                    pesq_val = f"{mr.metrics['pesq']:.2f}"
                if "wer_avg" in mr.metrics:
                    wer = f"{mr.metrics['wer_avg']:.3f}"

            lines.append(
                f"{r.scene_id:<12} {r.snr_db:>5.0f} {r.rt60:>5.1f} "
                f"{ssl_err:>8} {bf_sdr:>8} {pesq_val:>6} {wer:>6} "
                f"{r.total_latency_ms:>7.1f}ms {r.total_rtf:>6.3f}"
            )

        table = "\n".join(lines)
        logger.info(f"Summary:\n{table}")
        return table
