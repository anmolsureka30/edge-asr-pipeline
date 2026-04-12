"""PipelineService: Extracted pipeline orchestration from run_callbacks.py.

Builds modules from string identifiers, runs VAD pre-pipeline,
executes the cascade pipeline, and evaluates results.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def build_module(stage: str, value: str):
    """Instantiate a pipeline module from stage name + dropdown value.

    Extracted from dashboard/callbacks/run_callbacks.py:_build_module.
    """
    if value == "none" or not value:
        return None

    if stage == "ssl":
        if value == "gcc_phat":
            from edge_audio_intelligence.modules.ssl.gcc_phat import GccPhatSSL
            return GccPhatSSL(n_fft=1024)
        elif value == "srp_phat":
            from edge_audio_intelligence.modules.ssl.srp_phat import SrpPhatSSL
            return SrpPhatSSL(n_fft=1024, grid_resolution=360)

    elif stage == "beamforming":
        if value == "delay_and_sum":
            from edge_audio_intelligence.modules.beamforming.delay_and_sum import DelayAndSumBeamformer
            return DelayAndSumBeamformer(use_fractional_delay=True)
        elif value == "mvdr":
            from edge_audio_intelligence.modules.beamforming.mvdr import MVDRBeamformer
            return MVDRBeamformer()

    elif stage == "enhancement":
        if value == "spectral_subtraction":
            from edge_audio_intelligence.modules.enhancement.spectral_subtraction import SpectralSubtractionEnhancer
            return SpectralSubtractionEnhancer(alpha=2.0, beta=0.01)
        elif value == "wavelet_enhancement":
            from edge_audio_intelligence.modules.enhancement.wavelet_enhancement import WaveletEnhancer
            return WaveletEnhancer(wavelet="bior2.2", levels=3, threshold_scale=1.0)

    elif stage == "asr":
        if value == "whisper_tiny":
            from edge_audio_intelligence.modules.asr.whisper_offline import WhisperOfflineASR
            return WhisperOfflineASR(model_size="tiny")
        elif value == "whisper_base":
            from edge_audio_intelligence.modules.asr.whisper_offline import WhisperOfflineASR
            return WhisperOfflineASR(model_size="base")
        elif value == "whisper_small":
            from edge_audio_intelligence.modules.asr.whisper_offline import WhisperOfflineASR
            return WhisperOfflineASR(model_size="small")

    elif stage == "vad":
        if value in ("full_vad", "ten_vad"):
            from edge_audio_intelligence.modules.vad.ten_vad_module import TENVadModule
            return TENVadModule(hop_size=256, speech_threshold=0.5, noise_threshold=0.3)
        elif value == "wavelet_vad":
            from edge_audio_intelligence.modules.vad.wavelet_vad_module import WaveletVADModule
            return WaveletVADModule()
        elif value == "atomic_vad":
            from edge_audio_intelligence.modules.vad.atomic_vad_module import AtomicVADModule
            return AtomicVADModule()

    return None


def run_pipeline(
    scene,
    ssl_val: str,
    bf_val: str,
    enh_val: str,
    asr_val: str,
    vad_val: str,
    enh_gate: bool,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run the full pipeline on a scene.

    Args:
        scene: AcousticScene from simulator.
        ssl_val..vad_val: Module selection strings.
        enh_gate: Whether to use gated enhancement.
        on_progress: Optional callback(stage, progress, message, latency_ms, metrics)
            for WebSocket progress reporting.

    Returns:
        Dict with keys: data, result, module_names
    """
    from edge_audio_intelligence.pipeline.cascade import CascadePipeline
    from edge_audio_intelligence.testbench.evaluator import PipelineEvaluator

    def progress(stage, pct, msg="", latency=0.0, metrics=None):
        if on_progress:
            on_progress(stage, pct, msg, latency, metrics or {})

    data = scene.to_pipeline_dict()
    module_names = {}

    # ── PRE-PIPELINE: VAD on raw mic audio ──
    progress("vad", 0.1, "Running VAD...")

    vad_module = build_module("vad", vad_val) if vad_val and vad_val != "none" else None

    if vad_val == "full_vad":
        # AtomicVAD wake check + TEN VAD precision
        try:
            from edge_audio_intelligence.modules.vad.atomic_vad_module import AtomicVADModule
            atomic = AtomicVADModule()
            t0 = time.perf_counter()
            wake_result = atomic.process(data.copy())
            wake_latency = (time.perf_counter() - t0) * 1000.0

            wake_speech = any(wake_result.get("vad_is_speech", []))
            data["vad_wake_detected"] = wake_speech
            data["vad_wake_latency_ms"] = wake_latency
            data["vad_wake_method"] = "AtomicVAD"
        except Exception as e:
            logger.warning(f"AtomicVAD wake check failed: {e}")
            data["vad_wake_detected"] = True

        # TEN VAD precision
        if vad_module is not None:
            t0 = time.perf_counter()
            vad_result = vad_module.process(data.copy())
            vad_latency = (time.perf_counter() - t0) * 1000.0
            for k in ["vad_frame_probs", "vad_is_speech", "vad_is_noise",
                       "vad_speech_segments", "vad_frame_duration_ms"]:
                data[k] = vad_result[k]
            data["vad_method"] = "AtomicVAD + TEN VAD"
            data["vad_latency_ms"] = data.get("vad_wake_latency_ms", 0) + vad_latency
            module_names["vad"] = "AtomicVAD + TEN VAD"

        vad_metrics = {}
        if "vad_is_speech" in data:
            n_s = sum(data["vad_is_speech"])
            n_t = len(data["vad_is_speech"])
            vad_metrics = {"speech_pct": round(100 * n_s / max(n_t, 1), 1)}
        progress("vad", 0.2, "VAD complete", data.get("vad_latency_ms", 0), vad_metrics)

    elif vad_module is not None:
        t0 = time.perf_counter()
        vad_result = vad_module.process(data.copy())
        vad_latency = (time.perf_counter() - t0) * 1000.0
        for k in ["vad_frame_probs", "vad_is_speech", "vad_is_noise",
                   "vad_speech_segments", "vad_frame_duration_ms"]:
            data[k] = vad_result[k]
        data["vad_method"] = vad_result["vad_method"]
        data["vad_latency_ms"] = vad_latency
        module_names["vad"] = vad_module.name

        n_s = sum(data.get("vad_is_speech", []))
        n_t = len(data.get("vad_is_speech", []))
        progress("vad", 0.2, "VAD complete", vad_latency,
                 {"speech_pct": round(100 * n_s / max(n_t, 1), 1)})

    # ── BUILD PIPELINE ──
    pipeline = CascadePipeline(name="api_run")

    ssl_module = build_module("ssl", ssl_val)
    if ssl_module:
        pipeline.add_module(ssl_module)
        module_names["ssl"] = ssl_module.name

    bf_module = build_module("beamforming", bf_val)
    if bf_module:
        pipeline.add_module(bf_module)
        module_names["beamforming"] = bf_module.name

    enh_module = build_module("enhancement", enh_val)
    if enh_module:
        if enh_gate:
            from edge_audio_intelligence.modules.enhancement.gated import GatedEnhancer
            gated = GatedEnhancer(enh_module)
            pipeline.add_module(gated)
            module_names["enhancement"] = f"{enh_module.name} (gated)"
        else:
            pipeline.add_module(enh_module)
            module_names["enhancement"] = enh_module.name

    asr_module = build_module("asr", asr_val)
    if asr_module:
        pipeline.add_module(asr_module)
        module_names["asr"] = asr_module.name

    # ── RUN PIPELINE ──
    progress("ssl", 0.3, "Running SSL...")
    data = pipeline.run(data)

    # Report per-stage progress
    if "ssl_latency_ms" in data:
        ssl_metrics = {}
        if "estimated_doa" in data and len(scene.true_doas) > 0:
            ssl_err = abs(float(data["estimated_doa"]) - scene.true_doas[0])
            if ssl_err > 180:
                ssl_err = 360 - ssl_err
            ssl_metrics["angular_error"] = round(ssl_err, 1)
        progress("ssl", 0.4, "SSL complete", data.get("ssl_latency_ms", 0), ssl_metrics)

    if "bf_latency_ms" in data:
        progress("beamforming", 0.5, "Beamforming complete", data.get("bf_latency_ms", 0))

    if "enhancement_latency_ms" in data:
        progress("enhancement", 0.7, "Enhancement complete", data.get("enhancement_latency_ms", 0))

    if "asr_latency_ms" in data:
        progress("asr", 0.9, "ASR complete", data.get("asr_latency_ms", 0))

    # ── EVALUATE ──
    from edge_audio_intelligence.backend.config import RESULTS_DIR
    evaluator = PipelineEvaluator(results_dir=str(RESULTS_DIR))
    result = evaluator.run_full_evaluation(data, scene, "api_run")

    progress("complete", 1.0, "Pipeline complete")

    return {
        "data": data,
        "result": result,
        "module_names": module_names,
    }
