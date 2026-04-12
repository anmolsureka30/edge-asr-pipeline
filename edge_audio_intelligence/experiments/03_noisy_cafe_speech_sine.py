"""
Experiment 03: Noisy Cafe — LibriSpeech speech + sine wave interference.

Tests the full pipeline (VAD → SSL → BF → Enhancement → ASR) in a noisy cafe
scenario where source 1 is real speech and source 2 is a sine wave.

Since the sine wave is NOT speech (no speaker separation needed), this tests:
1. SSL angular accuracy — can GCC-PHAT localize the speech source correctly
   even with a strong tonal interferer from a different direction?
2. Beamforming — can DS/MVDR steer toward speech and null the sine wave?
3. VAD — does VAD correctly classify speech vs sine frames?
4. Enhancement — does it help or hurt when the "noise" is tonal?
5. ASR — what WER does Whisper achieve after beamforming + enhancement?

Test matrix:
  - 3 sine frequencies (500 Hz, 1000 Hz, 3000 Hz) — low/mid/high
  - 2 sine positions (same side as speech, opposite side)
  - 2 SNR levels (15 dB, 5 dB)
  - With and without enhancement
"""

import sys
import os
import time
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from edge_audio_intelligence.testbench.scene import (
    SceneConfig, SourceConfig, MicArrayConfig,
)
from edge_audio_intelligence.testbench.simulator import RIRSimulator
from edge_audio_intelligence.testbench.evaluator import PipelineEvaluator
from edge_audio_intelligence.modules.ssl.gcc_phat import GccPhatSSL
from edge_audio_intelligence.modules.beamforming.delay_and_sum import (
    DelayAndSumBeamformer,
)
from edge_audio_intelligence.modules.beamforming.mvdr import MVDRBeamformer
from edge_audio_intelligence.modules.enhancement.spectral_subtraction import (
    SpectralSubtractionEnhancer,
)
from edge_audio_intelligence.modules.enhancement.wavelet_enhancement import (
    WaveletEnhancer,
)
from edge_audio_intelligence.modules.enhancement.gated import GatedEnhancer
from edge_audio_intelligence.modules.asr.whisper_offline import WhisperOfflineASR
from edge_audio_intelligence.modules.vad.ten_vad_module import TENVadModule
from edge_audio_intelligence.pipeline.cascade import CascadePipeline
from edge_audio_intelligence.utils.metrics import (
    si_sdr, word_error_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("experiment_03")

# ── Constants ──────────────────────────────────────────────────────────

LIBRISPEECH_ROOT = project_root / "data" / "librispeech" / "LibriSpeech" / "test-clean"
RESULTS_DIR = project_root / "results" / "experiment_03"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use 3 different LibriSpeech utterances for variety
UTTERANCES = [
    {
        "path": "61/70970/61-70970-0000.flac",
        "text": "YOUNG FITZOOTH HAD BEEN COMMANDED TO HIS MOTHER'S CHAMBER SO SOON AS HE HAD COME OUT FROM HIS CONVERSE WITH THE SQUIRE",
    },
    {
        "path": "61/70970/61-70970-0002.flac",
        "text": "MOST OF ALL ROBIN THOUGHT OF HIS FATHER WHAT WOULD HE COUNSEL",
    },
    {
        "path": "61/70970/61-70970-0003.flac",
        "text": "IF FOR A WHIM YOU BEGGAR YOURSELF I CANNOT STAY YOU",
    },
]

# Room = cafe-like (larger, more reverberant)
ROOM_DIM = [8.0, 6.0, 3.0]  # meters

# Mic array at center of room
MIC_CENTER = [4.0, 3.0]

# Source positions — speech always from ~60° azimuth
SPEECH_POS = [5.5, 4.5, 1.5]  # ~60° relative to mic array

# Sine positions: same-side and opposite-side
SINE_POS_SAME = [6.0, 4.0, 1.5]    # Near speech source (~45°)
SINE_POS_OPP = [2.5, 1.5, 1.5]     # Opposite side (~240°)


def build_scene_config(
    utterance_idx: int,
    sine_freq: float,
    sine_pos: list,
    snr_db: float,
    rt60: float = 0.4,
    sine_amplitude: float = 0.8,
    seed: int = 42,
) -> SceneConfig:
    """Build a SceneConfig with speech + sine wave."""
    utt = UTTERANCES[utterance_idx % len(UTTERANCES)]
    audio_path = str(LIBRISPEECH_ROOT / utt["path"])

    sources = [
        SourceConfig(
            position=SPEECH_POS,
            signal_type="speech",
            audio_path=audio_path,
            amplitude=1.0,
            label="S0-speech",
            transcription=utt["text"],
            onset_s=0.2,  # Small onset delay
            offset_s=-1,
        ),
        SourceConfig(
            position=sine_pos,
            signal_type="sine",
            frequency=sine_freq,
            amplitude=sine_amplitude,
            label="S1-sine",
            transcription=None,
            onset_s=0.0,  # Sine runs from start
            offset_s=-1,
        ),
    ]

    return SceneConfig(
        room_dim=ROOM_DIM,
        rt60=rt60,
        snr_db=snr_db,
        sources=sources,
        mic_array=MicArrayConfig.linear_array(
            n_mics=4, spacing=0.015, center=MIC_CENTER,
        ),
        duration_s=5.0,
        fs=16000,
        noise_type="white",
        seed=seed,
    )


def compute_true_doa(source_pos, mic_center):
    """Compute true DOA azimuth from source to mic center (degrees)."""
    dx = source_pos[0] - mic_center[0]
    dy = source_pos[1] - mic_center[1]
    azimuth = np.degrees(np.arctan2(dy, dx))
    return azimuth


def run_single_case(
    case_name: str,
    config: SceneConfig,
    use_mvdr: bool = False,
    use_enhancement: bool = True,
    use_wavelet_enh: bool = False,
    use_vad: bool = True,
):
    """Run full pipeline on a single test case and return results dict."""
    logger.info(f"\n{'='*70}")
    logger.info(f"CASE: {case_name}")
    logger.info(f"{'='*70}")

    # Generate scene
    simulator = RIRSimulator(seed=config.seed)
    scene = simulator.generate_scene(config)

    # True DOAs
    true_speech_doa = scene.true_doas[0] if len(scene.true_doas) > 0 else None
    true_sine_doa = scene.true_doas[1] if len(scene.true_doas) > 1 else None
    logger.info(f"True speech DOA: {true_speech_doa:.1f}°")
    logger.info(f"True sine DOA: {true_sine_doa:.1f}°")

    # Build pipeline data dict
    data = scene.to_pipeline_dict()

    # ── Step 1: VAD ──
    vad_result = {}
    if use_vad:
        try:
            vad = TENVadModule(speech_threshold=0.5, noise_threshold=0.3)
            data = vad.process(data)
            n_speech = sum(data.get("vad_is_speech", []))
            n_noise = sum(data.get("vad_is_noise", []))
            n_total = len(data.get("vad_is_speech", []))
            vad_result = {
                "speech_frames": n_speech,
                "noise_frames": n_noise,
                "total_frames": n_total,
                "speech_pct": 100 * n_speech / max(n_total, 1),
                "noise_pct": 100 * n_noise / max(n_total, 1),
                "n_segments": len(data.get("vad_speech_segments", [])),
            }
            logger.info(
                f"VAD: {n_speech}/{n_total} speech frames "
                f"({vad_result['speech_pct']:.0f}%), "
                f"{n_noise} noise, "
                f"{vad_result['n_segments']} segments"
            )
        except Exception as e:
            logger.warning(f"VAD failed: {e}")

    # ── Step 2: SSL ──
    ssl = GccPhatSSL()
    data = ssl.process(data)
    est_doa = data.get("estimated_doa", None)
    ssl_error = abs(est_doa - true_speech_doa) if (est_doa is not None and true_speech_doa is not None) else None
    # Wrap to [0, 180] since GCC-PHAT has front/back ambiguity
    if ssl_error is not None and ssl_error > 180:
        ssl_error = 360 - ssl_error
    logger.info(f"SSL: estimated={est_doa:.1f}°, error={ssl_error:.1f}°")

    # ── Step 3: Beamforming ──
    if use_mvdr:
        bf = MVDRBeamformer()
    else:
        bf = DelayAndSumBeamformer()
    data = bf.process(data)

    # Compute SI-SDR of beamformed output vs reverberant clean speech
    bf_si_sdr = None
    bf_si_sdr_improv = None
    if scene.reverberant_sources is not None and len(scene.clean_sources) > 0:
        rev = scene.reverberant_sources
        if isinstance(rev, np.ndarray) and rev.ndim >= 2:
            ref = rev[0, 0] if rev.ndim == 3 else rev[0]
        elif isinstance(rev, list) and len(rev) > 0:
            r0 = rev[0]
            ref = r0[0] if r0.ndim >= 2 else r0
        else:
            ref = scene.clean_sources[0]

        bf_audio = data["beamformed_audio"]
        if bf_audio.ndim > 1:
            bf_audio = bf_audio[0]

        sr = scene.sample_rate
        onset = int(getattr(config.sources[0], 'onset_s', 0.0) * sr)
        min_len = min(len(bf_audio), len(ref))
        start = min(onset, min_len)
        ref_active = ref[start:min_len]
        bf_active = bf_audio[start:min_len]

        if len(ref_active) > sr // 2:
            bf_si_sdr = si_sdr(ref_active, bf_active)

            # SI-SDR improvement vs raw mic 0
            noisy_ref = scene.multichannel_audio[0][start:min_len]
            noisy_si_sdr_val = si_sdr(ref_active, noisy_ref)
            bf_si_sdr_improv = bf_si_sdr - noisy_si_sdr_val

    logger.info(f"BF SI-SDR: {bf_si_sdr:.1f} dB" if bf_si_sdr else "BF SI-SDR: N/A")
    if bf_si_sdr_improv is not None:
        logger.info(f"BF SI-SDR improvement: {bf_si_sdr_improv:+.1f} dB")

    # ── Step 4: Enhancement ──
    enh_applied = False
    if use_enhancement:
        if use_wavelet_enh:
            enhancer = WaveletEnhancer()
        else:
            enhancer = SpectralSubtractionEnhancer()
        # Use gated enhancer (skip if multi-speaker)
        gated = GatedEnhancer(enhancer)
        data = gated.process(data)
        enh_applied = data.get("enhancement_applied", False)
        logger.info(f"Enhancement: {'APPLIED' if enh_applied else 'SKIPPED'} "
                     f"({data.get('enhancement_gate_reason', 'unknown')})")
    else:
        # Pass beamformed audio as "enhanced" for ASR
        if "beamformed_audio" in data:
            audio = data["beamformed_audio"]
            if audio.ndim > 1:
                audio = audio[0]
            data["enhanced_audio"] = audio.copy()
            data["enhancement_method"] = "none"
            data["enhancement_latency_ms"] = 0.0

    # ── Step 5: ASR ──
    asr = WhisperOfflineASR(model_size="base")
    data = asr.process(data)

    # Compute WER
    ref_text = config.sources[0].transcription
    hyp_texts = data.get("transcriptions", [])
    hyp_text = hyp_texts[0] if hyp_texts else ""
    wer = word_error_rate(ref_text, hyp_text) if ref_text else None

    logger.info(f"Reference: {ref_text}")
    logger.info(f"Hypothesis: {hyp_text}")
    logger.info(f"WER: {wer:.1%}" if wer is not None else "WER: N/A")

    # ── Collect results ──
    result = {
        "case_name": case_name,
        "sine_freq": config.sources[1].frequency,
        "sine_position": "same_side" if config.sources[1].position == SINE_POS_SAME else "opposite",
        "snr_db": config.snr_db,
        "rt60": config.rt60,
        "beamformer": "MVDR" if use_mvdr else "DS",
        "enhancement": ("wavelet" if use_wavelet_enh else "spectral_sub") if use_enhancement else "none",
        "enhancement_applied": enh_applied,
        "true_speech_doa": true_speech_doa,
        "true_sine_doa": true_sine_doa,
        "estimated_doa": est_doa,
        "ssl_error_deg": ssl_error,
        "bf_si_sdr_db": bf_si_sdr,
        "bf_si_sdr_improvement_db": bf_si_sdr_improv,
        "wer": wer,
        "reference_text": ref_text,
        "hypothesis_text": hyp_text,
        "vad": vad_result,
        "ssl_latency_ms": data.get("ssl_latency_ms", 0),
        "bf_latency_ms": data.get("bf_latency_ms", 0),
        "enhancement_latency_ms": data.get("enhancement_latency_ms", 0),
        "asr_latency_ms": data.get("asr_latency_ms", 0),
        "total_latency_ms": (
            data.get("ssl_latency_ms", 0)
            + data.get("bf_latency_ms", 0)
            + data.get("enhancement_latency_ms", 0)
            + data.get("asr_latency_ms", 0)
        ),
    }

    return result


def main():
    t0 = time.time()
    all_results = []

    # ── Test Matrix ──
    # 3 sine freqs × 2 positions × 2 SNRs = 12 core cases
    # Then a few with MVDR and wavelet enhancement for comparison
    sine_freqs = [500, 1000, 3000]
    sine_positions = [
        ("same_side", SINE_POS_SAME),
        ("opposite", SINE_POS_OPP),
    ]
    snr_levels = [15, 5]

    # ── Core tests: DS + Spectral Sub ──
    case_num = 0
    for freq in sine_freqs:
        for pos_name, pos in sine_positions:
            for snr in snr_levels:
                case_name = f"DS_SpecSub_{freq}Hz_{pos_name}_SNR{snr}"
                config = build_scene_config(
                    utterance_idx=case_num % len(UTTERANCES),
                    sine_freq=float(freq),
                    sine_pos=pos,
                    snr_db=float(snr),
                    rt60=0.4,
                    seed=42 + case_num,
                )
                result = run_single_case(
                    case_name, config,
                    use_mvdr=False,
                    use_enhancement=True,
                    use_wavelet_enh=False,
                )
                all_results.append(result)
                case_num += 1

    # ── Comparison: No enhancement ──
    for freq in [1000]:
        for pos_name, pos in sine_positions:
            for snr in [15, 5]:
                case_name = f"DS_NoEnh_{freq}Hz_{pos_name}_SNR{snr}"
                config = build_scene_config(
                    utterance_idx=0,
                    sine_freq=float(freq),
                    sine_pos=pos,
                    snr_db=float(snr),
                    rt60=0.4,
                    seed=100,
                )
                result = run_single_case(
                    case_name, config,
                    use_mvdr=False,
                    use_enhancement=False,
                )
                all_results.append(result)

    # ── MVDR comparison (should null sine better) ──
    for freq in [1000]:
        for pos_name, pos in sine_positions:
            case_name = f"MVDR_{freq}Hz_{pos_name}_SNR15"
            config = build_scene_config(
                utterance_idx=0,
                sine_freq=float(freq),
                sine_pos=pos,
                snr_db=15.0,
                rt60=0.4,
                seed=200,
            )
            result = run_single_case(
                case_name, config,
                use_mvdr=True,
                use_enhancement=True,
                use_wavelet_enh=False,
            )
            all_results.append(result)

    # ── Wavelet enhancement comparison ──
    for freq in [1000]:
        case_name = f"DS_WaveletEnh_{freq}Hz_opposite_SNR15"
        config = build_scene_config(
            utterance_idx=0,
            sine_freq=float(freq),
            sine_pos=SINE_POS_OPP,
            snr_db=15.0,
            rt60=0.4,
            seed=300,
        )
        result = run_single_case(
            case_name, config,
            use_mvdr=False,
            use_enhancement=True,
            use_wavelet_enh=True,
        )
        all_results.append(result)

    # ── Print summary table ──
    print("\n" + "=" * 120)
    print("EXPERIMENT 03: NOISY CAFE — SPEECH + SINE WAVE RESULTS")
    print("=" * 120)
    print(f"{'Case':<40} {'SSL Err':>8} {'BF SDR':>8} {'SDR Imp':>8} "
          f"{'WER':>8} {'VAD Sp%':>8} {'Lat ms':>8}")
    print("-" * 120)

    for r in all_results:
        ssl_err = f"{r['ssl_error_deg']:.1f}°" if r['ssl_error_deg'] is not None else "--"
        bf_sdr = f"{r['bf_si_sdr_db']:.1f}" if r['bf_si_sdr_db'] is not None else "--"
        sdr_imp = f"{r['bf_si_sdr_improvement_db']:+.1f}" if r['bf_si_sdr_improvement_db'] is not None else "--"
        wer = f"{r['wer']:.1%}" if r['wer'] is not None else "--"
        vad_sp = f"{r['vad'].get('speech_pct', 0):.0f}%" if r['vad'] else "--"
        lat = f"{r['total_latency_ms']:.0f}"
        print(f"{r['case_name']:<40} {ssl_err:>8} {bf_sdr:>8} {sdr_imp:>8} "
              f"{wer:>8} {vad_sp:>8} {lat:>8}")

    # ── Analysis ──
    print("\n" + "=" * 120)
    print("ANALYSIS")
    print("=" * 120)

    # Group by freq
    for freq in sine_freqs:
        cases = [r for r in all_results if r["sine_freq"] == freq
                 and r["beamformer"] == "DS" and r["enhancement"] == "spectral_sub"]
        if not cases:
            continue
        avg_ssl = np.mean([r["ssl_error_deg"] for r in cases if r["ssl_error_deg"] is not None])
        avg_wer = np.mean([r["wer"] for r in cases if r["wer"] is not None])
        print(f"\n{freq} Hz sine: avg SSL error = {avg_ssl:.1f}°, avg WER = {avg_wer:.1%}")

    # DS vs MVDR
    ds_cases = [r for r in all_results if r["beamformer"] == "DS"
                and r["sine_freq"] == 1000 and r["enhancement"] == "spectral_sub"
                and r["snr_db"] == 15]
    mvdr_cases = [r for r in all_results if r["beamformer"] == "MVDR"
                  and r["sine_freq"] == 1000]
    if ds_cases and mvdr_cases:
        print("\n── DS vs MVDR at 1kHz, SNR=15 ──")
        for r in ds_cases + mvdr_cases:
            bf_sdr = r["bf_si_sdr_db"]
            wer = r["wer"]
            print(f"  {r['case_name']}: SI-SDR={bf_sdr:.1f}dB, WER={wer:.1%}" if bf_sdr and wer else f"  {r['case_name']}: N/A")

    # Enhancement vs no enhancement
    enh_cases = [r for r in all_results if "SpecSub" in r["case_name"]
                 and r["sine_freq"] == 1000]
    no_enh = [r for r in all_results if "NoEnh" in r["case_name"]]
    if enh_cases and no_enh:
        print("\n── Enhancement vs No Enhancement (1kHz sine) ──")
        for r in enh_cases + no_enh:
            wer = r["wer"]
            print(f"  {r['case_name']}: WER={wer:.1%}" if wer else f"  {r['case_name']}: N/A")

    # ── Save results ──
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    total_time = time.time() - t0
    print(f"\nTotal experiment time: {total_time:.1f}s ({len(all_results)} cases)")


if __name__ == "__main__":
    main()
