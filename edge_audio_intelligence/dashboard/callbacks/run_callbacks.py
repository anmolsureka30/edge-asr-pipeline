"""
Run callback: simulate scene + run pipeline + evaluate + generate plots + save.

This is the core callback that connects the dashboard to the pipeline.
"""

import logging
import time
import traceback
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, html, dcc, no_update
import dash_bootstrap_components as dbc

from ..state import RunHistory, RunRecord, encode_audio_data_uri, save_audio_signal
from ..components.results_panel import build_stage_card

logger = logging.getLogger(__name__)

# Module-level cache for current run data (arrays too large for JSON store)
_current_pipeline_data = {}
_current_scene = None
_run_history = RunHistory()

# LibriSpeech index (built lazily)
_libri_index = None
# Project root is 4 levels up: callbacks/ -> dashboard/ -> edge_audio_intelligence/ -> Wavelets/
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_LIBRI_DIR = _PROJECT_ROOT / "data" / "librispeech" / "LibriSpeech" / "test-clean"


def _build_libri_index():
    """Build LibriSpeech utterance index (lazy, cached)."""
    global _libri_index
    if _libri_index is not None:
        return _libri_index

    _libri_index = []
    if not _LIBRI_DIR.exists():
        logger.warning(f"LibriSpeech not found at {_LIBRI_DIR}")
        return _libri_index

    for trans_file in sorted(_LIBRI_DIR.rglob("*.trans.txt")):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    flac = trans_file.parent / f"{utt_id}.flac"
                    if flac.exists():
                        speaker_id = utt_id.split("-")[0]
                        _libri_index.append({
                            "id": utt_id,
                            "path": str(flac),
                            "text": text,
                            "speaker": speaker_id,
                            "label": f"[{speaker_id}] {text[:60]}...",
                        })

    logger.info(f"LibriSpeech index: {len(_libri_index)} utterances")
    return _libri_index


def _build_module(stage, value):
    """Instantiate a pipeline module from dropdown value."""
    if stage == "ssl":
        if value == "gcc_phat":
            from ...modules.ssl import GccPhatSSL
            return GccPhatSSL(n_fft=1024)
        elif value == "srp_phat":
            from ...modules.ssl import SrpPhatSSL
            return SrpPhatSSL(n_fft=1024, grid_resolution=360)
        elif value == "music":
            from ...modules.ssl import MusicSSL
            return MusicSSL(n_fft=1024, n_sources=1, grid_resolution=360)

    elif stage == "beamforming":
        if value == "delay_and_sum":
            from ...modules.beamforming import DelayAndSumBeamformer
            return DelayAndSumBeamformer(use_fractional_delay=True)
        elif value == "mvdr":
            from ...modules.beamforming import MVDRBeamformer
            return MVDRBeamformer()

    elif stage == "enhancement":
        if value == "spectral_subtraction":
            from ...modules.enhancement import SpectralSubtractionEnhancer
            return SpectralSubtractionEnhancer(alpha=2.0, beta=0.01)
        elif value == "wavelet_enhancement":
            from ...modules.enhancement import WaveletEnhancer
            return WaveletEnhancer(wavelet="bior2.2", levels=3, threshold_scale=1.0)

    elif stage == "asr":
        if value == "whisper_tiny":
            from ...modules.asr import WhisperOfflineASR
            return WhisperOfflineASR(model_size="tiny")
        elif value == "whisper_base":
            from ...modules.asr import WhisperOfflineASR
            return WhisperOfflineASR(model_size="base")
        elif value == "whisper_small":
            from ...modules.asr import WhisperOfflineASR
            return WhisperOfflineASR(model_size="small")

    elif stage == "vad":
        if value == "full_vad":
            from ...modules.vad.ten_vad_module import TENVadModule
            return TENVadModule(hop_size=256, speech_threshold=0.5, noise_threshold=0.3)
        elif value == "ten_vad":
            from ...modules.vad.ten_vad_module import TENVadModule
            return TENVadModule(hop_size=256, speech_threshold=0.5, noise_threshold=0.3)
        elif value == "wavelet_vad":
            from ...modules.vad.wavelet_vad_module import WaveletVADModule
            return WaveletVADModule()
        elif value == "atomic_vad":
            from ...modules.vad.atomic_vad_module import AtomicVADModule
            return AtomicVADModule()

    return None


def _make_waveform_fig(audio, sr, title="Waveform"):
    """Create a plotly waveform figure."""
    if audio.ndim > 1:
        audio = audio[0]  # First channel
    t = np.arange(len(audio)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=audio, mode="lines", line=dict(width=0.5),
        name=title,
    ))
    fig.update_layout(
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        title=title, margin=dict(l=40, r=10, t=30, b=30),
        height=180,
    )
    return fig


def _make_spectrogram_fig(audio, sr, title="Spectrogram"):
    """Create a plotly spectrogram figure using STFT."""
    from scipy.signal import stft as scipy_stft
    if audio.ndim > 1:
        audio = audio[0]

    f, t, Zxx = scipy_stft(audio, fs=sr, nperseg=512, noverlap=256)
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    fig = go.Figure(go.Heatmap(
        z=magnitude_db, x=t, y=f,
        colorscale="Viridis",
        colorbar=dict(title="dB"),
    ))
    fig.update_layout(
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        title=title, margin=dict(l=40, r=10, t=30, b=30),
        height=180,
    )
    return fig


def _make_scalogram_fig(audio, sr, title="Wavelet Scalogram"):
    """Create a plotly scalogram figure using DWT."""
    try:
        from ...wavelet.analysis import WaveletAnalyzer
        analyzer = WaveletAnalyzer(wavelet="bior2.2", levels=4)
        scalogram = analyzer.compute_scalogram(audio if audio.ndim == 1 else audio[0], sr)

        band_names = scalogram["band_names"]
        coeffs = scalogram["coefficients"]

        # Build heatmap from coefficients
        max_len = max(len(c) for c in coeffs)
        grid = np.zeros((len(coeffs), max_len))
        for i, c in enumerate(coeffs):
            # Resample each band to same length
            if len(c) < max_len:
                indices = np.linspace(0, len(c) - 1, max_len).astype(int)
                grid[i] = np.abs(c[indices])
            else:
                grid[i] = np.abs(c[:max_len])

        t_axis = np.linspace(0, len(audio if audio.ndim == 1 else audio[0]) / sr, max_len)

        fig = go.Figure(go.Heatmap(
            z=grid, x=t_axis, y=band_names,
            colorscale="Hot",
            colorbar=dict(title="|coeff|"),
        ))
        fig.update_layout(
            xaxis_title="Time (s)", yaxis_title="Sub-band",
            title=title, margin=dict(l=60, r=10, t=30, b=30),
            height=180,
        )
        return fig
    except Exception as e:
        logger.warning(f"Scalogram failed: {e}")
        return None


def register_run_callbacks(app):
    """Register the RUN PIPELINE callback and audio playback."""

    from dash import ALL

    # LibriSpeech dropdown options — populate ALL source cards
    @app.callback(
        Output({"type": "source-libri", "index": ALL}, "options"),
        Input("scene-store", "data"),
    )
    def populate_libri_dropdowns(scene_data):
        index = _build_libri_index()
        options = [
            {"label": u["label"], "value": u["id"]}
            for u in index[:500]
        ]
        # Return same options for every source card
        n_sources = len(scene_data.get("sources", [])) if scene_data else 1
        return [options] * max(n_sources, 1)

    # RUN PIPELINE
    @app.callback(
        [
            Output("run-status", "children"),
            Output("run-status", "color"),
            Output("stage-results-container", "children"),
            Output("live-output-card", "style"),
            Output("audio-signal-dropdown", "options"),
            Output("audio-signal-dropdown", "value"),
            Output("audio-player-container", "children"),
            Output("transcription-output", "children"),
            Output("metrics-table-container", "children"),
            Output("summary-card", "style"),
            Output("run-summary-content", "children"),
            Output("current-run-store", "data"),
            Output("run-history-container", "children"),
        ],
        Input("btn-run-pipeline", "n_clicks"),
        [
            State("scene-store", "data"),
            State("room-width", "value"),
            State("room-depth", "value"),
            State("room-height", "value"),
            State("rt60-slider", "value"),
            State("snr-slider", "value"),
            State("duration-slider", "value"),
            State("pipeline-ssl", "value"),
            State("pipeline-bf", "value"),
            State("pipeline-enh", "value"),
            State("pipeline-asr", "value"),
            State("pipeline-vad", "value"),
            State("pipeline-enh-gate", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_pipeline(
        n_clicks,
        scene_data, room_w, room_d, room_h, rt60, snr, duration,
        ssl_val, bf_val, enh_val, asr_val, vad_val, enh_gate_val,
    ):
        global _current_pipeline_data, _current_scene

        if not n_clicks:
            return no_update

        try:
            from ...testbench.scene import SceneConfig, SourceConfig, MicArrayConfig
            from ...testbench.simulator import RIRSimulator
            from ...testbench.evaluator import PipelineEvaluator
            from ...pipeline.cascade import CascadePipeline

            # Build sources
            sources = []
            libri_index = _build_libri_index()
            libri_map = {u["id"]: u for u in libri_index}

            for src_data in scene_data.get("sources", []):
                signal_type = src_data.get("signal", "sine")
                audio_path = None
                transcription = None
                freq = 440.0

                if signal_type == "librispeech":
                    # Use the user's selected utterance from the dropdown
                    selected_id = src_data.get("libri_selection", "")
                    if selected_id and selected_id in libri_map:
                        utt = libri_map[selected_id]
                    elif libri_index:
                        utt = libri_index[0]  # Fallback to first
                    else:
                        utt = None

                    if utt:
                        audio_path = utt["path"]
                        transcription = utt["text"]
                        signal_type = "speech"
                    else:
                        signal_type = "sine"

                volume = src_data.get("volume", 1.0)

                sources.append(SourceConfig(
                    position=[src_data.get("x", 2), src_data.get("y", 3.5), src_data.get("z", 1.5)],
                    signal_type=signal_type,
                    audio_path=audio_path,
                    frequency=freq,
                    amplitude=volume,
                    transcription=transcription,
                    label=f"S{len(sources)}",
                    onset_s=src_data.get("onset_s", 0.0),
                    offset_s=src_data.get("offset_s", -1.0),
                ))

            if not sources:
                sources.append(SourceConfig(
                    position=[2.0, 3.5, 1.5],
                    signal_type="sine",
                    frequency=1000.0,
                    label="S0",
                ))

            # Build mic array from store
            mic_positions = scene_data.get("mic_positions", [])
            if not mic_positions:
                mac = MicArrayConfig.linear_array(
                    n_mics=4, spacing=0.05,
                    center=[room_w / 2, room_d / 2], height=1.2,
                )
                mic_positions = mac.positions
            mic_array = MicArrayConfig(positions=mic_positions)

            # Build scene config
            config = SceneConfig(
                room_dim=[room_w, room_d, room_h],
                rt60=rt60,
                snr_db=snr,
                sources=sources,
                mic_array=mic_array,
                duration_s=duration,
                fs=16000,
                noise_type="white",
                seed=42,
            )

            # Simulate
            simulator = RIRSimulator(seed=42)
            scene = simulator.generate_scene(config)
            _current_scene = scene

            # Build pipeline
            pipeline = CascadePipeline(name="dashboard_run")
            module_names = {}
            enh_gate_on = "on" in (enh_gate_val or [])

            # ---- PRE-PIPELINE: VAD on raw mic audio (for MVDR Phi_nn gating) ----
            # VAD must run BEFORE beamforming so MVDR can use noise labels.
            # The chicken-and-egg: we run VAD on raw mic 0 (coarse but sufficient
            # for Phi_nn estimation), then beamforming uses those labels.
            vad_module = _build_module("vad", vad_val) if vad_val and vad_val != "none" else None

            # ---- PIPELINE STAGES ----
            ssl_module = _build_module("ssl", ssl_val)
            if ssl_module:
                pipeline.add_module(ssl_module)
                module_names["ssl"] = ssl_module.name

            bf_module = _build_module("beamforming", bf_val)
            if bf_module:
                pipeline.add_module(bf_module)
                module_names["beamforming"] = bf_module.name

            # Enhancement — with optional gate
            enh_module = _build_module("enhancement", enh_val)
            if enh_module:
                if enh_gate_on:
                    from ...modules.enhancement.gated import GatedEnhancer
                    gated = GatedEnhancer(enh_module)
                    pipeline.add_module(gated)
                    module_names["enhancement"] = f"{enh_module.name} (gated)"
                else:
                    pipeline.add_module(enh_module)
                    module_names["enhancement"] = enh_module.name

            asr_module = _build_module("asr", asr_val)
            if asr_module:
                pipeline.add_module(asr_module)
                module_names["asr"] = asr_module.name

            # Run pipeline
            data = scene.to_pipeline_dict()

            # PRE-PIPELINE: Run VAD on raw mic 0 audio for Phi_nn gating
            # This gives MVDR noise frame labels BEFORE beamforming runs.
            import time as _time

            # Full VAD mode: AtomicVAD wake check THEN TEN VAD precision
            if vad_val == "full_vad":
                # Stage 0: AtomicVAD wake gate
                try:
                    from ...modules.vad.atomic_vad_module import AtomicVADModule
                    atomic = AtomicVADModule()
                    t_wake = _time.perf_counter()
                    wake_result = atomic.process(data.copy())
                    wake_latency = (_time.perf_counter() - t_wake) * 1000.0

                    # Check if any speech detected (wake decision)
                    wake_speech = any(wake_result.get("vad_is_speech", []))
                    data["vad_wake_detected"] = wake_speech
                    data["vad_wake_latency_ms"] = wake_latency
                    data["vad_wake_method"] = "AtomicVAD"

                    if not wake_speech:
                        logger.info("AtomicVAD: no speech detected, pipeline stays asleep")
                        data["vad_method"] = "AtomicVAD (asleep)"
                        data["vad_latency_ms"] = wake_latency
                        module_names["vad"] = "AtomicVAD (asleep)"
                except Exception as e:
                    logger.warning(f"AtomicVAD wake check failed: {e}, proceeding with TEN VAD")
                    data["vad_wake_detected"] = True
                    wake_latency = 0.0

                # Stage 1: TEN VAD precision (always runs — wake gate is informational in sim)
                if vad_module is not None:
                    t_vad = _time.perf_counter()
                    vad_result = vad_module.process(data.copy())
                    vad_latency = (_time.perf_counter() - t_vad) * 1000.0
                    data["vad_frame_probs"] = vad_result["vad_frame_probs"]
                    data["vad_is_speech"] = vad_result["vad_is_speech"]
                    data["vad_is_noise"] = vad_result["vad_is_noise"]
                    data["vad_speech_segments"] = vad_result["vad_speech_segments"]
                    data["vad_method"] = "AtomicVAD + TEN VAD"
                    data["vad_latency_ms"] = wake_latency + vad_latency
                    data["vad_frame_duration_ms"] = vad_result["vad_frame_duration_ms"]
                    module_names["vad"] = "AtomicVAD + TEN VAD"

            elif vad_module is not None:
                # Single VAD mode
                t_vad = _time.perf_counter()
                vad_result = vad_module.process(data.copy())
                vad_latency = (_time.perf_counter() - t_vad) * 1000.0
                data["vad_frame_probs"] = vad_result["vad_frame_probs"]
                data["vad_is_speech"] = vad_result["vad_is_speech"]
                data["vad_is_noise"] = vad_result["vad_is_noise"]
                data["vad_speech_segments"] = vad_result["vad_speech_segments"]
                data["vad_method"] = vad_result["vad_method"]
                data["vad_latency_ms"] = vad_latency
                data["vad_frame_duration_ms"] = vad_result["vad_frame_duration_ms"]
                module_names["vad"] = vad_module.name

            data = pipeline.run(data)
            _current_pipeline_data = data

            # Evaluate
            results_dir = str(Path(__file__).parent.parent.parent / "results")
            evaluator = PipelineEvaluator(results_dir=results_dir)
            result = evaluator.run_full_evaluation(data, scene, "dashboard_run")

            # Build stage result cards
            stage_cards = []
            sr = data.get("sample_rate", 16000)

            # SSL stage
            if "estimated_doa" in data:
                ssl_metrics = {}
                for mr in result.module_results:
                    if "angular_error_deg" in mr.metrics:
                        ssl_metrics = mr.metrics
                        break
                # Spatial spectrum plot
                spatial_fig = None
                if "spatial_spectrum" in data:
                    spec = data["spatial_spectrum"]
                    angles = np.linspace(0, 360, len(spec))
                    spatial_fig = go.Figure(go.Scatter(
                        x=angles, y=spec, mode="lines",
                    ))
                    spatial_fig.update_layout(
                        xaxis_title="Angle (deg)", yaxis_title="Power",
                        title="Spatial Spectrum", height=180,
                        margin=dict(l=40, r=10, t=30, b=30),
                    )

                input_wf = _make_waveform_fig(scene.multichannel_audio, sr, "Multichannel Input (Mic 0)")
                stage_cards.append(build_stage_card(
                    "SSL", module_names.get("ssl", "?"),
                    data.get("ssl_latency_ms", 0),
                    ssl_metrics,
                    waveform_fig=input_wf,
                    spectrogram_fig=spatial_fig,
                ))

            # Beamforming stage
            if "beamformed_audio" in data:
                bf_metrics = {}
                for mr in result.module_results:
                    if "si_sdr_improvement_db" in mr.metrics:
                        bf_metrics = mr.metrics
                        break
                bf_audio = data["beamformed_audio"]
                bf_wf = _make_waveform_fig(bf_audio, sr, "Beamformed Output")
                bf_spec = _make_spectrogram_fig(bf_audio, sr, "Beamformed Spectrogram")
                bf_scal = _make_scalogram_fig(bf_audio, sr, "Beamformed Scalogram")
                stage_cards.append(build_stage_card(
                    "Beamforming", module_names.get("beamforming", "?"),
                    data.get("bf_latency_ms", 0),
                    bf_metrics,
                    waveform_fig=bf_wf,
                    spectrogram_fig=bf_spec,
                    scalogram_fig=bf_scal,
                ))

            # VAD stage — waveform with speech/noise overlay
            if "vad_frame_probs" in data:
                vad_probs = data["vad_frame_probs"]
                vad_speech = data.get("vad_is_speech", [])
                vad_noise = data.get("vad_is_noise", [])
                frame_dur_ms = data.get("vad_frame_duration_ms", 16.0)

                # Get the audio that VAD processed
                if "beamformed_audio" in data:
                    vad_audio = data["beamformed_audio"]
                else:
                    vad_audio = scene.multichannel_audio[0]
                if vad_audio.ndim > 1:
                    vad_audio = vad_audio[0]

                # Build waveform with VAD overlay
                t_audio = np.arange(len(vad_audio)) / sr
                vad_fig = go.Figure()
                vad_fig.add_trace(go.Scatter(
                    x=t_audio, y=vad_audio, mode="lines",
                    line=dict(width=0.5, color="#666"),
                    name="Audio", showlegend=True,
                ))

                # Add colored background regions
                frame_dur_s = frame_dur_ms / 1000.0
                for i in range(len(vad_probs)):
                    t_start = i * frame_dur_s
                    t_end = (i + 1) * frame_dur_s
                    if i < len(vad_speech) and vad_speech[i]:
                        color = "rgba(76,175,80,0.25)"   # green = speech
                    elif i < len(vad_noise) and vad_noise[i]:
                        color = "rgba(244,67,54,0.15)"   # red = noise
                    else:
                        color = "rgba(158,158,158,0.1)"  # gray = uncertain

                    vad_fig.add_vrect(
                        x0=t_start, x1=t_end,
                        fillcolor=color, line_width=0,
                        layer="below",
                    )

                # Enhancement gate annotation
                gate_reason = data.get("enhancement_gate_reason", "")
                enh_applied = data.get("enhancement_applied", True)
                gate_text = ""
                if gate_reason:
                    gate_text = f" | Gate: {'APPLIED' if enh_applied else 'SKIPPED'} ({gate_reason})"

                vad_fig.update_layout(
                    title=f"VAD: {data.get('vad_method', '?')}{gate_text}",
                    xaxis_title="Time (s)", yaxis_title="Amplitude",
                    height=200, margin=dict(l=40, r=10, t=35, b=30),
                    legend=dict(x=1, y=1, xanchor="right"),
                )

                # VAD metrics
                n_speech = sum(vad_speech)
                n_noise = sum(vad_noise)
                n_total = len(vad_probs)
                n_uncertain = n_total - n_speech - n_noise
                vad_metrics = {
                    "speech_frames": n_speech,
                    "noise_frames": n_noise,
                    "uncertain_frames": n_uncertain,
                    "speech_pct": round(100.0 * n_speech / max(n_total, 1), 1),
                    "n_segments": len(data.get("vad_speech_segments", [])),
                }

                stage_cards.append(build_stage_card(
                    "VAD", module_names.get("vad", data.get("vad_method", "?")),
                    data.get("vad_latency_ms", 0),
                    vad_metrics,
                    waveform_fig=vad_fig,
                ))

            # Enhancement stage
            if "enhanced_audio" in data:
                enh_metrics = {}
                for mr in result.module_results:
                    if "pesq" in mr.metrics:
                        enh_metrics = mr.metrics
                        break
                enh_audio = data["enhanced_audio"]
                enh_wf = _make_waveform_fig(enh_audio, sr, "Enhanced Output")
                enh_spec = _make_spectrogram_fig(enh_audio, sr, "Enhanced Spectrogram")
                enh_scal = _make_scalogram_fig(enh_audio, sr, "Enhanced Scalogram")

                # Sub-band energy comparison
                try:
                    from ...wavelet.analysis import WaveletAnalyzer
                    analyzer = WaveletAnalyzer()
                    signals_to_compare = {"Input (Mic 0)": scene.multichannel_audio[0]}
                    if "beamformed_audio" in data:
                        bf = data["beamformed_audio"]
                        signals_to_compare["Beamformed"] = bf if bf.ndim == 1 else bf[0]
                    signals_to_compare["Enhanced"] = enh_audio if enh_audio.ndim == 1 else enh_audio[0]
                    if scene.clean_sources:
                        signals_to_compare["Clean Ref"] = scene.clean_sources[0]

                    stage_energies = analyzer.compare_pipeline_stages(signals_to_compare, sr)
                    bands = list(list(stage_energies.values())[0].keys())

                    energy_fig = go.Figure()
                    for stage_name, energies in stage_energies.items():
                        vals = [np.log10(energies[b] + 1e-10) for b in bands]
                        energy_fig.add_trace(go.Bar(name=stage_name, x=bands, y=vals))
                    energy_fig.update_layout(
                        barmode="group", title="Sub-band Energy (log scale)",
                        xaxis_title="Sub-band", yaxis_title="log10(Energy)",
                        height=200, margin=dict(l=40, r=10, t=30, b=30),
                    )
                    extra = dcc.Graph(figure=energy_fig, style={"height": "200px"})
                except Exception as e:
                    logger.warning(f"Sub-band comparison failed: {e}")
                    extra = None

                stage_cards.append(build_stage_card(
                    "Enhancement", module_names.get("enhancement", "?"),
                    data.get("enhancement_latency_ms", 0),
                    enh_metrics,
                    waveform_fig=enh_wf,
                    spectrogram_fig=enh_spec,
                    scalogram_fig=enh_scal,
                    extra_content=extra,
                ))

            # ASR stage
            if data.get("asr_method"):
                asr_metrics = {}
                for mr in result.module_results:
                    if "wer_avg" in mr.metrics:
                        asr_metrics = mr.metrics
                        break

                ref_text = scene.transcriptions[0] if scene.transcriptions else "N/A"
                hyp_text = data["transcriptions"][0] if data.get("transcriptions") else "N/A"

                asr_content = html.Div([
                    html.Div([
                        html.Strong("Reference: "), html.Span(ref_text),
                    ], className="small"),
                    html.Div([
                        html.Strong("Hypothesis: "), html.Span(hyp_text),
                    ], className="small mt-1"),
                ])

                stage_cards.append(build_stage_card(
                    "ASR", module_names.get("asr", "?"),
                    data.get("asr_latency_ms", 0),
                    asr_metrics,
                    extra_content=asr_content,
                ))

            # Audio options — clearly label what each signal is
            audio_options = [
                {"label": f"Mic {i} (room mix — all speakers + noise + reverb)", "value": f"mic_{i}"}
                for i in range(scene.multichannel_audio.shape[0])
            ]
            if "beamformed_audio" in data:
                audio_options.append({
                    "label": f"Beamformed (steered toward {data.get('ssl_method', 'SSL')} estimate)",
                    "value": "beamformed",
                })
            if "enhanced_audio" in data:
                audio_options.append({
                    "label": f"Enhanced ({data.get('enhancement_method', 'enhancement')} output)",
                    "value": "enhanced",
                })
            # Per-source clean references
            for i, _ in enumerate(scene.clean_sources):
                src_label = scene.config.sources[i].label if i < len(scene.config.sources) else f"S{i}"
                trans = scene.transcriptions[i][:40] if i < len(scene.transcriptions) and scene.transcriptions[i] else "non-speech"
                audio_options.append({
                    "label": f"Clean {src_label} (dry, no room): {trans}...",
                    "value": f"clean_{i}",
                })

            # Collect comprehensive metrics
            from ...utils.metrics import character_error_rate, wer_breakdown, output_snr as compute_output_snr
            from ..components.results_panel import build_metrics_table

            all_metrics = {}
            for mr in result.module_results:
                for k, v in mr.metrics.items():
                    if k == "angular_error_deg":
                        all_metrics["ssl_angular_error"] = v
                    elif k == "si_sdr_db" and "bf" in mr.module_name.lower().replace("-", "").replace(" ", ""):
                        all_metrics["bf_si_sdr"] = v
                    elif k == "si_sdr_improvement_db":
                        all_metrics["bf_si_sdr_improvement"] = v
                    elif k == "pesq":
                        all_metrics["enh_pesq"] = v
                    elif k == "stoi":
                        all_metrics["enh_stoi"] = v
                    elif k == "si_sdr_db" and "enh" not in all_metrics.get("bf_si_sdr", ""):
                        all_metrics["enh_si_sdr"] = v
                    elif k == "wer_avg":
                        all_metrics["asr_wer"] = v

            all_metrics["total_latency_ms"] = result.total_latency_ms
            all_metrics["total_rtf"] = result.total_rtf

            # Latency breakdown
            timings = data.get("pipeline_timings", [])
            if timings:
                breakdown = " | ".join(f"{name}: {ms:.0f}ms" for name, ms in timings)
                all_metrics["latency_breakdown"] = breakdown

            # Compute metrics: MACs, energy, parameters, memory
            from ...utils.profiling import estimate_energy_mj, _format_ops
            total_macs = 0
            total_params = 0
            for mod in pipeline.modules:
                try:
                    macs = mod.estimate_macs(data)
                    total_macs += macs
                except Exception:
                    pass
                total_params += mod.count_parameters()

            all_metrics["total_macs"] = total_macs
            all_metrics["total_macs_formatted"] = _format_ops(total_macs)
            all_metrics["total_parameters"] = total_params
            all_metrics["total_energy_mj"] = estimate_energy_mj(
                result.total_latency_ms, device="raspberry_pi_4b"
            )

            # ASR: CER and WER breakdown
            if data.get("transcriptions") and scene.transcriptions:
                ref_text = scene.transcriptions[0]
                hyp_text = data["transcriptions"][0] if data["transcriptions"] else ""
                if ref_text:
                    all_metrics["asr_cer"] = character_error_rate(ref_text, hyp_text)
                    wb = wer_breakdown(ref_text, hyp_text)
                    all_metrics["asr_insertions"] = wb["insertions"]
                    all_metrics["asr_deletions"] = wb["deletions"]
                    all_metrics["asr_substitutions"] = wb["substitutions"]

            # Get reverberant reference (correct domain for SI-SDR)
            ref_rev = None
            if scene.reverberant_sources is not None:
                rev = scene.reverberant_sources
                if isinstance(rev, np.ndarray) and rev.ndim >= 2:
                    ref_rev = rev[0, 0] if rev.ndim == 3 else rev[0]
                elif isinstance(rev, list) and len(rev) > 0:
                    r0 = rev[0]
                    ref_rev = r0[0] if r0.ndim >= 2 else r0
            if ref_rev is None and scene.clean_sources:
                ref_rev = scene.clean_sources[0]

            # Beamforming: output SNR (vs reverberant reference)
            if "beamformed_audio" in data and ref_rev is not None:
                bf_audio = data["beamformed_audio"]
                if bf_audio.ndim > 1:
                    bf_audio = bf_audio[0]
                min_len = min(len(bf_audio), len(ref_rev))
                noise_est = bf_audio[:min_len] - ref_rev[:min_len]
                all_metrics["bf_output_snr"] = compute_output_snr(ref_rev[:min_len], noise_est)

            # Enhancement: SI-SDR improvement vs beamformed (reverberant domain)
            if "enhanced_audio" in data and "beamformed_audio" in data and ref_rev is not None:
                enh = data["enhanced_audio"]
                bf = data["beamformed_audio"]
                if enh.ndim > 1: enh = enh[0]
                if bf.ndim > 1: bf = bf[0]
                min_len = min(len(enh), len(bf), len(ref_rev))
                from ...utils.metrics import si_sdr
                enh_sdr = si_sdr(ref_rev[:min_len], enh[:min_len])
                bf_sdr = si_sdr(ref_rev[:min_len], bf[:min_len])
                all_metrics["enh_si_sdr"] = enh_sdr
                all_metrics["enh_si_sdr_improvement"] = enh_sdr - bf_sdr

            # Build transcription output — show ALL speakers' references + ASR result
            transcription_content = html.Div()
            n_speech_sources = sum(1 for t in scene.transcriptions if t)

            if data.get("asr_method"):
                trans_items = []

                # Show each speaker's ground truth reference
                for i, ref in enumerate(scene.transcriptions):
                    if not ref:
                        continue
                    src_cfg = scene.config.sources[i] if i < len(scene.config.sources) else None
                    label = src_cfg.label if src_cfg else f"S{i}"
                    role = "target" if i == 0 else "interferer"
                    role_color = "danger" if i == 0 else "warning"

                    trans_items.append(html.Div([
                        dbc.Badge(f"{label} ({role})", color=role_color, className="me-1"),
                        html.Small("Ground truth: ", className="text-muted"),
                        html.Span(ref[:150], className="small"),
                    ], className="mb-1"))

                # Show what ASR actually produced (from the mixed/beamformed audio)
                hyp = data["transcriptions"][0] if data.get("transcriptions") else ""
                trans_items.append(html.Hr(className="my-1"))
                trans_items.append(html.Div([
                    dbc.Badge("ASR Output", color="success", className="me-1"),
                    html.Span(hyp[:200] if hyp else "(empty)", className="small fw-bold"),
                ], className="mb-1"))

                # If multi-speaker, show a diagnostic note
                if n_speech_sources > 1:
                    trans_items.append(html.Div([
                        dbc.Badge("Note", color="info", className="me-1"),
                        html.Small(
                            f"{n_speech_sources} speakers detected in scene. "
                            "Current pipeline has no separation module — ASR receives "
                            "the mixed signal. WER is computed against the target speaker (S0) only. "
                            "To properly transcribe both speakers, add a Speaker Separation module (V2).",
                            className="text-muted fst-italic",
                        ),
                    ]))

                transcription_content = html.Div(trans_items)

            elif not data.get("asr_method"):
                transcription_content = html.Div(
                    html.Small("ASR skipped (select Whisper to get transcription)", className="text-muted"),
                )

            # Build metrics table
            metrics_table = build_metrics_table(all_metrics)

            # Summary
            pipeline_str = " → ".join(
                module_names.get(k, "skip") for k in ["ssl", "beamforming", "enhancement", "asr"]
                if module_names.get(k)
            )
            summary = html.Div([
                html.Div([html.Strong("Pipeline: "), html.Span(pipeline_str)]),
                html.Div([
                    html.Strong("Total: "),
                    html.Span(f"{result.total_latency_ms:.0f}ms"),
                    html.Span(" | "),
                    html.Strong("RTF: "),
                    html.Span(f"{result.total_rtf:.3f}"),
                ]),
            ])

            # Save to history
            record = RunRecord(
                scene_config={
                    "room_dim": [room_w, room_d, room_h],
                    "rt60": rt60, "snr_db": snr, "duration_s": duration,
                },
                pipeline_config=module_names,
                sources=[s.__dict__ if hasattr(s, "__dict__") else str(s) for s in sources],
                metrics=all_metrics,
            )
            _run_history.add_run(record)

            from ..components.run_history import build_history_card
            history_cards = [
                build_history_card(r, i) for i, r in enumerate(_run_history.runs)
            ]

            # Auto-play mic 0 audio
            mic0_audio = scene.multichannel_audio[0].astype(np.float32)
            mic0_uri = encode_audio_data_uri(mic0_audio, sr)
            auto_player = html.Audio(
                src=mic0_uri, controls=True, autoPlay=True,
                style={"width": "100%"},
            )

            return (
                f"Pipeline complete! {len(pipeline.modules)} modules, {result.total_latency_ms:.0f}ms total.",
                "success",
                stage_cards,
                {"display": "block"},       # live-output-card
                audio_options,               # audio dropdown options
                "mic_0",                     # audio dropdown value (auto-select mic 0)
                auto_player,                 # audio player with mic 0 auto-playing
                transcription_content,       # transcription
                metrics_table,               # metrics table
                {"display": "block"},        # summary card
                summary,
                record.run_id,
                history_cards,
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {traceback.format_exc()}")
            return (
                f"Pipeline failed: {str(e)}",
                "danger",
                [],
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update, no_update,
            )

    # Audio playback callback (when user changes dropdown selection)
    @app.callback(
        Output("audio-player-container", "children", allow_duplicate=True),
        Input("audio-signal-dropdown", "value"),
        prevent_initial_call=True,
    )
    def play_audio(signal_key):
        if not signal_key or _current_scene is None:
            return no_update

        sr = _current_scene.sample_rate
        audio = None

        if signal_key.startswith("mic_"):
            idx = int(signal_key.split("_")[1])
            audio = _current_scene.multichannel_audio[idx]
        elif signal_key == "beamformed":
            bf = _current_pipeline_data.get("beamformed_audio")
            if bf is not None:
                audio = bf if bf.ndim == 1 else bf[0]
        elif signal_key == "enhanced":
            enh = _current_pipeline_data.get("enhanced_audio")
            if enh is not None:
                audio = enh if enh.ndim == 1 else enh[0]
        elif signal_key.startswith("clean_"):
            idx = int(signal_key.split("_")[1])
            if idx < len(_current_scene.clean_sources):
                audio = _current_scene.clean_sources[idx]

        if audio is None:
            return html.Div("No audio available.", className="text-muted small")

        data_uri = encode_audio_data_uri(audio.astype(np.float32), sr)
        return html.Audio(src=data_uri, controls=True, style={"width": "100%"})
