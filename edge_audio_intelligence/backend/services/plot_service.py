"""PlotService: Generates Plotly figures for the API.

Extracted from dashboard/callbacks/run_callbacks.py.
Returns Plotly figure dicts (JSON-serializable) that the React frontend
renders with react-plotly.js.
"""

import logging

import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def make_waveform_fig(audio: np.ndarray, sr: int, title: str = "Waveform") -> dict:
    """Create a waveform plot as Plotly JSON."""
    if audio.ndim > 1:
        audio = audio[0]
    t = np.arange(len(audio)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=audio, mode="lines", line=dict(width=0.5), name=title,
    ))
    fig.update_layout(
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        title=title, margin=dict(l=40, r=10, t=30, b=30), height=180,
    )
    return fig.to_plotly_json()


def make_spectrogram_fig(audio: np.ndarray, sr: int, title: str = "Spectrogram") -> dict:
    """Create a spectrogram plot as Plotly JSON."""
    from scipy.signal import stft as scipy_stft
    if audio.ndim > 1:
        audio = audio[0]

    f, t, Zxx = scipy_stft(audio, fs=sr, nperseg=512, noverlap=256)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

    fig = go.Figure(go.Heatmap(
        z=magnitude_db, x=t, y=f,
        colorscale="Viridis", colorbar=dict(title="dB"),
    ))
    fig.update_layout(
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        title=title, margin=dict(l=40, r=10, t=30, b=30), height=180,
    )
    return fig.to_plotly_json()


def make_scalogram_fig(audio: np.ndarray, sr: int, title: str = "Scalogram") -> dict:
    """Create a wavelet scalogram as Plotly JSON."""
    try:
        from edge_audio_intelligence.wavelet.analysis import WaveletAnalyzer
        analyzer = WaveletAnalyzer(wavelet="bior2.2", levels=4)
        if audio.ndim > 1:
            audio = audio[0]
        scalogram = analyzer.compute_scalogram(audio, sr)

        band_names = scalogram["band_names"]
        coeffs = scalogram["coefficients"]

        max_len = max(len(c) for c in coeffs)
        grid = np.zeros((len(coeffs), max_len))
        for i, c in enumerate(coeffs):
            if len(c) < max_len:
                indices = np.linspace(0, len(c) - 1, max_len).astype(int)
                grid[i] = np.abs(c[indices])
            else:
                grid[i] = np.abs(c[:max_len])

        t_axis = np.linspace(0, len(audio) / sr, max_len)

        fig = go.Figure(go.Heatmap(
            z=grid, x=t_axis, y=band_names,
            colorscale="Hot", colorbar=dict(title="|coeff|"),
        ))
        fig.update_layout(
            xaxis_title="Time (s)", yaxis_title="Sub-band",
            title=title, margin=dict(l=60, r=10, t=30, b=30), height=180,
        )
        return fig.to_plotly_json()
    except Exception as e:
        logger.warning(f"Scalogram generation failed: {e}")
        return None


def make_vad_overlay_fig(
    audio: np.ndarray, sr: int,
    vad_probs: list, vad_speech: list, vad_noise: list,
    frame_dur_ms: float, title: str = "VAD",
) -> dict:
    """Create a waveform with VAD speech/noise overlay as Plotly JSON."""
    if audio.ndim > 1:
        audio = audio[0]

    t_audio = np.arange(len(audio)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_audio, y=audio, mode="lines",
        line=dict(width=0.5, color="#666"), name="Audio",
    ))

    frame_dur_s = frame_dur_ms / 1000.0
    for i in range(len(vad_probs)):
        t_start = i * frame_dur_s
        t_end = (i + 1) * frame_dur_s
        if i < len(vad_speech) and vad_speech[i]:
            color = "rgba(76,175,80,0.25)"
        elif i < len(vad_noise) and vad_noise[i]:
            color = "rgba(244,67,54,0.15)"
        else:
            color = "rgba(158,158,158,0.1)"
        fig.add_vrect(x0=t_start, x1=t_end, fillcolor=color, line_width=0, layer="below")

    fig.update_layout(
        title=title, xaxis_title="Time (s)", yaxis_title="Amplitude",
        height=200, margin=dict(l=40, r=10, t=35, b=30),
    )
    return fig.to_plotly_json()


def make_spatial_spectrum_fig(spectrum: np.ndarray, title: str = "Spatial Spectrum") -> dict:
    """Create a spatial spectrum plot as Plotly JSON."""
    angles = np.linspace(0, 360, len(spectrum))
    fig = go.Figure(go.Scatter(x=angles, y=spectrum, mode="lines"))
    fig.update_layout(
        xaxis_title="Angle (deg)", yaxis_title="Power",
        title=title, height=180, margin=dict(l=40, r=10, t=30, b=30),
    )
    return fig.to_plotly_json()


def generate_all_plots(data: dict, scene) -> dict:
    """Generate all stage plots from pipeline output.

    Returns: {stage_name: plotly_figure_dict}
    """
    sr = data.get("sample_rate", 16000)
    plots = {}

    # Input waveform
    if scene.multichannel_audio is not None:
        plots["input_waveform"] = make_waveform_fig(
            scene.multichannel_audio, sr, "Input (Mic 0)")

    # SSL spatial spectrum
    if "spatial_spectrum" in data:
        plots["spatial_spectrum"] = make_spatial_spectrum_fig(data["spatial_spectrum"])

    # Beamformed
    if "beamformed_audio" in data:
        bf = data["beamformed_audio"]
        plots["bf_waveform"] = make_waveform_fig(bf, sr, "Beamformed")
        plots["bf_spectrogram"] = make_spectrogram_fig(bf, sr, "Beamformed Spectrogram")
        plots["bf_scalogram"] = make_scalogram_fig(bf, sr, "Beamformed Scalogram")

    # VAD overlay
    if "vad_frame_probs" in data:
        vad_audio = data.get("beamformed_audio", scene.multichannel_audio[0])
        plots["vad_overlay"] = make_vad_overlay_fig(
            vad_audio, sr,
            data["vad_frame_probs"],
            data.get("vad_is_speech", []),
            data.get("vad_is_noise", []),
            data.get("vad_frame_duration_ms", 16.0),
            f"VAD: {data.get('vad_method', '?')}",
        )

    # Enhanced
    if "enhanced_audio" in data:
        enh = data["enhanced_audio"]
        plots["enh_waveform"] = make_waveform_fig(enh, sr, "Enhanced")
        plots["enh_spectrogram"] = make_spectrogram_fig(enh, sr, "Enhanced Spectrogram")
        plots["enh_scalogram"] = make_scalogram_fig(enh, sr, "Enhanced Scalogram")

    return plots
