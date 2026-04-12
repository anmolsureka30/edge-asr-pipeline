"""
Pipeline configuration: algorithm selection per stage + visual flow diagram.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


SSL_OPTIONS = [
    {"label": "GCC-PHAT (recommended)", "value": "gcc_phat"},
    {"label": "SRP-PHAT", "value": "srp_phat"},
]

BF_OPTIONS = [
    {"label": "Delay-and-Sum", "value": "delay_and_sum"},
    {"label": "MVDR", "value": "mvdr"},
    {"label": "None (skip)", "value": "none"},
]

ENH_OPTIONS = [
    {"label": "Spectral Subtraction", "value": "spectral_subtraction"},
    {"label": "Wavelet Enhancement", "value": "wavelet_enhancement"},
    {"label": "None (skip)", "value": "none"},
]

ASR_OPTIONS = [
    {"label": "Whisper tiny", "value": "whisper_tiny"},
    {"label": "Whisper base", "value": "whisper_base"},
    {"label": "Whisper small", "value": "whisper_small"},
    {"label": "None (skip)", "value": "none"},
]

VAD_OPTIONS = [
    {"label": "None (skip)", "value": "none"},
    {"label": "Full VAD (AtomicVAD wake + TEN VAD precision)", "value": "full_vad"},
    {"label": "TEN VAD only (precision timestamps)", "value": "ten_vad"},
    {"label": "AtomicVAD only (wake gate)", "value": "atomic_vad"},
    {"label": "Wavelet Energy only (EE678 baseline)", "value": "wavelet_vad"},
]

# Display names for the flow diagram
DISPLAY_NAMES = {
    "gcc_phat": "GCC-PHAT",
    "srp_phat": "SRP-PHAT",
    "music": "MUSIC",
    "delay_and_sum": "Delay-and-Sum",
    "mvdr": "MVDR",
    "spectral_subtraction": "Spectral Sub",
    "wavelet_enhancement": "Wavelet Enh",
    "whisper_tiny": "Whisper tiny",
    "whisper_base": "Whisper base",
    "whisper_small": "Whisper small",
    "full_vad": "AtomicVAD + TEN VAD",
    "ten_vad": "TEN VAD",
    "wavelet_vad": "Wavelet VAD",
    "atomic_vad": "AtomicVAD",
    "none": "skip",
}

# Pipeline architecture presets
PIPELINE_PRESETS = {
    "custom": {
        "label": "Custom",
        "description": "Select each module manually",
        "ssl": "gcc_phat", "bf": "delay_and_sum",
        "enh": "spectral_subtraction", "asr": "none",
        "vad": "none", "enh_gate": False,
    },
    "cascade_basic": {
        "label": "Basic Cascade (no VAD)",
        "description": "GCC-PHAT → Delay-Sum → Spectral Sub → Whisper tiny. No VAD, no gating.",
        "ssl": "gcc_phat", "bf": "delay_and_sum",
        "enh": "spectral_subtraction", "asr": "whisper_tiny",
        "vad": "none", "enh_gate": False,
    },
    "vad_gated_mvdr": {
        "label": "VAD-Gated MVDR",
        "description": "TEN VAD noise labels → MVDR Phi_nn gating → Wavelet enhancement (gated by speaker count)",
        "ssl": "gcc_phat", "bf": "mvdr",
        "enh": "wavelet_enhancement", "asr": "whisper_base",
        "vad": "ten_vad", "enh_gate": True,
    },
    "full_pipeline": {
        "label": "Full Pipeline (all stages)",
        "description": "AtomicVAD (wake) + TEN VAD (Phi_nn) → MUSIC → MVDR → Wavelet Enh (gated) → Whisper small",
        "ssl": "gcc_phat", "bf": "mvdr",
        "enh": "wavelet_enhancement", "asr": "whisper_small",
        "vad": "full_vad", "enh_gate": True,
    },
    "wavelet_baseline": {
        "label": "Wavelet Baseline (EE678)",
        "description": "Wavelet VAD → GCC-PHAT → Delay-Sum → Wavelet Enh → Whisper tiny. All wavelet-based.",
        "ssl": "gcc_phat", "bf": "delay_and_sum",
        "enh": "wavelet_enhancement", "asr": "whisper_tiny",
        "vad": "wavelet_vad", "enh_gate": False,
    },
}

PIPELINE_PRESET_OPTIONS = [
    {"label": v["label"], "value": k}
    for k, v in PIPELINE_PRESETS.items()
]


def create_pipeline_config():
    """Create the pipeline configuration bar with visual flow."""
    return dbc.Card([
        dbc.CardHeader("Pipeline Configuration", className="py-2"),
        dbc.CardBody([
            # Pipeline preset selector
            dbc.Row([
                dbc.Col([
                    html.Label("Pipeline Preset", className="fw-bold small"),
                    dcc.Dropdown(
                        id="pipeline-preset", options=PIPELINE_PRESET_OPTIONS,
                        value="custom", clearable=False,
                    ),
                ], width=4),
                dbc.Col([
                    html.Div(id="pipeline-preset-description",
                             className="small text-muted mt-3"),
                ], width=8),
            ], className="mb-2"),
            html.Hr(className="my-2"),
            # Algorithm dropdowns — row 1: core pipeline
            dbc.Row([
                dbc.Col([
                    html.Label("SSL", className="fw-bold small"),
                    dcc.Dropdown(
                        id="pipeline-ssl", options=SSL_OPTIONS,
                        value="gcc_phat", clearable=False,
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Beamforming", className="fw-bold small"),
                    dcc.Dropdown(
                        id="pipeline-bf", options=BF_OPTIONS,
                        value="delay_and_sum", clearable=False,
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Enhancement", className="fw-bold small"),
                    dcc.Dropdown(
                        id="pipeline-enh", options=ENH_OPTIONS,
                        value="spectral_subtraction", clearable=False,
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("ASR", className="fw-bold small"),
                    dcc.Dropdown(
                        id="pipeline-asr", options=ASR_OPTIONS,
                        value="none", clearable=False,
                    ),
                ], width=3),
            ], className="mb-2"),
            # Row 2: VAD + Enhancement Gate
            dbc.Row([
                dbc.Col([
                    html.Label("VAD", className="fw-bold small"),
                    dcc.Dropdown(
                        id="pipeline-vad", options=VAD_OPTIONS,
                        value="none", clearable=False,
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Enhancement Gate", className="fw-bold small"),
                    dbc.Checklist(
                        id="pipeline-enh-gate",
                        options=[{"label": " Skip enhancement if multi-speaker", "value": "on"}],
                        value=[],
                        inline=True,
                        className="mt-1",
                    ),
                ], width=5),
            ], className="mb-3"),

            # Visual pipeline flow (updated by callback)
            html.Div(
                id="pipeline-flow-display",
                className="text-center mb-3",
                style={"fontSize": "0.9rem"},
            ),

            # RUN button
            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        "RUN PIPELINE", id="btn-run-pipeline",
                        color="success", size="lg", className="w-100",
                    ),
                    width={"size": 6, "offset": 3},
                ),
            ]),
        ], className="py-2"),
    ], className="mb-3")


def build_flow_badge(name, is_skip=False):
    """Build a single badge for the pipeline flow display."""
    if is_skip:
        return dbc.Badge(
            "skip", color="light", text_color="secondary",
            className="px-2 py-1",
            style={"border": "1px dashed #ccc"},
        )
    return dbc.Badge(name, color="primary", className="px-2 py-1")


def build_pipeline_flow(ssl_val, bf_val, enh_val, asr_val):
    """Build the visual pipeline flow from dropdown values."""
    stages = [
        ("SSL", ssl_val),
        ("BF", bf_val),
        ("Enh", enh_val),
        ("ASR", asr_val),
    ]

    elements = []
    for i, (stage_label, val) in enumerate(stages):
        name = DISPLAY_NAMES.get(val, val)
        is_skip = (val == "none")

        if i > 0:
            elements.append(html.Span(" → ", className="mx-1 text-muted fw-bold"))

        elements.append(build_flow_badge(name, is_skip))

    return html.Div(elements, className="d-flex align-items-center justify-content-center")
