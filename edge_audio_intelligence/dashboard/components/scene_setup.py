"""
Scene setup component: room config, source placement, mic array, presets.
All sidebar sections are collapsible.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


# Scene presets matching config/scenes/*.yaml
SCENE_PRESETS = {
    "clean_room": {
        "label": "Clean Room (anechoic, 1 speaker)",
        "room_dim": [4.0, 3.5, 2.8], "rt60": 0.0, "snr_db": 30.0,
        "noise_type": "white", "duration_s": 15.0,
        "sources": [
            {"role": "target_speaker", "signal": "librispeech",
             "x": 2.0, "y": 2.5, "z": 1.5, "libri_selection": "1089-134686-0000",
             "onset_s": 1.0, "offset_s": 14.0},
        ],
    },
    "office_meeting": {
        "label": "Office Meeting (2 speakers, overlapping)",
        "room_dim": [7.0, 6.0, 3.0], "rt60": 0.4, "snr_db": 20.0,
        "noise_type": "white", "duration_s": 15.0,
        "sources": [
            {"role": "target_speaker", "signal": "librispeech",
             "x": 2.5, "y": 4.0, "z": 1.5, "libri_selection": "1089-134686-0000",
             "onset_s": 1.0, "offset_s": 10.0},
            {"role": "interfering_speaker", "signal": "librispeech",
             "x": 5.0, "y": 2.0, "z": 1.5, "libri_selection": "1221-135766-0000",
             "onset_s": 7.0, "offset_s": 14.0},
        ],
    },
    "sequential_speakers": {
        "label": "Sequential Speakers (no overlap, VAD test)",
        "room_dim": [6.0, 5.0, 3.0], "rt60": 0.3, "snr_db": 25.0,
        "noise_type": "white", "duration_s": 15.0,
        "sources": [
            {"role": "target_speaker", "signal": "librispeech",
             "x": 2.0, "y": 3.5, "z": 1.5, "libri_selection": "1089-134686-0000",
             "onset_s": 1.0, "offset_s": 6.0},
            {"role": "interfering_speaker", "signal": "librispeech",
             "x": 4.5, "y": 1.5, "z": 1.5, "libri_selection": "1221-135766-0000",
             "onset_s": 8.0, "offset_s": 14.0},
        ],
    },
    "noisy_cafe": {
        "label": "Noisy Cafe (2 speakers + heavy noise)",
        "room_dim": [10.0, 8.0, 3.5], "rt60": 0.6, "snr_db": 5.0,
        "noise_type": "white", "duration_s": 15.0,
        "sources": [
            {"role": "target_speaker", "signal": "librispeech",
             "x": 3.0, "y": 5.0, "z": 1.5, "libri_selection": "1089-134686-0000",
             "onset_s": 1.0, "offset_s": 14.0},
            {"role": "interfering_speaker", "signal": "librispeech",
             "x": 7.0, "y": 3.0, "z": 1.5, "libri_selection": "1284-1180-0000",
             "onset_s": 3.0, "offset_s": 12.0},
        ],
    },
    "reverberant_hall": {
        "label": "Reverberant Hall (1 speaker, high reverb)",
        "room_dim": [12.0, 9.0, 4.5], "rt60": 1.2, "snr_db": 15.0,
        "noise_type": "white", "duration_s": 15.0,
        "sources": [
            {"role": "target_speaker", "signal": "librispeech",
             "x": 4.0, "y": 6.0, "z": 1.5, "libri_selection": "1089-134691-0000",
             "onset_s": 2.0, "offset_s": 14.0},
        ],
    },
}

SOURCE_ROLES = [
    {"label": "Target Speaker", "value": "target_speaker"},
    {"label": "Interfering Speaker", "value": "interfering_speaker"},
    {"label": "Background Music", "value": "background_music"},
    {"label": "Ambient Noise", "value": "ambient_noise"},
]

MIC_ARRAY_OPTIONS = [
    {"label": "Edge device 4-mic linear (15mm spacing)", "value": "linear_4"},
    {"label": "Smart speaker 4-mic circular (58mm dia)", "value": "smart_speaker_4"},
    {"label": "Phone 2-mic (14mm spacing)", "value": "phone_2"},
]


def _slider(id_, label, min_val, max_val, value, step, marks=None):
    return html.Div([
        html.Label(label, className="fw-bold small mb-0"),
        dcc.Slider(
            id=id_, min=min_val, max=max_val, value=value, step=step,
            marks=marks or {min_val: str(min_val), max_val: str(max_val)},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ], className="mb-2")


def _collapsible_card(card_id, title, body_content, default_open=True):
    """Create a card with a collapsible body toggled by a button header."""
    return dbc.Card([
        dbc.Button(
            html.Div([
                html.Span(title, className="fw-bold small"),
                html.Span(
                    "▼" if default_open else "▶",
                    id=f"{card_id}-chevron",
                    className="float-end",
                ),
            ]),
            id=f"{card_id}-header",
            color="light",
            className="w-100 text-start py-2 px-3 border-0",
            style={"backgroundColor": "#f8f9fa", "borderBottom": "1px solid #dee2e6"},
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.CardBody(body_content, className="py-2"),
            id=f"{card_id}-collapse",
            is_open=default_open,
        ),
    ], className="mb-2")


def create_scene_setup():
    """Create the scene setup panel (left side) with collapsible sections."""
    return html.Div([
        # Scene preset (always visible, small)
        dbc.Card([
            dbc.CardHeader("Scene Preset", className="py-2"),
            dbc.CardBody([
                dcc.Dropdown(
                    id="preset-dropdown",
                    options=[{"label": v["label"], "value": k} for k, v in SCENE_PRESETS.items()],
                    value="clean_room",
                    clearable=False,
                ),
            ], className="py-2"),
        ], className="mb-2"),

        # Room parameters (collapsible, default open)
        _collapsible_card("room-params", "Room Parameters", [
            _slider("room-width", "Width (m)", 3, 20, 6, 0.5),
            _slider("room-depth", "Depth (m)", 3, 20, 5, 0.5),
            _slider("room-height", "Height (m)", 2.5, 6, 3, 0.5),
            _slider("rt60-slider", "RT60 (s)", 0.0, 2.0, 0.0, 0.05),
            _slider("snr-slider", "SNR (dB)", -5, 40, 30, 1),
            _slider("duration-slider", "Simulation length (s)", 1, 30, 15, 1),
            html.Small(
                "3-10s for quick pipeline tests. 10-30s for VAD/streaming tests.",
                className="text-muted",
                style={"fontSize": "0.7rem"},
            ),
        ], default_open=True),

        # Sources (collapsible, default open)
        _collapsible_card("sources", "Sources", [
            html.Div(id="source-list-container"),
            dbc.Button(
                "+ Add Source", id="btn-add-source",
                color="outline-primary", size="sm", className="mt-2 w-100",
            ),
        ], default_open=True),

        # Mic array (collapsible, default collapsed)
        _collapsible_card("mic-array", "Microphone Array", [
            dcc.Dropdown(
                id="mic-array-dropdown",
                options=MIC_ARRAY_OPTIONS,
                value="linear_4",
                clearable=False,
            ),
        ], default_open=False),

        # Save / Load Setup (collapsible, default collapsed)
        _collapsible_card("saved-setups", "Saved Setups", [
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id="load-setup-dropdown",
                        options=[],
                        placeholder="Load saved setup...",
                        style={"fontSize": "0.85rem"},
                    ),
                    width=8,
                ),
                dbc.Col(
                    dbc.Button("Load", id="btn-load-setup", color="primary", size="sm", className="w-100"),
                    width=4,
                ),
            ], className="g-1 mb-2"),
            dbc.Row([
                dbc.Col(
                    dbc.Input(id="setup-name-input", placeholder="Setup name...", type="text", size="sm"),
                    width=8,
                ),
                dbc.Col(
                    dbc.Button("Save", id="btn-save-setup", color="success", size="sm", className="w-100"),
                    width=4,
                ),
            ], className="g-1"),
            dbc.Checkbox(
                id="setup-default-check",
                label="Set as default (auto-load on start)",
                value=True,
                className="mt-2 small",
            ),
            dbc.Button(
                "Delete Selected", id="btn-delete-setup",
                color="outline-danger", size="sm", className="w-100 mt-2",
            ),
            html.Div(id="setup-status", className="small text-muted mt-1"),
        ], default_open=False),
    ])


def create_source_card(index: int):
    """Create a single source configuration card."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Label(f"S{index}", className="fw-bold"), width=1),
                dbc.Col(dcc.Dropdown(
                    id={"type": "source-role", "index": index},
                    options=SOURCE_ROLES,
                    value="target_speaker" if index == 0 else "interfering_speaker",
                    clearable=False,
                    style={"fontSize": "0.85rem"},
                ), width=4),
                dbc.Col(dcc.Dropdown(
                    id={"type": "source-signal", "index": index},
                    options=[
                        {"label": "LibriSpeech", "value": "librispeech"},
                        {"label": "Sine Wave", "value": "sine"},
                        {"label": "White Noise", "value": "white_noise"},
                        {"label": "Chirp", "value": "chirp"},
                        {"label": "Upload File", "value": "upload"},
                    ],
                    value="librispeech" if index == 0 else "sine",
                    clearable=False,
                    style={"fontSize": "0.85rem"},
                ), width=5),
                dbc.Col(dbc.Button(
                    "X", id={"type": "source-delete", "index": index},
                    color="danger", size="sm", outline=True,
                ), width=1),
            ], className="g-1 align-items-center"),

            # LibriSpeech dropdown
            html.Div(
                dcc.Dropdown(
                    id={"type": "source-libri", "index": index},
                    options=[],
                    placeholder="Select utterance...",
                    className="mt-1",
                    style={"fontSize": "0.8rem"},
                    searchable=True,
                ),
                id={"type": "source-libri-container", "index": index},
            ),

            # Upload container
            html.Div(
                dcc.Upload(
                    id={"type": "source-upload", "index": index},
                    children=html.Div(["Drop or ", html.A("select file")]),
                    style={
                        "borderStyle": "dashed", "borderWidth": "1px",
                        "borderRadius": "5px", "textAlign": "center",
                        "padding": "5px", "fontSize": "0.8rem",
                    },
                    accept=".wav,.flac,.mp3",
                ),
                id={"type": "source-upload-container", "index": index},
                style={"display": "none"},
            ),

            # Volume + Position
            dbc.Row([
                dbc.Col([
                    html.Small("Vol:"),
                    dcc.Slider(
                        id={"type": "source-volume", "index": index},
                        min=0.1, max=2.0, value=1.0, step=0.1,
                        marks={0.1: "0.1", 1.0: "1.0", 2.0: "2.0"},
                        tooltip={"placement": "bottom"},
                    ),
                ], width=12),
            ], className="mt-1"),
            dbc.Row([
                dbc.Col([
                    html.Small("x:"),
                    dcc.Input(
                        id={"type": "source-x", "index": index},
                        type="number", value=2.0 + index * 1.5, step=0.5,
                        style={"width": "60px", "fontSize": "0.8rem"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Small("y:"),
                    dcc.Input(
                        id={"type": "source-y", "index": index},
                        type="number", value=3.5, step=0.5,
                        style={"width": "60px", "fontSize": "0.8rem"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Small("z:"),
                    dcc.Input(
                        id={"type": "source-z", "index": index},
                        type="number", value=1.5, step=0.1,
                        style={"width": "60px", "fontSize": "0.8rem"},
                    ),
                ], width=4),
            ], className="mt-1 g-1"),
        ], className="py-1 px-2"),
    ], className="mb-1", style={"fontSize": "0.85rem"})


def create_room_view():
    """Create the 2D room view with source selector for click-to-place."""
    return dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(
                    html.Span("Room View", className="fw-bold"),
                    width="auto",
                ),
                dbc.Col(
                    html.Small("Click on room to move selected source:", className="text-muted"),
                    width="auto",
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="click-target-source",
                        options=[{"label": "S0", "value": 0}],
                        value=0,
                        clearable=False,
                        style={"width": "80px", "fontSize": "0.8rem", "display": "inline-block"},
                    ),
                    width="auto",
                ),
            ], align="center", className="g-2"),
            className="py-2",
        ),
        dbc.CardBody(
            dcc.Graph(
                id="room-graph",
                config={"scrollZoom": False, "displayModeBar": False},
                style={"height": "400px"},
            ),
            className="p-1",
        ),
    ])
