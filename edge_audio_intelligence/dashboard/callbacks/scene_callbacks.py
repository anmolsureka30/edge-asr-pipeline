"""
Scene callbacks: preset loading, room view updates, source management,
collapsible sections, click-to-place sources, pipeline flow.
"""

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, no_update

from ..components.scene_setup import SCENE_PRESETS, create_source_card
from ..components.pipeline_config import build_pipeline_flow

logger = logging.getLogger(__name__)


def _make_collapse_callback(app, section_id):
    """Create a collapse toggle callback for a specific section."""
    @app.callback(
        [
            Output(f"{section_id}-collapse", "is_open"),
            Output(f"{section_id}-chevron", "children"),
        ],
        Input(f"{section_id}-header", "n_clicks"),
        State(f"{section_id}-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle(n, is_open):
        if n:
            new_state = not is_open
            return new_state, "▼" if new_state else "▶"
        return is_open, "▼" if is_open else "▶"


def register_scene_callbacks(app):
    """Register all scene-related callbacks."""

    # --- Collapsible section toggles (each registered separately) ---
    _make_collapse_callback(app, "room-params")
    _make_collapse_callback(app, "sources")
    _make_collapse_callback(app, "mic-array")
    _make_collapse_callback(app, "saved-setups")

    # --- Preset dropdown -> fill sliders ---
    @app.callback(
        [
            Output("room-width", "value", allow_duplicate=True),
            Output("room-depth", "value", allow_duplicate=True),
            Output("room-height", "value", allow_duplicate=True),
            Output("rt60-slider", "value", allow_duplicate=True),
            Output("snr-slider", "value", allow_duplicate=True),
            Output("duration-slider", "value", allow_duplicate=True),
            Output("scene-store", "data", allow_duplicate=True),
            Output("source-list-container", "children", allow_duplicate=True),
        ],
        Input("preset-dropdown", "value"),
        prevent_initial_call=True,
    )
    def load_preset(preset_key):
        if not preset_key or preset_key not in SCENE_PRESETS:
            return (no_update,) * 8
        p = SCENE_PRESETS[preset_key]

        # Build scene_data with preset sources
        sources = p.get("sources", [
            {"role": "target_speaker", "signal": "librispeech",
             "x": 2.0, "y": 2.5, "z": 1.5, "libri_selection": ""},
        ])
        scene_data = {"sources": sources, "mic_positions": []}

        # Build source cards
        cards = [create_source_card(i) for i in range(len(sources))]

        return (
            p["room_dim"][0], p["room_dim"][1], p["room_dim"][2],
            p["rt60"], p["snr_db"], p["duration_s"],
            scene_data, cards,
        )

    # --- Room view rendering ---
    @app.callback(
        Output("room-graph", "figure"),
        [
            Input("room-width", "value"),
            Input("room-depth", "value"),
            Input("scene-store", "data"),
            Input("click-target-source", "value"),
        ],
    )
    def update_room_view(width, depth, scene_data, selected_source):
        width = width or 6
        depth = depth or 5
        fig = go.Figure()

        # Room boundary
        fig.add_shape(
            type="rect", x0=0, y0=0, x1=width, y1=depth,
            fillcolor="rgba(250, 248, 244, 0.6)",
            line=dict(color="#444", width=2),
        )

        # Grid lines (1m)
        for x_val in np.arange(1, width, 1):
            fig.add_shape(
                type="line", x0=x_val, y0=0, x1=x_val, y1=depth,
                line=dict(color="rgba(0,0,0,0.05)", width=0.5, dash="dot"),
            )
        for y_val in np.arange(1, depth, 1):
            fig.add_shape(
                type="line", x0=0, y0=y_val, x1=width, y1=y_val,
                line=dict(color="rgba(0,0,0,0.05)", width=0.5, dash="dot"),
            )

        # Source colors and symbols
        role_styles = {
            "target_speaker":      {"color": "#E53935", "sym": "star",    "label": "Target Speaker"},
            "interfering_speaker": {"color": "#FF9800", "sym": "diamond", "label": "Interferer"},
            "background_music":    {"color": "#9C27B0", "sym": "square",  "label": "Music"},
            "ambient_noise":       {"color": "#78909C", "sym": "hexagon", "label": "Noise"},
        }

        # Plot sources — always visible with clear markers
        if scene_data and "sources" in scene_data:
            for i, src in enumerate(scene_data["sources"]):
                sx = src.get("x", 2.0 + i * 1.5)
                sy = src.get("y", 3.5)
                role = src.get("role", "target_speaker")
                st = role_styles.get(role, role_styles["target_speaker"])
                vol = src.get("volume", 1.0)
                is_sel = (selected_source == i)

                fig.add_trace(go.Scatter(
                    x=[sx], y=[sy],
                    mode="markers+text",
                    marker=dict(
                        size=20 if is_sel else 16,
                        color=st["color"],
                        symbol=st["sym"],
                        line=dict(
                            color="#000" if is_sel else "white",
                            width=2.5 if is_sel else 1,
                        ),
                    ),
                    text=[f"S{i}"],
                    textposition="top center",
                    textfont=dict(size=11, color=st["color"]),
                    name=f"S{i}: {st['label']}",
                    hovertext=(
                        f"<b>S{i}: {st['label']}</b><br>"
                        f"({sx:.1f}, {sy:.1f})<br>"
                        f"Volume: {vol:.1f}"
                    ),
                    hoverinfo="text",
                ))

        # Plot mic array — mics are mm apart so they appear as a single cluster
        # Show as one group marker at center + a label showing mic count
        if scene_data and "mic_positions" in scene_data:
            mics = scene_data["mic_positions"]
            if mics:
                mic_xs = [m[0] for m in mics]
                mic_ys = [m[1] for m in mics]
                cx = np.mean(mic_xs)
                cy = np.mean(mic_ys)

                # Single mic array marker (mics are mm apart, invisible at room scale)
                fig.add_trace(go.Scatter(
                    x=[cx], y=[cy],
                    mode="markers+text",
                    marker=dict(
                        size=22, color="#1565C0", symbol="circle",
                        line=dict(color="white", width=2),
                    ),
                    text=[f"Mic Array ({len(mics)}x)"],
                    textposition="bottom center",
                    textfont=dict(size=10, color="#1565C0"),
                    name=f"Mic Array ({len(mics)} mics)",
                    hovertext=(
                        f"<b>Microphone Array</b><br>"
                        f"{len(mics)} mics at ({cx:.2f}, {cy:.2f})<br>"
                        f"Spacing: {abs(mic_xs[-1]-mic_xs[0])*1000:.0f}mm total span"
                    ),
                    hoverinfo="text",
                ))

                # Small highlight circle around mic array
                r = 0.2
                fig.add_shape(
                    type="circle",
                    x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                    fillcolor="rgba(21, 101, 192, 0.08)",
                    line=dict(color="rgba(21, 101, 192, 0.25)", width=1, dash="dash"),
                )

        fig.update_layout(
            xaxis=dict(
                range=[-0.3, width + 0.3], title="x (m)",
                scaleanchor="y", zeroline=False,
            ),
            yaxis=dict(
                range=[-0.3, depth + 0.3], title="y (m)",
                zeroline=False,
            ),
            showlegend=True,
            legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
            margin=dict(l=40, r=10, t=10, b=50),
            plot_bgcolor="white",
            height=420,
            clickmode="event",
            hoverlabel=dict(bgcolor="white", font_size=12),
        )
        return fig

    # --- Source management: add source + rebuild cards + update mic positions ---
    @app.callback(
        [
            Output("scene-store", "data", allow_duplicate=True),
            Output("source-list-container", "children", allow_duplicate=True),
            Output("click-target-source", "options"),
            Output("click-target-source", "value"),
        ],
        [
            Input("btn-add-source", "n_clicks"),
            Input("mic-array-dropdown", "value"),
            Input("room-width", "value"),
            Input("room-depth", "value"),
            Input("room-graph", "clickData"),
        ],
        [
            State("scene-store", "data"),
            State("click-target-source", "value"),
        ],
        prevent_initial_call="initial_duplicate",
    )
    def manage_sources_and_clicks(add_clicks, mic_type, room_w, room_h, click_data, scene_data, selected_source):
        if scene_data is None:
            scene_data = {"sources": [], "mic_positions": []}

        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        # Add source on button click
        if "btn-add-source" in triggered:
            n = len(scene_data.get("sources", []))
            scene_data.setdefault("sources", []).append({
                "role": "interfering_speaker" if n > 0 else "target_speaker",
                "signal": "librispeech" if n == 0 else "sine",
                "x": 2.0 + n * 1.5,
                "y": 3.5,
                "z": 1.5,
                "volume": 1.0,
                "libri_selection": "",
            })

        # Click-to-place: move the SELECTED source to click point
        if "room-graph.clickData" in triggered and click_data:
            point = click_data["points"][0]
            room_w_val = room_w or 6
            room_h_val = room_h or 5
            x = max(0.2, min(point.get("x", 2), room_w_val - 0.2))
            y = max(0.2, min(point.get("y", 3), room_h_val - 0.2))

            sources = scene_data.get("sources", [])
            target_idx = selected_source if selected_source is not None else len(sources) - 1
            if 0 <= target_idx < len(sources):
                sources[target_idx]["x"] = round(x, 1)
                sources[target_idx]["y"] = round(y, 1)
                scene_data["sources"] = sources

        # Ensure at least one source
        if not scene_data.get("sources"):
            scene_data["sources"] = [{
                "role": "target_speaker",
                "signal": "librispeech",
                "x": 2.0, "y": 3.5, "z": 1.5,
                "volume": 1.0,
                "libri_selection": "",
            }]

        # Compute mic positions
        room_w = room_w or 6
        room_h = room_h or 5
        center = [room_w / 2, room_h / 2]

        from ...testbench.scene import MicArrayConfig
        if mic_type == "phone_2":
            cfg = MicArrayConfig.phone_2mic(center=center, height=1.2)
        elif mic_type == "smart_speaker_4":
            cfg = MicArrayConfig.smart_speaker_4mic(center=center, height=1.2)
        else:
            cfg = MicArrayConfig.linear_array(n_mics=4, center=center, height=1.2)

        scene_data["mic_positions"] = cfg.positions

        # Build source cards
        n_src = len(scene_data["sources"])
        cards = [create_source_card(i) for i in range(n_src)]

        # Source selector options for click-to-place
        src_options = [{"label": f"S{i}", "value": i} for i in range(n_src)]
        current_sel = selected_source if selected_source is not None and selected_source < n_src else n_src - 1

        return scene_data, cards, src_options, current_sel

    # --- Pipeline preset auto-fill ---
    @app.callback(
        [
            Output("pipeline-ssl", "value", allow_duplicate=True),
            Output("pipeline-bf", "value", allow_duplicate=True),
            Output("pipeline-enh", "value", allow_duplicate=True),
            Output("pipeline-asr", "value", allow_duplicate=True),
            Output("pipeline-vad", "value", allow_duplicate=True),
            Output("pipeline-enh-gate", "value", allow_duplicate=True),
            Output("pipeline-preset-description", "children"),
        ],
        Input("pipeline-preset", "value"),
        prevent_initial_call=True,
    )
    def apply_pipeline_preset(preset_key):
        from ..components.pipeline_config import PIPELINE_PRESETS
        if not preset_key or preset_key not in PIPELINE_PRESETS:
            return no_update, no_update, no_update, no_update, no_update, no_update, ""

        p = PIPELINE_PRESETS[preset_key]
        enh_gate_val = ["on"] if p.get("enh_gate", False) else []
        return (
            p["ssl"], p["bf"], p["enh"], p["asr"],
            p["vad"], enh_gate_val,
            p.get("description", ""),
        )

    # --- Pipeline flow diagram ---
    @app.callback(
        Output("pipeline-flow-display", "children"),
        [
            Input("pipeline-ssl", "value"),
            Input("pipeline-bf", "value"),
            Input("pipeline-enh", "value"),
            Input("pipeline-asr", "value"),
            Input("pipeline-vad", "value"),
            Input("pipeline-enh-gate", "value"),
        ],
    )
    def update_pipeline_flow(ssl_val, bf_val, enh_val, asr_val, vad_val, enh_gate_val):
        from ..components.pipeline_config import DISPLAY_NAMES, build_flow_badge
        import dash_bootstrap_components as dbc
        from dash import html

        elements = []
        enh_gated = "on" in (enh_gate_val or [])

        # VAD badge (if active)
        if vad_val and vad_val != "none":
            vad_name = DISPLAY_NAMES.get(vad_val, vad_val)
            elements.append(dbc.Badge(
                f"VAD: {vad_name}", color="secondary",
                className="px-2 py-1",
                style={"backgroundColor": "#7B1FA2"},  # purple
            ))
            elements.append(html.Span(" → ", className="mx-1 text-muted"))

        # Core pipeline stages
        stages = [
            ("SSL", ssl_val, "#E8913A"),
            ("BF", bf_val, "#3A7EE8"),
            ("Enh", enh_val, "#4CAF50"),
            ("ASR", asr_val, "#E53935"),
        ]

        for i, (label, val, color) in enumerate(stages):
            if val and val != "none":
                name = DISPLAY_NAMES.get(val, val)
                suffix = ""
                if label == "Enh" and enh_gated:
                    suffix = " (gated)"
                if label == "BF" and val == "mvdr" and vad_val and vad_val != "none":
                    suffix = " (Φ_nn)"
                elements.append(dbc.Badge(
                    f"{name}{suffix}", className="px-2 py-1",
                    style={"backgroundColor": color},
                ))
            else:
                elements.append(dbc.Badge(
                    "skip", color="light", text_color="secondary",
                    className="px-2 py-1",
                    style={"border": "1px dashed #ccc"},
                ))

            if i < len(stages) - 1:
                elements.append(html.Span(" → ", className="mx-1 text-muted"))

        return html.Div(elements, className="d-flex align-items-center justify-content-center flex-wrap")

    # --- Timeline editor ---
    @app.callback(
        Output("timeline-graph", "figure"),
        [
            Input("scene-store", "data"),
            Input("duration-slider", "value"),
        ],
    )
    def update_timeline(scene_data, duration):
        from ..components.timeline_editor import build_timeline_figure
        sources = scene_data.get("sources", []) if scene_data else []
        dur = duration or 15.0
        return build_timeline_figure(sources, dur)

    # --- Timeline drag -> update scene-store onset/offset ---
    @app.callback(
        Output("scene-store", "data", allow_duplicate=True),
        Input("timeline-graph", "relayoutData"),
        State("scene-store", "data"),
        prevent_initial_call=True,
    )
    def timeline_drag_update(relayout_data, scene_data):
        if not relayout_data or not scene_data:
            return no_update

        sources = scene_data.get("sources", [])
        updated = False

        # Plotly shape drag events look like: shapes[0].x0, shapes[0].x1, etc.
        for key, val in relayout_data.items():
            if key.startswith("shapes[") and (".x0" in key or ".x1" in key):
                # Extract shape index
                idx = int(key.split("[")[1].split("]")[0])
                if idx < len(sources):
                    if ".x0" in key:
                        sources[idx]["onset_s"] = round(max(0, float(val)), 1)
                        updated = True
                    elif ".x1" in key:
                        sources[idx]["offset_s"] = round(max(0, float(val)), 1)
                        updated = True

        if updated:
            scene_data["sources"] = sources
            return scene_data
        return no_update
