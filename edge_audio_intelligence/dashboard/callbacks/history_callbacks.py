"""
History callbacks: expand/collapse, delete runs, save remarks.
"""

import json
import logging

from dash import Input, Output, State, html, no_update, ALL, MATCH, callback_context
import dash_bootstrap_components as dbc

from ..components.results_panel import build_metrics_table

logger = logging.getLogger(__name__)


def _make_stage_toggle(app, stage_id):
    """Create a collapse toggle for a pipeline stage detail card."""
    @app.callback(
        Output(f"collapse-{stage_id}", "is_open"),
        Input(f"btn-{stage_id}", "n_clicks"),
        State(f"collapse-{stage_id}", "is_open"),
        prevent_initial_call=True,
    )
    def toggle(n, is_open):
        if n:
            return not is_open
        return is_open


def register_history_callbacks(app):
    """Register history management callbacks."""

    # Save remark for current run
    @app.callback(
        Output("run-status", "children", allow_duplicate=True),
        Input("btn-save-remark", "n_clicks"),
        [State("remark-input", "value"), State("current-run-store", "data")],
        prevent_initial_call=True,
    )
    def save_remark(n_clicks, remark, run_id):
        if not n_clicks or not run_id or not remark:
            return no_update

        from .run_callbacks import _run_history
        _run_history.update_remark(run_id, remark)
        return f"Remark saved for run {run_id[-4:]}."

    # Delete a history run
    @app.callback(
        Output("run-history-container", "children", allow_duplicate=True),
        Input({"type": "history-delete", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def delete_run(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        for triggered in ctx.triggered:
            if triggered["value"]:
                prop_id = triggered["prop_id"]
                id_dict = json.loads(prop_id.split(".")[0])
                run_id = id_dict["index"]

                from .run_callbacks import _run_history
                _run_history.delete_run(run_id)

                from ..components.run_history import build_history_card
                return [
                    build_history_card(r, i) for i, r in enumerate(_run_history.runs)
                ]

        return no_update

    # Expand/collapse history details
    @app.callback(
        [
            Output({"type": "history-collapse", "index": MATCH}, "is_open"),
            Output({"type": "history-details", "index": MATCH}, "children"),
        ],
        Input({"type": "history-expand", "index": MATCH}, "n_clicks"),
        [
            State({"type": "history-collapse", "index": MATCH}, "is_open"),
            State({"type": "history-expand", "index": MATCH}, "id"),
        ],
        prevent_initial_call=True,
    )
    def toggle_history_details(n_clicks, is_open, btn_id):
        if not n_clicks:
            return no_update, no_update

        new_state = not is_open
        if not new_state:
            # Collapsing — no need to rebuild content
            return False, no_update

        # Expanding — build the detail content from the run record
        run_id = btn_id["index"]
        from .run_callbacks import _run_history
        record = _run_history.get_run(run_id)

        if record is None:
            return True, html.Div("Run data not found.", className="text-muted small")

        # Build detail content: metrics table + config info
        details = []

        # Metrics table
        if record.metrics:
            details.append(html.H6("Metrics", className="mt-2 mb-1"))
            details.append(build_metrics_table(record.metrics))

        # Pipeline config
        if record.pipeline_config:
            details.append(html.H6("Pipeline", className="mt-2 mb-1"))
            pc = record.pipeline_config
            pipeline_items = []
            for stage, algo in pc.items():
                pipeline_items.append(
                    html.Li(f"{stage.title()}: {algo}", className="small")
                )
            details.append(html.Ul(pipeline_items, className="mb-1"))

        # Scene config
        if record.scene_config:
            details.append(html.H6("Scene", className="mt-2 mb-1"))
            sc = record.scene_config
            scene_info = []
            if "room_dim" in sc:
                d = sc["room_dim"]
                scene_info.append(f"Room: {d[0]}x{d[1]}x{d[2]}m")
            if "rt60" in sc:
                scene_info.append(f"RT60: {sc['rt60']}s")
            if "snr_db" in sc:
                scene_info.append(f"SNR: {sc['snr_db']}dB")
            if "duration_s" in sc:
                scene_info.append(f"Duration: {sc['duration_s']}s")
            details.append(html.P(" | ".join(scene_info), className="small text-muted"))

        # Sources
        if record.sources:
            details.append(html.H6("Sources", className="mt-2 mb-1"))
            for i, src in enumerate(record.sources):
                if isinstance(src, dict):
                    role = src.get("role", src.get("signal_type", "unknown"))
                    pos = f"({src.get('x', src.get('position', ['?'])[0] if isinstance(src.get('position'), list) else '?')})"
                    details.append(html.Small(f"S{i}: {role} {pos}", className="d-block text-muted"))

        return True, html.Div(details)

    # Stage detail collapse toggles (for pipeline result cards)
    _make_stage_toggle(app, "ssl")
    _make_stage_toggle(app, "beamforming")
    _make_stage_toggle(app, "enhancement")
    _make_stage_toggle(app, "asr")
