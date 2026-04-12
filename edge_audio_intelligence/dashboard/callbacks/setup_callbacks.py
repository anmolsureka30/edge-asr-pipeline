"""
Setup callbacks: save/load/delete scene+pipeline configurations.

Allows users to save their entire setup (room, sources, mics, pipeline)
and reload it instantly, avoiding reconfiguration between runs.
"""

import logging

from dash import Input, Output, State, no_update, callback_context

from ..state import SetupManager, SavedSetup

logger = logging.getLogger(__name__)

_setup_manager = SetupManager()


def register_setup_callbacks(app):
    """Register all setup save/load callbacks."""

    # Populate the load-setup dropdown on page load and after save/delete
    @app.callback(
        Output("load-setup-dropdown", "options"),
        [
            Input("page-loaded", "data"),
            Input("btn-save-setup", "n_clicks"),
            Input("btn-delete-setup", "n_clicks"),
        ],
    )
    def refresh_setup_list(*_):
        _setup_manager._load()  # Refresh from disk
        options = []
        for s in _setup_manager.setups:
            label = s.name
            if s.is_default:
                label += " (default)"
            options.append({"label": label, "value": s.name})
        return options

    # Save current setup
    @app.callback(
        Output("setup-status", "children"),
        Input("btn-save-setup", "n_clicks"),
        [
            State("setup-name-input", "value"),
            State("setup-default-check", "value"),
            State("room-width", "value"),
            State("room-depth", "value"),
            State("room-height", "value"),
            State("rt60-slider", "value"),
            State("snr-slider", "value"),
            State("duration-slider", "value"),
            State("mic-array-dropdown", "value"),
            State("pipeline-ssl", "value"),
            State("pipeline-bf", "value"),
            State("pipeline-enh", "value"),
            State("pipeline-asr", "value"),
            State("scene-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def save_setup(
        n_clicks, name, is_default,
        room_w, room_d, room_h, rt60, snr, duration,
        mic_type, ssl_val, bf_val, enh_val, asr_val,
        scene_data,
    ):
        if not n_clicks:
            return no_update

        name = (name or "").strip()
        if not name:
            return "Enter a name first."

        setup = SavedSetup(
            name=name,
            room_dim=[room_w or 6, room_d or 5, room_h or 3],
            rt60=rt60 or 0.3,
            snr_db=snr or 15,
            duration_s=duration or 3,
            sources=scene_data.get("sources", []) if scene_data else [],
            mic_array_type=mic_type or "linear_4",
            pipeline_config={
                "ssl": ssl_val or "gcc_phat",
                "beamforming": bf_val or "delay_and_sum",
                "enhancement": enh_val or "spectral_subtraction",
                "asr": asr_val or "none",
            },
            is_default=bool(is_default),
        )

        _setup_manager.save_setup(setup)
        default_str = " (set as default)" if is_default else ""
        logger.info(f"Saved setup: {name}{default_str}")
        return f"Saved '{name}'{default_str}"

    # Load a saved setup -> fill all controls
    @app.callback(
        [
            Output("room-width", "value", allow_duplicate=True),
            Output("room-depth", "value", allow_duplicate=True),
            Output("room-height", "value", allow_duplicate=True),
            Output("rt60-slider", "value", allow_duplicate=True),
            Output("snr-slider", "value", allow_duplicate=True),
            Output("duration-slider", "value", allow_duplicate=True),
            Output("mic-array-dropdown", "value", allow_duplicate=True),
            Output("pipeline-ssl", "value", allow_duplicate=True),
            Output("pipeline-bf", "value", allow_duplicate=True),
            Output("pipeline-enh", "value", allow_duplicate=True),
            Output("pipeline-asr", "value", allow_duplicate=True),
            Output("scene-store", "data", allow_duplicate=True),
            Output("setup-status", "children", allow_duplicate=True),
        ],
        [
            Input("btn-load-setup", "n_clicks"),
            Input("page-loaded", "data"),  # Auto-load default on start
        ],
        State("load-setup-dropdown", "value"),
        prevent_initial_call=True,
    )
    def load_setup(load_clicks, page_loaded, selected_name):
        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        _setup_manager._load()

        setup = None
        if "btn-load-setup" in triggered:
            if not selected_name:
                return tuple([no_update] * 13)
            setup = _setup_manager.get_setup(selected_name)
        elif "page-loaded" in triggered:
            setup = _setup_manager.get_default()

        if setup is None:
            return tuple([no_update] * 13)

        # Build scene store data
        scene_data = {
            "sources": setup.sources if setup.sources else [
                {"role": "target_speaker", "signal": "librispeech",
                 "x": 2.0, "y": 3.5, "z": 1.5}
            ],
        }

        pc = setup.pipeline_config
        msg = f"Loaded setup: '{setup.name}'"
        logger.info(msg)

        return (
            setup.room_dim[0],
            setup.room_dim[1],
            setup.room_dim[2],
            setup.rt60,
            setup.snr_db,
            setup.duration_s,
            setup.mic_array_type,
            pc.get("ssl", "gcc_phat"),
            pc.get("beamforming", "delay_and_sum"),
            pc.get("enhancement", "spectral_subtraction"),
            pc.get("asr", "none"),
            scene_data,
            msg,
        )

    # Delete a setup
    @app.callback(
        Output("setup-status", "children", allow_duplicate=True),
        Input("btn-delete-setup", "n_clicks"),
        State("load-setup-dropdown", "value"),
        prevent_initial_call=True,
    )
    def delete_setup(n_clicks, name):
        if not n_clicks or not name:
            return no_update
        if _setup_manager.delete_setup(name):
            return f"Deleted setup: '{name}'"
        return f"Setup '{name}' not found."
