"""
Dashboard layout: single-page scrollable with all sections.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

from .components.scene_setup import create_scene_setup, create_room_view
from .components.pipeline_config import create_pipeline_config
from .components.results_panel import create_results_panel
from .components.run_history import create_run_history_panel
from .components.timeline_editor import create_timeline_editor


def _live_output_panel():
    """Live output panel: audio player + transcription + metrics table.

    Sits between room view and detailed pipeline results.
    Shows the most important info at a glance after a run.
    """
    return dbc.Card([
        dbc.CardHeader(
            html.Span("Live Output", className="fw-bold"),
            className="py-2",
        ),
        dbc.CardBody([
            # Audio player (mic perspective)
            dbc.Row([
                dbc.Col([
                    html.Label("Audio", className="fw-bold small"),
                    dcc.Dropdown(
                        id="audio-signal-dropdown",
                        options=[],
                        placeholder="Run pipeline first...",
                        clearable=False,
                        style={"fontSize": "0.85rem"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Div(id="audio-player-container", className="mt-3"),
                ], width=8),
            ], className="mb-2"),

            # Transcription output
            html.Div(id="transcription-output", className="mb-2"),

            # Compact metrics table
            html.Div(id="metrics-table-container"),

        ], className="py-2"),
    ], id="live-output-card", className="mb-3", style={"display": "none"})


def create_layout() -> dbc.Container:
    """Create the full page layout."""
    return dbc.Container([
        # Hidden stores
        dcc.Store(id="scene-store", data={"sources": [], "n_sources": 1}),
        dcc.Store(id="current-run-store", data=None),
        dcc.Store(id="audio-signals-store", data={}),
        dcc.Store(id="page-loaded", data=True),

        # Header
        dbc.Row(
            dbc.Col(
                html.H3(
                    "Edge Audio Intelligence Lab",
                    className="text-primary my-3",
                ),
            ),
        ),

        # Section 1: Scene setup + Room view
        dbc.Row([
            dbc.Col(create_scene_setup(), width=4),
            dbc.Col([
                create_room_view(),
                create_timeline_editor(),
            ], width=8),
        ], className="mb-3"),

        # Section 2: Pipeline config + RUN button
        create_pipeline_config(),

        # Section 3: Live Output (audio + transcription + metrics table)
        _live_output_panel(),

        # Section 4: Detailed pipeline stage results (spinner while running)
        dbc.Spinner(
            create_results_panel(),
            color="primary",
            type="border",
            spinner_style={"width": "3rem", "height": "3rem"},
        ),

        # Section 5: Run history
        create_run_history_panel(),

        # Footer
        html.Hr(),
        html.P(
            "EE678 Wavelets & Multiresolution Signal Processing | IIT Bombay",
            className="text-center text-muted small mb-4",
        ),
    ], fluid=True)
