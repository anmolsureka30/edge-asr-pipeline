"""Register all dashboard callbacks."""

from .scene_callbacks import register_scene_callbacks
from .run_callbacks import register_run_callbacks
from .history_callbacks import register_history_callbacks
from .setup_callbacks import register_setup_callbacks


def register_all_callbacks(app):
    """Register all Dash callbacks with the app."""
    register_scene_callbacks(app)
    register_run_callbacks(app)
    register_history_callbacks(app)
    register_setup_callbacks(app)
