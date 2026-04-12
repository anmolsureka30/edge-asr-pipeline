"""
Dash application factory for the Edge Audio Intelligence Lab.

Usage:
    python -m edge_audio_intelligence.dashboard.app
"""

import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dash import Dash
import dash_bootstrap_components as dbc

from .layout import create_layout
from .callbacks import register_all_callbacks


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="Edge Audio Intelligence Lab",
        suppress_callback_exceptions=True,
        assets_folder=str(Path(__file__).parent / "assets"),
    )

    app.layout = create_layout()
    register_all_callbacks(app)

    return app


def run_server(debug: bool = True, port: int = 8050):
    """Create and run the Dash server."""
    app = create_app()
    print(f"\n  Edge Audio Intelligence Lab")
    print(f"  http://localhost:{port}\n")
    app.run(debug=debug, port=port, host="0.0.0.0")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge Audio Intelligence Lab")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()
    run_server(debug=not args.no_debug, port=args.port)
