"""
Timeline Editor: Visual drag-bar interface for source onset/offset timing.

Shows horizontal colored bars per source, draggable to adjust timing.
Used below the room view to configure when each speaker is active.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Colors per source role
SOURCE_COLORS = {
    "target_speaker": "#4CAF50",       # green
    "interfering_speaker": "#FF9800",   # orange
    "background_music": "#9C27B0",      # purple
    "ambient_noise": "#607D8B",         # gray
}

DEFAULT_COLOR = "#2196F3"  # blue


def create_timeline_editor():
    """Create the timeline editor component."""
    return dbc.Card([
        dbc.CardHeader([
            html.Span("Speaker Timeline", className="fw-bold"),
            html.Small(
                " (drag bar edges to adjust onset/offset)",
                className="text-muted ms-2",
            ),
        ], className="py-1"),
        dbc.CardBody([
            dcc.Graph(
                id="timeline-graph",
                config={
                    "editable": True,
                    "editSelection": False,
                    "modeBarButtonsToRemove": [
                        "zoom2d", "pan2d", "select2d", "lasso2d",
                        "zoomIn2d", "zoomOut2d", "autoScale2d",
                        "hoverClosestCartesian", "hoverCompareCartesian",
                        "toggleSpikelines",
                    ],
                },
                style={"height": "120px"},
            ),
        ], className="py-1 px-2"),
    ], className="mb-2")


def build_timeline_figure(sources, duration_s=15.0):
    """Build the Plotly figure with draggable bars per source.

    Args:
        sources: List of source dicts from scene-store, each with:
            - label, role, onset_s, offset_s
        duration_s: Total simulation duration in seconds.

    Returns:
        Plotly figure with one horizontal bar per source.
    """
    fig = go.Figure()

    n_sources = len(sources)
    if n_sources == 0:
        fig.update_layout(
            xaxis=dict(range=[0, duration_s], title="Time (s)"),
            yaxis=dict(visible=False),
            height=80,
            margin=dict(l=60, r=20, t=10, b=30),
        )
        return fig

    shapes = []
    annotations = []

    for i, src in enumerate(sources):
        label = src.get("label", f"S{i}")
        role = src.get("role", "target_speaker")
        onset = src.get("onset_s", 0.0)
        offset = src.get("offset_s", duration_s)
        if offset < 0:
            offset = duration_s

        color = SOURCE_COLORS.get(role, DEFAULT_COLOR)

        # Horizontal bar as a shape (editable for dragging)
        shapes.append(dict(
            type="rect",
            x0=onset, x1=offset,
            y0=i - 0.35, y1=i + 0.35,
            fillcolor=color,
            opacity=0.7,
            line=dict(color=color, width=2),
            editable=True,
            name=label,
        ))

        # Label annotation
        mid_x = (onset + offset) / 2
        annotations.append(dict(
            x=mid_x, y=i,
            text=f"<b>{label}</b> ({onset:.1f}s - {offset:.1f}s)",
            showarrow=False,
            font=dict(size=11, color="white"),
        ))

    # Y-axis labels (source names)
    y_labels = [src.get("label", f"S{i}") for i, src in enumerate(sources)]

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            range=[0, duration_s],
            title="Time (s)",
            gridcolor="#eee",
            dtick=1,
        ),
        yaxis=dict(
            range=[-0.5, n_sources - 0.5],
            tickvals=list(range(n_sources)),
            ticktext=y_labels,
            fixedrange=True,
        ),
        height=max(80, 40 + n_sources * 40),
        margin=dict(l=60, r=20, t=10, b=30),
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig
