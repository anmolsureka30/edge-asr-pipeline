"""
Run history component: collapsible timeline cards.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

from ..state import RunRecord


def create_run_history_panel():
    """Create the run history section."""
    return html.Div([
        html.Hr(),
        html.H5("Run History", className="text-muted mb-3"),
        html.Div(id="run-history-container"),
    ])


def build_history_card(record: RunRecord, index: int) -> dbc.Card:
    """Build a single collapsible history card.

    Args:
        record: RunRecord with all run data.
        index: Position in the list.
    """
    # Key metrics
    m = record.metrics
    metric_text = []
    if "ssl_angular_error" in m:
        metric_text.append(f"SSL: {m['ssl_angular_error']:.1f}deg")
    if "enh_pesq" in m:
        metric_text.append(f"PESQ: {m['enh_pesq']:.2f}")
    if "enh_stoi" in m:
        metric_text.append(f"STOI: {m['enh_stoi']:.2f}")
    if "asr_wer" in m:
        metric_text.append(f"WER: {m['asr_wer']:.3f}")
    if "total_latency_ms" in m:
        metric_text.append(f"{m['total_latency_ms']:.0f}ms")

    metrics_str = " | ".join(metric_text) if metric_text else "No metrics"

    # Timestamp
    ts = record.timestamp
    if "T" in ts:
        ts = ts.replace("T", " ")[:16]

    collapse_id = f"history-collapse-{record.run_id}"

    card = dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col([
                    html.Strong(f"Run #{record.run_id[-4:]}", className="me-2"),
                    html.Small(ts, className="text-muted"),
                ], width="auto"),
                dbc.Col([
                    dbc.Button(
                        "X", id={"type": "history-delete", "index": record.run_id},
                        color="danger", size="sm", outline=True,
                        className="py-0 px-1",
                    ),
                ], width="auto", className="ms-auto"),
            ], align="center"),
            className="py-1",
        ),
        dbc.CardBody([
            html.Div([
                html.Small(record.pipeline_summary, className="fw-bold d-block"),
                html.Small(f"{record.scene_summary} | {metrics_str}", className="text-muted d-block"),
            ]),
            html.Div([
                html.Small(
                    f'"{record.remark}"' if record.remark else "",
                    className="fst-italic text-primary",
                ),
            ], className="mt-1") if record.remark else html.Div(),

            dbc.Button(
                "Expand Details",
                id={"type": "history-expand", "index": record.run_id},
                color="link", size="sm", className="p-0 mt-1",
            ),

            dbc.Collapse(
                html.Div(id={"type": "history-details", "index": record.run_id}),
                id={"type": "history-collapse", "index": record.run_id},
                is_open=False,
            ),
        ], className="py-2"),
    ], className="mb-2")

    return card
