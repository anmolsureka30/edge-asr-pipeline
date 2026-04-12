"""
Results panel: per-stage collapsible result cards with plots and metrics.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


# Rating thresholds from DATASETS_AND_METRICS.md Section 4
RATING_THRESHOLDS = {
    "angular_error_deg": [(5, "Excellent"), (10, "Good"), (20, "Acceptable"), (999, "Poor")],
    "pesq": [(3.5, "Excellent"), (2.5, "Good"), (2.0, "Acceptable"), (-1, "Poor")],
    "stoi": [(0.9, "Excellent"), (0.8, "Good"), (0.7, "Acceptable"), (-1, "Poor")],
    "si_sdr_db": [(15, "Excellent"), (10, "Good"), (5, "Acceptable"), (-99, "Poor")],
    "wer": [(0.05, "Excellent"), (0.15, "Good"), (0.30, "Acceptable"), (99, "Poor")],
    "cer": [(0.03, "Excellent"), (0.10, "Good"), (0.20, "Acceptable"), (99, "Poor")],
    "rtf": [(0.5, "Excellent"), (1.0, "Good"), (2.0, "Acceptable"), (99, "Poor")],
}

RATING_COLORS = {
    "Excellent": "success",
    "Good": "primary",
    "Acceptable": "warning",
    "Poor": "danger",
}


def get_rating(metric_name, value):
    """Get rating label and color for a metric value."""
    thresholds = RATING_THRESHOLDS.get(metric_name)
    if not thresholds:
        return "", "secondary"

    # For "lower is better" metrics (angular error, wer, cer, rtf)
    lower_better = metric_name in ("angular_error_deg", "wer", "cer", "rtf")

    if lower_better:
        for thresh, label in thresholds:
            if value <= thresh:
                return label, RATING_COLORS[label]
        return "Poor", "danger"
    else:
        # Higher is better (pesq, stoi, si_sdr)
        for thresh, label in reversed(thresholds):
            if value >= thresh:
                return label, RATING_COLORS[label]
        return "Poor", "danger"


def build_metrics_table(all_metrics: dict) -> dbc.Table:
    """Build a comprehensive metrics table with color-coded ratings.

    Args:
        all_metrics: flat dict of metric_name: value from the evaluator.
    """
    rows = []

    # SSL
    if "ssl_angular_error" in all_metrics:
        v = all_metrics["ssl_angular_error"]
        label, color = get_rating("angular_error_deg", v)
        rows.append(("SSL", "Angular Error", f"{v:.1f} deg", label, color))

    # Beamforming
    if "bf_si_sdr" in all_metrics:
        v = all_metrics["bf_si_sdr"]
        rows.append(("Beamforming", "SI-SDR", f"{v:.1f} dB", "", "secondary"))
    if "bf_si_sdr_improvement" in all_metrics:
        v = all_metrics["bf_si_sdr_improvement"]
        sign = "+" if v >= 0 else ""
        rows.append(("Beamforming", "SI-SDR Improvement", f"{sign}{v:.1f} dB", "", "secondary"))
    if "bf_output_snr" in all_metrics:
        v = all_metrics["bf_output_snr"]
        rows.append(("Beamforming", "Output SNR", f"{v:.1f} dB", "", "secondary"))

    # Enhancement
    if "enh_pesq" in all_metrics:
        v = all_metrics["enh_pesq"]
        label, color = get_rating("pesq", v)
        rows.append(("Enhancement", "PESQ", f"{v:.2f}", label, color))
    if "enh_stoi" in all_metrics:
        v = all_metrics["enh_stoi"]
        label, color = get_rating("stoi", v)
        rows.append(("Enhancement", "STOI", f"{v:.3f}", label, color))
    if "enh_si_sdr" in all_metrics:
        v = all_metrics["enh_si_sdr"]
        rows.append(("Enhancement", "SI-SDR", f"{v:.1f} dB", "", "secondary"))
    if "enh_si_sdr_improvement" in all_metrics:
        v = all_metrics["enh_si_sdr_improvement"]
        sign = "+" if v >= 0 else ""
        rows.append(("Enhancement", "SI-SDR Improvement", f"{sign}{v:.1f} dB", "", "secondary"))

    # ASR
    if "asr_wer" in all_metrics:
        v = all_metrics["asr_wer"]
        label, color = get_rating("wer", v)
        rows.append(("ASR", "WER", f"{v*100:.1f}%", label, color))
    if "asr_cer" in all_metrics:
        v = all_metrics["asr_cer"]
        label, color = get_rating("cer", v)
        rows.append(("ASR", "CER", f"{v*100:.1f}%", label, color))
    if "asr_insertions" in all_metrics:
        i = all_metrics.get("asr_insertions", 0)
        d = all_metrics.get("asr_deletions", 0)
        s = all_metrics.get("asr_substitutions", 0)
        rows.append(("ASR", "Ins / Del / Sub", f"{i} / {d} / {s}", "", "secondary"))

    # System / Compute
    if "total_latency_ms" in all_metrics:
        v = all_metrics["total_latency_ms"]
        rows.append(("System", "Total Latency", f"{v:.0f} ms", "", "secondary"))
    if "total_rtf" in all_metrics:
        v = all_metrics["total_rtf"]
        label, color = get_rating("rtf", v)
        rows.append(("System", "RTF", f"{v:.3f}", label, color))
    if "latency_breakdown" in all_metrics:
        rows.append(("System", "Per-Module Latency", all_metrics["latency_breakdown"], "", "secondary"))
    if "total_macs" in all_metrics:
        v = all_metrics["total_macs"]
        rows.append(("Compute", "Total MACs", all_metrics.get("total_macs_formatted", f"{v:,}"), "", "secondary"))
    if "total_energy_mj" in all_metrics:
        v = all_metrics["total_energy_mj"]
        rows.append(("Compute", "Energy (est.)", f"{v:.1f} mJ", "", "secondary"))
    if "total_parameters" in all_metrics:
        v = all_metrics["total_parameters"]
        if v > 1e6:
            rows.append(("Compute", "Parameters", f"{v/1e6:.1f}M", "", "secondary"))
        else:
            rows.append(("Compute", "Parameters", f"{v:,}", "", "secondary"))
    if "peak_memory_kb" in all_metrics:
        v = all_metrics["peak_memory_kb"]
        if v > 1024:
            rows.append(("Compute", "Peak Memory", f"{v/1024:.1f} MB", "", "secondary"))
        else:
            rows.append(("Compute", "Peak Memory", f"{v:.0f} KB", "", "secondary"))

    if not rows:
        return html.Div("No metrics available.", className="text-muted small")

    # Build table
    table_rows = []
    prev_stage = ""
    for stage, metric, value, rating, color in rows:
        stage_cell = html.Td(html.Strong(stage), className="small") if stage != prev_stage else html.Td("")
        rating_badge = dbc.Badge(rating, color=color, className="ms-1") if rating else ""
        table_rows.append(html.Tr([
            stage_cell,
            html.Td(metric, className="small"),
            html.Td(value, className="small fw-bold"),
            html.Td(rating_badge),
        ]))
        prev_stage = stage

    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Stage", style={"width": "20%"}),
            html.Th("Metric", style={"width": "30%"}),
            html.Th("Value", style={"width": "25%"}),
            html.Th("Rating", style={"width": "25%"}),
        ]))] + [html.Tbody(table_rows)],
        bordered=True,
        hover=True,
        size="sm",
        className="mb-0",
    )


def create_results_panel():
    """Create the results section (stage cards + remark)."""
    return html.Div([
        # Status
        dbc.Alert(
            id="run-status", children="Configure scene and click RUN PIPELINE.",
            color="light", className="py-2 text-center",
        ),

        # Stage results container
        html.Div(id="stage-results-container"),

        # Summary + remark
        dbc.Card([
            dbc.CardHeader("Run Summary", className="py-2"),
            dbc.CardBody([
                html.Div(id="run-summary-content"),
                dbc.Row([
                    dbc.Col(
                        dbc.Input(id="remark-input", placeholder="Add a remark for this run...", type="text"),
                        width=9,
                    ),
                    dbc.Col(
                        dbc.Button("Save Remark", id="btn-save-remark", color="primary", size="sm", className="w-100"),
                        width=3,
                    ),
                ], className="mt-2"),
            ], className="py-2"),
        ], id="summary-card", style={"display": "none"}),
    ])


def build_stage_card(
    stage_name, method_name, latency_ms, metrics,
    waveform_fig=None, spectrogram_fig=None, scalogram_fig=None, extra_content=None,
):
    """Build a collapsible card for one pipeline stage result."""
    metric_badges = []
    for k, v in metrics.items():
        if isinstance(v, float):
            metric_badges.append(dbc.Badge(f"{k}: {v:.3f}", color="info", className="me-1"))

    stage_id = stage_name.lower().replace(" ", "-")

    header = dbc.CardHeader(
        dbc.Row([
            dbc.Col(html.Span([html.Strong(f"{stage_name}: "), html.Span(method_name)]), width="auto"),
            dbc.Col(html.Div(metric_badges), className="text-end"),
            dbc.Col(html.Small(f"{latency_ms:.1f}ms", className="text-muted"), width="auto"),
            dbc.Col(
                dbc.Button("Details", id=f"btn-{stage_id}", color="link", size="sm", className="p-0"),
                width="auto",
            ),
        ], align="center", className="g-2"),
        className="py-2",
    )

    details = []
    for fig in [waveform_fig, spectrogram_fig, scalogram_fig]:
        if fig:
            details.append(dcc.Graph(figure=fig, style={"height": "200px"}))
    if extra_content:
        details.append(extra_content)

    body = dbc.Collapse(
        dbc.CardBody(details, className="py-1") if details else html.Div(),
        id=f"collapse-{stage_id}",
        is_open=False,
    )

    return dbc.Card([header, body], className="mb-1")
