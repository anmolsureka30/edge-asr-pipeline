"""
Audio player component: signal selector + HTML5 audio playback.
"""

from dash import html


def create_audio_element(data_uri: str) -> html.Div:
    """Create an HTML5 audio element from a data URI."""
    if not data_uri:
        return html.Div("No audio available.", className="text-muted small")

    return html.Audio(
        src=data_uri,
        controls=True,
        autoPlay=False,
        style={"width": "100%"},
    )
