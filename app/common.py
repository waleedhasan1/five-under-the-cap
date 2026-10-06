"""Shared app helpers: cached data, formatting, chart colors."""

from __future__ import annotations

import streamlit as st

from fivecap.appdata import AppData, load

# Diverging pair validated for colour-vision deficiency (red = hurts, blue = helps).
NEG, MID, POS = "#e34948", "#f0efec", "#2a78d6"
DIVERGING = [[0.0, NEG], [0.5, MID], [1.0, POS]]
OVER = "#e34948"


@st.cache_resource(show_spinner="Loading player data…")
def data() -> AppData:
    """App data, loaded once per server process."""
    return load()


def money(x: float) -> str:
    """$12.3M style."""
    return f"${x / 1e6:,.1f}M"


def md(text: str) -> str:
    """Escape dollar signs so Streamlit markdown doesn't read "$...$" as LaTeX."""
    return text.replace("$", "\\$")


def money_md(x: float) -> str:
    """``money`` for use inside markdown."""
    return md(money(x))


def signed(x: float) -> str:
    """+1.2 / −1.2 with a real minus sign."""
    return f"{x:+.1f}".replace("-", "−")


def approx_note() -> None:
    st.caption(
        "Predictions are approximations from a lineup model with a measured, limited accuracy. "
        "See **Methods** for how they were validated and what they can't tell you."
    )
