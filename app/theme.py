"""Windows XP window chrome for the app, matching the owner's other projects.

Palette and bevels follow edudataproj / freestuffchicago ``globals.css``: teal desktop,
#C0C0C0 window body with outset bevels, XP-blue gradient title bar, navy section headers,
Tahoma, white inset inputs, XP scrollbars and a taskbar.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

XP_BLUE = "#0054E3"
NAVY = "#000080"
TEAL = "#008080"
SURFACE = "#fcfcfb"  # chart surface inside windows
INK, INK_2 = "#0b0b0b", "#52514e"
FONT = "Tahoma, 'MS Sans Serif', Verdana, sans-serif"

PAGES = [
    ("Home", "/", "🏀 Game"),
    ("Player Explorer", "/Player_Explorer", "📋 Player Explorer"),
    ("Methods", "/Methods", "📄 Methods"),
]

CSS = f"""
<style>
/* ---- Desktop ---- */
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: {TEAL} !important;
  font-family: {FONT} !important;
}}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp div[data-baseweb],
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMetricValue"] {{ font-family: {FONT}; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ---- The window ---- */
[data-testid="stMainBlockContainer"] {{
  background: #C0C0C0;
  border: 2px solid; border-color: #DFDFDF #808080 #808080 #DFDFDF;
  box-shadow: 1px 1px 0 #000;
  padding: 0 18px 22px 18px !important;
  margin-top: 28px; margin-bottom: 60px;
  max-width: 1180px;
}}
.xp-titlebar {{
  background: linear-gradient(90deg, #0997FF 0%, #0053EE 100%);
  margin: 0 -18px 14px -18px; padding: 4px 4px 4px 8px;
  display: flex; align-items: center; justify-content: space-between; user-select: none;
}}
.xp-title {{ color: #fff; font-weight: bold; font-size: 13px; text-shadow: 1px 1px 0 rgba(0,0,0,.4); }}
.xp-controls {{ display: flex; gap: 2px; }}
.xp-controls span {{
  width: 18px; height: 16px; background: #C0C0C0; color: #000; font-size: 9px; font-weight: bold;
  border: 1px solid; border-color: #DFDFDF #000 #000 #DFDFDF;
  display: flex; align-items: center; justify-content: center; line-height: 1;
}}
.xp-section {{
  background: {NAVY}; color: #fff; padding: 5px 8px; font-weight: bold; font-size: 12px;
  margin: 18px 0 8px 0;
}}
.xp-panel {{
  border: 2px inset #808080; background: #fff; padding: 8px 10px; font-size: 13px; margin: 4px 0;
}}
.xp-note {{ color: {INK_2}; font-size: 11px; }}
h1, h2, h3 {{ font-family: {FONT} !important; color: #000 !important; }}
h1 {{ font-size: 22px !important; }}
.stMarkdown p, .stMarkdown li {{ font-size: 13px; }}

/* ---- Buttons ---- */
.stButton button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-segmented_control"],
[data-testid="stBaseButton-segmented_controlActive"] {{
  background: #C0C0C0 !important; color: #000 !important; border-radius: 0 !important;
  border: 2px outset #DFDFDF !important; font-weight: bold !important; font-size: 12px !important;
  box-shadow: none !important;
}}
[data-testid="stBaseButton-segmented_controlActive"] {{
  border: 2px inset #808080 !important; background: #B0B0B0 !important;
}}
.stButton [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] {{
  background: {XP_BLUE} !important; color: #fff !important; border-radius: 0 !important;
  border: 2px outset #4d8fff !important; font-weight: bold !important;
  text-shadow: 1px 1px 0 rgba(0,0,0,.3);
}}
.stButton button:active, [data-testid="stBaseButton-primary"]:active {{ border-style: inset !important; }}
button:disabled {{ opacity: .55; }}

/* ---- Inputs ---- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, input {{
  border-radius: 0 !important; background: #fff !important;
  border: 2px inset #808080 !important;
}}
[data-baseweb="tag"] {{
  border-radius: 0 !important; background: #ECE9D8 !important; color: #000 !important;
  border: 1px solid #808080 !important;
}}
[data-baseweb="tag"] span {{ color: #000 !important; }}
[data-baseweb="popover"] li {{ font-size: 12px; }}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [aria-selected="true"] {{ background: #316AC5 !important; color: #fff !important; }}
label, [data-testid="stWidgetLabel"] p {{ font-weight: bold; font-size: 12px !important; }}

/* ---- Metrics, tables, alerts as inset white panels ---- */
[data-testid="stMetric"] {{ border: 2px inset #808080; background: #fff; padding: 6px 10px; }}
[data-testid="stMetricLabel"] p {{ font-size: 11px !important; font-weight: bold; color: {INK_2}; }}
[data-testid="stMetricValue"] {{ font-size: 26px !important; color: #000; }}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{ border: 2px inset #808080; background: {SURFACE}; }}
[data-testid="stAlert"] > div {{ border-radius: 0 !important; border: 2px solid; border-color: #DFDFDF #808080 #808080 #DFDFDF; }}
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentWarning"] {{ color: #000; }}
[data-testid="stExpander"] details {{ border-radius: 0; border: 2px groove #DFDFDF; background: #C0C0C0; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {INK_2} !important; font-size: 11px !important; }}

/* ---- Sidebar as an XP explorer pane ---- */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #7BA2E7 0%, #6375D6 100%); border-right: 2px solid #0053EE; }}
[data-testid="stSidebarNav"] a {{ border-radius: 0; }}
[data-testid="stSidebarNav"] a span {{ color: #fff !important; font-weight: bold; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(255,255,255,.25); }}

/* ---- Taskbar ---- */
.xp-taskbar {{
  position: fixed; bottom: 0; left: 0; right: 0; height: 34px; z-index: 1000;
  background: linear-gradient(180deg, #3168D5 0%, #245DDB 50%, #1941A5 100%);
  border-top: 1px solid #0831D9; display: flex; align-items: center; gap: 6px; padding: 0 0 0 0;
}}
.xp-start {{
  height: 34px; padding: 0 18px 0 12px; display: flex; align-items: center; gap: 6px;
  background: linear-gradient(180deg, #3C9A3C 0%, #76B541 45%, #3C8A2C 100%);
  color: #fff; font-weight: bold; font-style: italic; font-size: 15px;
  border-radius: 0 12px 12px 0; text-shadow: 1px 1px 1px rgba(0,0,0,.5); text-decoration: none !important;
}}
.xp-task {{
  height: 24px; padding: 0 12px; display: flex; align-items: center; gap: 5px; min-width: 150px;
  background: #3C81F3; border: 1px solid #1941A5; border-radius: 3px;
  color: #fff !important; font-size: 11px; font-weight: bold; text-decoration: none !important;
}}
.xp-task.active {{ background: #1E52B7; box-shadow: inset 1px 1px 2px rgba(0,0,0,.5); }}
.xp-tray {{
  margin-left: auto; height: 34px; padding: 0 14px; display: flex; align-items: center;
  background: #0F8FEB; border-left: 1px solid #1941A5; color: #fff; font-size: 11px;
}}

/* ---- XP scrollbars ---- */
::-webkit-scrollbar {{ width: 16px; height: 16px; }}
::-webkit-scrollbar-track {{ background: #ECE9D8; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient(90deg, #C0C7D8 0%, #A0A7B8 100%); border: 1px solid #808080; }}
::-webkit-scrollbar-button {{ height: 16px; background: linear-gradient(180deg, #ECE9D8 0%, #C0C7D8 100%); border: 1px solid #808080; }}
</style>
"""


def window(page: str, title: str) -> None:
    """Draw the desktop, this page's title bar and the taskbar. Call once at the top of a page."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="xp-titlebar"><span class="xp-title">🏀 {html.escape(title)}</span>'
        '<span class="xp-controls"><span>_</span><span>□</span><span>✕</span></span></div>',
        unsafe_allow_html=True,
    )
    tasks = "".join(
        f'<a class="xp-task{" active" if name == page else ""}" href="{href}" target="_self">{label}</a>'
        for name, href, label in PAGES
    )
    st.markdown(
        f'<div class="xp-taskbar"><a class="xp-start" href="/" target="_self">⊞ start</a>{tasks}'
        '<span class="xp-tray">Five Under the Cap · 2026-27</span></div>',
        unsafe_allow_html=True,
    )


def section(text: str) -> None:
    """Navy section header bar."""
    st.markdown(f'<div class="xp-section">{html.escape(text)}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Plotly styling to sit inside an XP window: light surface, Tahoma, recessive axes."""
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK),
        title_font=dict(family=FONT, size=13, color=INK),
        hoverlabel=dict(font_family=FONT, bgcolor="#FFFFE1", bordercolor="#000", font_color="#000"),
    )
    fig.update_xaxes(
        gridcolor="#e6e5e1", linecolor="#999", tickfont=dict(color=INK_2), zerolinecolor="#999"
    )
    fig.update_yaxes(
        gridcolor="#e6e5e1", linecolor="#999", tickfont=dict(color=INK_2), zerolinecolor="#999"
    )
    if height:
        fig.update_layout(height=height)
    return fig
