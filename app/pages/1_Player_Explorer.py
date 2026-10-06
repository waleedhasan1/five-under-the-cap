"""Player explorer: the game pool with ratings, salary, estimated value and surplus."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import DIVERGING, HEADSHOT_COL, approx_note, data, headshot, money_md
from theme import style_fig, window

st.set_page_config(
    page_title="Player Explorer · Five Under the Cap",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)
window("Player Explorer", "Player Explorer — Five Under the Cap")
d = data()
p = d.players.copy()
arch = d.archetype_cols

st.markdown(
    f"All **{len(p)}** players in the game pool: at least "
    f"500 minutes in {d.meta['ratings_through']} and a {d.meta['salary_season']} salary. "
    "**Talent** is the player's Model B effect in points per 100 possessions. **Value** turns "
    "that into dollars over a starter's minutes; **surplus** is value minus salary. Value and "
    "surplus are rough approximations (see Methods)."
)

choice = st.multiselect(
    "Filter by top archetype", [d.archetype_name(a) for a in arch], placeholder="All archetypes"
)
if choice:
    p = p[p["top_archetype"].isin(choice)]

p["Archetype mix"] = p[arch].to_numpy().tolist()
p["headshot"] = p["player_id"].map(headshot)
table = p[
    [
        "headshot",
        "player_name",
        "contract_team",
        "top_archetype",
        "Archetype mix",
        "salary",
        "effect",
        "value",
        "surplus",
        "minutes",
    ]
].rename(
    columns={
        "headshot": " ",
        "player_name": "Player",
        "contract_team": "Team",
        "top_archetype": "Top archetype",
        "salary": "Salary",
        "effect": "Talent",
        "value": "Value (approx.)",
        "surplus": "Surplus (approx.)",
        "minutes": f"Minutes {d.meta['ratings_through']}",
    }
)
st.dataframe(
    table.sort_values("Talent", ascending=False),
    hide_index=True,
    use_container_width=True,
    height=420,
    column_config={
        " ": HEADSHOT_COL,
        "Archetype mix": st.column_config.BarChartColumn(
            help="Membership in each archetype, in order: "
            + ", ".join(d.archetype_name(a) for a in arch),
            y_min=0,
            y_max=1,
        ),
        "Salary": st.column_config.NumberColumn(format="$%,.0f"),
        "Talent": st.column_config.NumberColumn(format="%+.2f"),
        "Value (approx.)": st.column_config.NumberColumn(format="$%,.0f"),
        "Surplus (approx.)": st.column_config.NumberColumn(format="$%,.0f"),
        f"Minutes {d.meta['ratings_through']}": st.column_config.NumberColumn(format="%.0f"),
    },
)

lim = float(p["surplus"].abs().quantile(0.95))
fig = go.Figure(
    go.Scatter(
        x=p["salary"],
        y=p["effect"],
        mode="markers",
        marker=dict(
            size=9,
            color=p["surplus"],
            colorscale=DIVERGING,
            cmin=-lim,
            cmax=lim,
            line=dict(width=1, color="rgba(255,255,255,0.8)"),
            colorbar=dict(title="Surplus", tickformat="$.2s", thickness=12),
        ),
        customdata=p[["player_name", "top_archetype", "surplus"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Salary %{x:$,.0f}"
            "<br>Talent %{y:+.2f}<br>Surplus %{customdata[2]:$,.0f}<extra></extra>"
        ),
    )
)
fig.update_layout(
    title="Salary vs. talent (blue = more value than salary, red = less)",
    height=480,
    margin=dict(l=70, r=10, t=40, b=50),
    xaxis=dict(title=f"{d.meta['salary_season']} salary", tickformat="$.2s"),
    yaxis=dict(title="Talent (pts per 100 possessions)", zeroline=True),
)
st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)
st.caption(
    f"Price of a win used for value: {money_md(d.meta['valuation']['price_per_win'])} "
    "(estimated from league payroll and wins; see Methods)."
)
approx_note()
