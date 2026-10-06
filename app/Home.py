"""Five Under the Cap: pick five real players under a salary budget."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import (
    DIVERGING,
    HEADSHOT_COL,
    OVER,
    approx_note,
    data,
    headshot,
    money,
    money_md,
    signed,
)
from theme import XP_BLUE, section, style_fig, window

from fivecap.game import pair_grid

st.set_page_config(
    page_title="Five Under the Cap",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)
window("Home", "Five Under the Cap — Build a lineup")
d = data()
players, game, meta = d.players, d.game, d.meta

st.markdown(
    f"Build a five-man lineup from **{len(players)}** real NBA players under a "
    f"{meta['salary_season']} salary budget. The model predicts its net rating per 100 "
    "possessions and shows how much comes from **talent** and how much from **fit**."
)

DIFFICULTY = {"Easy": "easy", "Standard": "standard", "Hard": "hard"}
c1, c2 = st.columns([2, 3])
with c1:
    level = st.segmented_control("Difficulty", list(DIFFICULTY), default="Standard") or "Standard"
key = DIFFICULTY[level]
budget = float(meta["budgets"][key])
with c2:
    st.metric(
        "Budget",
        money(budget),
        help=(
            "Standard is the median, across teams, of the sum of each team's five highest "
            f"{meta['salary_season']} salaries; Easy is the 75th percentile and Hard the 25th."
        ),
    )

labels = {
    i: f"{r.player_name} · {r.contract_team} · {money(r.salary)} · {r.top_archetype}"
    for i, r in players.iterrows()
}
order = players.sort_values("salary", ascending=False).index.tolist()
picked: list[int] = st.multiselect(
    "Pick five players (type to search)",
    order,
    format_func=labels.__getitem__,
    max_selections=5,
    key=f"pick_{key}",
    placeholder="Start typing a name…",
)
if picked:
    for col, i in zip(st.columns(5)[: len(picked)], picked, strict=True):
        r = players.loc[i]
        col.image(headshot(r.player_id), width="stretch")
        col.caption(f"**{r.player_name}** · {money_md(r.salary)}")
payroll = float(players.loc[picked, "salary"].sum()) if picked else 0.0
over = payroll > budget

# Payroll bar
bar = go.Figure()
bar.add_trace(
    go.Bar(
        x=[payroll],
        y=[""],
        orientation="h",
        marker_color=OVER if over else XP_BLUE,
        width=0.5,
        hovertemplate=f"Payroll {money(payroll)}<extra></extra>",
    )
)
bar.add_vline(
    x=budget,
    line_width=2,
    line_dash="dash",
    annotation_text=f"Budget {money(budget)}",
    annotation_position="top",
)
bar.update_layout(
    height=90,
    margin=dict(l=0, r=0, t=28, b=0),
    showlegend=False,
    xaxis=dict(range=[0, max(budget, payroll) * 1.08], tickformat="$.3s", showgrid=False),
    yaxis=dict(visible=False),
)
st.plotly_chart(
    style_fig(bar), use_container_width=True, config={"displayModeBar": False}, theme=None
)
remaining = budget - payroll
st.markdown(
    f"**Payroll {money_md(payroll)}** · "
    + (
        f":red[**Over budget by {money_md(-remaining)}** — this lineup can't be submitted]"
        if over
        else f"{money_md(remaining)} left · {5 - len(picked)} slot(s) open"
    )
)

if len(picked) < 5:
    st.info("Pick five players to see the prediction.")
    approx_note()
    st.stop()

idx = players.index.get_indexer(picked)
pred = {k_: float(v[0]) for k_, v in game.predict(idx[None, :]).items()}
names = players.loc[picked, "player_name"].tolist()

if pred["ood"]:
    st.warning(
        "**Unusual lineup.** Its mix of player types is unlike any lineup the model has seen "
        "(further from the nearest observed lineup than 95% of real lineups are from theirs). "
        f"The prediction is an extrapolation, so its range has been widened ×{meta['ood_multiplier']}.",
        icon="⚠️",
    )

m1, m2, m3 = st.columns(3)
m1.metric(
    "Predicted net rating",
    signed(pred["total"]),
    help="Points per 100 possessions above or below the opponent.",
)
m2.metric(
    "Talent",
    signed(pred["talent"]),
    help="Sum of the five players' individual effects (Model A part).",
)
m3.metric(
    "Fit", signed(pred["fit"]), help="Archetype-pair terms, relative to an average lineup's fit."
)
st.markdown(
    f"**{int(meta['interval'] * 100)}% range: {signed(pred['low'])} to {signed(pred['high'])}** "
    "— the range for this lineup's underlying net rating, from model uncertainty plus "
    "lineup-to-lineup variation the model can't explain."
)

# Pairwise fit grid
grid = pair_grid(game.memberships[idx], game.g)
short = [n.split(" ", 1)[-1] for n in names]
lim = max(0.25, float(np.abs(grid).max()))
text = [["" if i == j else signed(grid[i, j]) for j in range(5)] for i in range(5)]
heat = go.Figure(
    go.Heatmap(
        z=np.where(np.eye(5, dtype=bool), np.nan, grid),
        x=short,
        y=short,
        zmin=-lim,
        zmax=lim,
        colorscale=DIVERGING,
        text=text,
        texttemplate="%{text}",
        xgap=2,
        ygap=2,
        colorbar=dict(title="pts/100", thickness=12),
        hovertemplate="%{y} + %{x}: %{z:+.2f} pts/100<extra></extra>",
    )
)
heat.update_layout(
    height=380,
    margin=dict(l=150, r=10, t=40, b=80),
    xaxis=dict(showgrid=False),
    title="How each pair fits (blue helps, red hurts)",
    yaxis=dict(autorange="reversed", showgrid=False),
)
gc1, gc2 = st.columns([3, 2])
with gc1:
    st.plotly_chart(style_fig(heat), use_container_width=True, theme=None)
with gc2:
    st.markdown("**Your five**")
    st.dataframe(
        players.loc[picked, ["player_id", "player_name", "top_archetype", "salary", "effect"]]
        .assign(player_id=lambda x: x["player_id"].map(headshot))
        .rename(
            columns={
                "player_id": " ",
                "player_name": "Player",
                "top_archetype": "Archetype",
                "salary": "Salary",
                "effect": "Talent",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            " ": HEADSHOT_COL,
            "Salary": st.column_config.NumberColumn(format="$%,.0f"),
            "Talent": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
    st.caption(
        "Fit values are small: in validation, fit terms did not measurably improve "
        "predictions over talent alone (see Methods)."
    )

# Submit
submitted = st.button("Submit lineup", type="primary", disabled=over)
state_key = f"submitted_{key}"
if submitted:
    st.session_state[state_key] = tuple(sorted(picked))
if st.session_state.get(state_key) == tuple(sorted(picked)):
    rand = d.random_scores.loc[d.random_scores["difficulty"] == key, "total"].to_numpy()
    best = d.optimal[(d.optimal["difficulty"] == key) & (d.optimal["rank"] == 1)].iloc[0]
    pct = 100 * float((rand < pred["total"]).mean())
    gap = float(best["total"]) - pred["total"]
    section("Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Your score", signed(pred["total"]))
    r2.metric(
        "Percentile",
        f"{pct:.0f}th",
        help=f"Versus {len(rand):,} random valid lineups at this budget.",
    )
    r3.metric("Gap to best possible", f"{gap:.1f}" if gap > 1e-6 else "0.0 — you found it!")
    if pred["ood"]:
        st.caption(
            "Unusual lineups are excluded from the optimizer, so the gap compares against "
            "the best in-distribution lineup."
        )
    if st.toggle("Reveal the best lineup at this budget"):
        top = d.optimal[d.optimal["difficulty"] == key].copy()
        lookup = players.set_index("player_id")["player_name"]
        top["Lineup"] = top[["p1", "p2", "p3", "p4", "p5"]].apply(
            lambda r: ", ".join(lookup[int(x)] for x in r), axis=1
        )
        st.dataframe(
            top[["rank", "Lineup", "payroll", "total", "talent", "fit", "low", "high"]].rename(
                columns={
                    "rank": "#",
                    "payroll": "Payroll",
                    "total": "Predicted",
                    "talent": "Talent",
                    "fit": "Fit",
                    "low": "Low",
                    "high": "High",
                }
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Payroll": st.column_config.NumberColumn(format="$%,.0f"),
                **{
                    c: st.column_config.NumberColumn(format="%+.1f")
                    for c in ["Predicted", "Talent", "Fit", "Low", "High"]
                },
            },
        )
        st.caption(
            "Top 10 lineups under this budget by the model's prediction, restricted to "
            "lineup types the model has seen. Overlapping ranges mean the ranking is uncertain."
        )
approx_note()
