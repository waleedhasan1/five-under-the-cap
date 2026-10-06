"""Methods: data, archetypes, models, the validation result, and the limitations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import POS, data, money

st.set_page_config(page_title="Methods · Five Under the Cap", page_icon="🏀", layout="wide")
d = data()
vm, am, gm = d.validation_meta, d.archetype_meta, d.meta
h = vm["headline"]
train = vm["training_seasons"]

st.title("Methods")
st.markdown(
    "**Question.** Does player *fit* matter beyond individual talent? Specifically: does adding "
    "role-pairing terms to a lineup model predict next-season lineup net rating better than "
    "summing individual player ratings?"
)
st.info(
    f"**Result. Measured carefully, fit terms did not beat summed talent out of sample.** "
    f"On {h['n_test_lineups']} lineups from {vm['test_season']}, the fit model's error was "
    f"{h['rmse_a_minus_b']:.2f} points per 100 possessions lower than talent alone, with a 95% "
    f"interval of {h['ci95_low']:+.2f} to {h['ci95_high']:+.2f}: no detectable difference.",
    icon="📏",
)

st.header("Data")
st.markdown(f"""
- **Lineups, players, teams:** NBA.com stats via `nba_api`, regular seasons {train[0]} to
  {vm["test_season"]}. Five-man lineup totals (possessions, net rating), player box, advanced,
  scoring, shot-zone, bio and Synergy play-type stats.
- **Salaries:** the Basketball Reference contracts table for {gm["salary_season"]}, saved once.
  Players are matched to NBA.com by normalized name and team, with a manual override file.
- **Salary cap ({gm["salary_cap"]["season"]}):** {money(gm["salary_cap"]["amount"])}. Source:
  {gm["salary_cap"]["source"]}; retrieved {gm["salary_cap"]["retrieved"]}. Shown for context;
  game budgets come from team payrolls (below).
- **Game pool:** {gm["pool_size"]} players with at least 500 minutes in {gm["ratings_through"]}
  and a {gm["salary_season"]} salary. **Rookies without NBA minutes are excluded**, as are
  players missing from the contracts table when it was saved (for example, unsigned players).
""")

st.header("Archetypes")
names = gm["archetype_names"]
st.markdown(f"""
Each player-season (500+ minutes, {train[0]} to {train[-1]}; {am["n_player_seasons"]:,} in all) is
described by {len(am["features"])} style features: shot-zone shares, free-throw rate, share of
made baskets that were assisted, usage, assist and rebound rates, steal and block rates, height,
weight and Synergy play-type shares. Features are standardized, reduced with PCA to
{am["pca_components"]} components ({am["pca_explained_variance"]:.0%} of the variance), and fit
with a Gaussian mixture. Players get **soft memberships** — a probability for each archetype —
not a single label.

The number of archetypes (**{am["k"]}**) was chosen by BIC over 5–12. Refitting with 10 other
random seeds, every seed chose the same number; label agreement with the reference fit averaged
{am["stability_ari_mean"]:.2f} (adjusted Rand index), with a minimum of
{am["stability_ari_min"]:.2f}: most seeds find the same groups, a few find a different split.
A first version without PCA picked the edge of the search range and was unstable across seeds;
it was rejected.
""")
top = d.archetype_top[d.archetype_top["rank"] <= 4]
rows = []
for a, col in enumerate(names):
    t = top[top["archetype"] == a]
    share = float(d.archetype_summary.loc[d.archetype_summary["archetype"] == a, "share"].iloc[0])
    rows.append(
        {
            "Archetype": names[col],
            "Share of player-seasons": share,
            "Most typical members": ", ".join(
                f"{n} ({s[2:]})" for n, s in t[["player_name", "season"]].values
            ),
        }
    )
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Share of player-seasons": st.column_config.ProgressColumn(
            format="%.0f%%", min_value=0, max_value=1
        )
    },
)
st.caption(
    'Names are human labels for statistical groups. "3-and-D" is defined by shooting '
    "(corner threes, off-screen and handoff actions); steal and block rates do not set it apart."
)

st.header("Lineup models")
pb = vm["params_b"]
st.markdown(f"""
The target is a lineup's **net rating per 100 possessions** in a season. Every fit is weighted
by possessions, and older seasons are down-weighted (recency weight {pb["decay"]} per season).

- **Model 0:** the league average.
- **Model A (talent):** an intercept plus one effect per player, fit with ridge regression.
- **Model B (talent + fit):** Model A plus 21 archetype-pair terms. For each pair of players in
  a lineup and each pair of archetypes, the term multiplies their memberships. Fit terms get
  their own ridge penalty.

Penalties and the recency weight were tuned by **time-blocked cross-validation inside the
training seasons only**: each validation season was predicted from the seasons before it.
""")

st.header("Validation")
st.markdown(f"""
Trained on {train[0]}–{train[-1]}, tested **once** on {vm["test_season"]} lineups with at least
{vm["test_min_possessions"]} possessions ({h["n_test_lineups"]} lineups,
{h["test_possessions"]:,.0f} possessions). For test lineups, each player's archetype mix comes
from an earlier season, never from the test season itself.
""")
cmp_ = d.comparison.copy()
labels = {"0": "0 · league mean", "A": "A · talent", "B": "B · talent + fit"}
cmp_["label"] = cmp_["model"].astype(str).map(labels)
fig = go.Figure(
    go.Bar(
        x=cmp_["label"],
        y=cmp_["wrmse"],
        marker_color=POS,
        width=0.5,
        error_y=dict(
            type="data",
            symmetric=False,
            array=cmp_["wrmse_ci_high"] - cmp_["wrmse"],
            arrayminus=cmp_["wrmse"] - cmp_["wrmse_ci_low"],
            color="#888",
        ),
        text=cmp_["wrmse"].map("{:.2f}".format),
        textposition="outside",
        hovertemplate="%{x}<br>RMSE %{y:.2f}<extra></extra>",
    )
)
fig.add_hline(
    y=h["noise_floor_wrmse"],
    line_dash="dash",
    line_color="#888",
    annotation_text=f"noise floor ≈ {h['noise_floor_wrmse']:.1f}",
    annotation_position="bottom right",
)
fig.update_layout(
    title=f"Test-season error, {vm['test_season']} (lower is better; bars show 95% CIs)",
    height=380,
    margin=dict(l=0, r=0, t=40, b=0),
    yaxis=dict(title="weighted RMSE, pts per 100", rangemode="tozero"),
)
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
with c2:
    st.dataframe(
        cmp_[["label", "wrmse", "wrmse_ci_low", "wrmse_ci_high", "wr2"]].rename(
            columns={
                "label": "Model",
                "wrmse": "RMSE",
                "wrmse_ci_low": "95% low",
                "wrmse_ci_high": "95% high",
                "wr2": "R²",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.2f")
            for c in ["RMSE", "95% low", "95% high", "R²"]
        },
    )
iv = vm["interval"]
st.markdown(f"""
- **Headline:** RMSE(A) − RMSE(B) = **{h["rmse_a_minus_b"]:.2f}**, paired-bootstrap 95% interval
  **{h["ci95_low"]:+.2f} to {h["ci95_high"]:+.2f}**. In cross-validation B was slightly better
  than A; that did not carry over to the test season.
- **Why R² is low:** lineup net rating over a few hundred possessions is mostly noise. Splitting
  observed variance into sampling noise and true spread gives a noise floor of about
  {h["noise_floor_wrmse"]:.1f} RMSE, so even a perfect model would reach an R² of only about
  {h["max_achievable_wr2_approx"]:.2f}. (Estimated from how variance shrinks with possessions,
  because game-level lineup data for a split-half check is not in this dataset.)
- **Why R² is negative here:** see the selection-bias limitation below.
- **Intervals:** {int(iv["nominal"] * 100)}% prediction intervals covered
  {iv["empirical_unweighted"]:.0%} of test lineups ({iv["empirical_possession_weighted"]:.0%}
  possession-weighted).
""")

st.header("Value and surplus (approximation)")
v = gm["valuation"]
st.markdown(f"""
A deliberately simple chain, with every input computed from this project's data:

1. **Net rating → wins:** regressing team wins on team net rating ({train[0]}–{train[-1]}) gives
   **{v["wins_per_point"]:.2f} wins per point** of net rating.
2. **A starter's share:** the five most-used players on each team were on court for
   **{v["starter_share"]:.0%}** of team minutes on average in {gm["ratings_through"]}.
3. **Replacement level:** the average effect of low-sample players
   ({v["replacement_effect"]:+.2f}, vs. a possession-weighted average of {v["average_effect"]:+.2f}).
   A team of them would win about {v["replacement_team_wins"]:.0f} games.
4. **Price of a win:** league payroll ({money(v["league_payroll"])}) above a minimum-salary league
   ({v["n_teams"]} teams × {v["roster_size"]:.0f} players × {money(v["min_salary"])}), divided by
   wins above a replacement-level league: **{money(v["price_per_win"])} per win**.

**Value** = wins added over a starter's share of minutes × price per win. **Surplus** = value −
salary. Ridge shrinks player effects toward zero, so values are conservative, and one price per
win ignores contract structure, cap rules and market timing.
""")

st.header("The game")
st.markdown(f"""
- **Budgets:** Standard is the median, across teams, of the sum of each team's five highest
  {gm["salary_season"]} salaries; Easy is the 75th percentile and Hard the 25th
  ({money(gm["budgets"]["easy"])} / {money(gm["budgets"]["standard"])} / {money(gm["budgets"]["hard"])}).
- **Ratings** use Model B refit on all five seasons; archetype mixes come from {gm["ratings_through"]}.
- **Unusual lineups:** a lineup's archetype composition (the sum of its five membership vectors)
  is compared with every observed lineup. If it is further from its nearest neighbour than 95% of
  real lineups are from theirs, the app flags it, widens its range ×{gm["ood_multiplier"]}, and
  the optimizer never picks it.
- **Best lineups** were found offline with CP-SAT, an exact integer solver, over
  {gm["optimizer_pool_size"]} players left after removing clearly dominated ones (cheaper and
  better players of the same top archetype). Each solve had a 20-second limit and none proved
  optimality within it, so these are the best lineups found, not guaranteed optima. Your score's
  percentile compares against random valid lineups at the same budget.
""")

st.header("Limitations")
st.markdown(f"""
- **Selection bias in the test set (to be fixed).** Lineups that reach
  {vm["test_min_possessions"]} possessions are the ones coaches kept playing, which favours
  lineups that were going well. In the test season they outperformed both models' predictions by
  about 2 points per 100, which is why R² is negative even though A and B beat the league average.
- **Star-heavy lineups are extrapolations.** The unusual-lineup check looks at the mix of
  player *types*, not talent level. Five of the league's highest-rated players together is a
  combination the model has rarely or never seen, so predictions for the best lineups (around
  +30 or more) are far outside the range of observed lineups and should be read as rankings,
  not forecasts.
- **No opponent adjustment.** Lineup totals average over whoever the lineup faced; starters
  mostly face starters.
- **Collinearity.** Teammates who almost always play together can't be separated; ridge shrinks
  them toward each other instead.
- **Small samples.** Most lineups play few possessions. Fits weight by possessions; raw lineup
  net ratings are never shown as if they were reliable.
- **Changing roles.** Archetype mixes come from a player's latest season; a trade or a new role
  can change them.
- **New players.** Players with no earlier season get the average effect of low-sample players
  and the average archetype mix when predicting the test season.
- **Mid-range shots** are one zone; short and long mid-range are not separated.
- **Salaries** are one snapshot; two-way deals, waived players and later signings may differ.
""")
