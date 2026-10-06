# Five Under the Cap

**Question:** Does player *fit* matter beyond individual talent? Does adding
role-pairing interaction terms to a lineup model predict next-season lineup
net rating better than summing individual player ratings?

**Status:** work in progress. See [`project_plan.md`](project_plan.md) for the full plan.

| Milestone | State |
|---|---|
| 1. Scaffold | done |
| 2. Ingestion | done for NBA stats (5 seasons cached, staging built); salaries loaded (2026-27, retrieved 2026-10-05) |
| 3. Warehouse | done: marts built by SQL, data tests pass |
| 4. Archetypes | done (PCA + mixture, k=6); names approved |
| 5. Lineup models + validation | done; result reported below |
| 6–8 | not started |

## Results

**Measured carefully, fit terms did not beat summed talent out of sample.**

Trained on 2021-22 to 2024-25, tested once on the 262 lineups with at least 100 possessions
in 2025-26. Penalties and recency weighting were tuned by time-blocked CV inside the training
seasons; the test season was not used for any fitting or tuning.

| Model | Weighted RMSE (95% CI) | Weighted R² |
|---|---|---|
| 0: league mean | 13.92 (12.58–15.20) | −0.12 |
| A: sum of player effects | 13.31 (12.04–14.53) | −0.03 |
| B: A + archetype-pair fit terms | 13.30 (12.02–14.55) | −0.02 |

- RMSE(A) − RMSE(B) = **0.01** points per 100 possessions, paired-bootstrap 95% CI **−0.10 to +0.11**.
- Noise floor: possession-level sampling noise alone gives a weighted RMSE of about 11.8, which caps
  achievable R² near **0.2**. Most of what separates lineups over a few hundred possessions is noise.
- 80% prediction intervals covered 79% of test lineups (76% possession-weighted).

Details: `reports/lineup_model_comparison.csv`, `reports/lineup_model_comparison.html`,
`reports/lineup_models_meta.json`.

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
make install   # create the environment
make test      # run the tests
make lint      # ruff + mypy
make data      # pull NBA stats into data/raw/ (cached) and build staging tables
make app       # run the Streamlit app
```

### Salaries

Salaries come from the Basketball Reference contracts table
(<https://www.basketball-reference.com/contracts/players.html>). Either export it
with "Share & Export → Get table as CSV", or save the page once as
`data/raw/salaries/contracts_players_<date>.html` and convert it:

```bash
uv run python -c "from pathlib import Path; from fivecap.ingest.salaries import html_to_csv; \
  print(html_to_csv(Path('data/raw/salaries/contracts_players_<date>.html')))"
```

Then run `make data`. Fetch the page once only; Sports Reference blocks clients
that send more than 20 requests a minute. Both files are gitignored.

## Limitations

- **Selection bias in the test set (to fix later).** Lineups that reach 100 possessions are the
  ones coaches kept playing, which favours lineups that were going well. They averaged +4.5 net
  rating while both models predicted about +2.2 for them, which is why R² is negative even though
  A and B beat the league-mean baseline. The same gap appears in the CV seasons. *Follow-up:*
  revisit the test-set definition (or model selection into the sample) together with the
  stint-level stretch goal; see "Follow-ups" below.
- Players new in the test season have no earlier season to take an archetype mix or minutes level
  from, so they get the mean effect of low-sample training players and the average archetype mix.
- Archetypes were fit on all four training seasons, so during CV they had seen the style stats
  (not outcomes) of the validation seasons. The test season is unaffected.
- Collinearity: teammates who nearly always play together cannot be separated; ridge shrinks them
  toward each other rather than identifying them.
- Mid-range shots are one zone (no short/long split) because the zone endpoint does not separate them.
- Salaries are a single snapshot of the Basketball Reference contracts table; players missing from it
  (for example, unsigned at that date) are not in the game pool.
- No opponent adjustment in v1: lineup aggregates average over whoever the lineup faced. Starters
  mostly face starters, which biases comparisons.

## Follow-ups

- [ ] Fix the selection bias in the lineup test set (see Limitations). Do this as a new,
      pre-registered validation, not by re-tuning against the current test season.
- [ ] Stint-level data with opponent adjustment (stretch goal).

