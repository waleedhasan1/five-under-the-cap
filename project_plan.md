# Five Under the Cap — Project Plan

*Working title. Rename freely.*

## 1. Read this first

This repo is a portfolio project for a new graduate applying to NBA team data science roles (the target posting is a Basketball Data Scientist job with the Utah Jazz). The people who will read the repo and use the app are professional basketball analysts. They will judge it on three things, in this order:

1. **Statistical honesty.** Sound validation, stated limitations, uncertainty shown. A small, carefully measured effect beats an impressive-looking one.
2. **Code quality.** Clean, typed, documented, tested, reproducible from a fresh clone.
3. **App polish.** A fast, good-looking web app that is fun to use.

When these conflict, the higher one wins. Never trade correctness for a nicer demo.

## 2. What we are building

**The research question:** Does player *fit* matter beyond individual talent? Specifically, does adding role-pairing interaction terms to a lineup model predict next-season lineup performance better than summing individual player ratings?

**The product:** A web game. The user gets a salary budget and picks five real NBA players. The app shows payroll against the budget, the predicted net rating of the lineup with an uncertainty range, which pairings help or hurt, and how close the user came to the best possible lineup at that budget.

The game is the front end. The lineup model and its validation are the project.

## 3. Scope

**In v1**

- Data pipeline from public sources into DuckDB
- Player archetypes from a mixture model (soft memberships)
- Two lineup models: sum-of-parts baseline and sum-of-parts plus fit terms
- Out-of-sample comparison of the two on a held-out season
- Player dollar values and surplus value
- Precomputed optimal lineups per budget level
- Streamlit app with the game, a player explorer, and a methods page
- Tests, CI, README with results and limitations

**Stretch (only after v1 is done and deployed)**

- Stint-level data with opponent adjustment (proper regularized adjusted plus-minus)
- Separate offensive and defensive models
- Hierarchical Bayesian version of the lineup model in PyMC
- Eight-man rotation mode with minutes allocation
- Shareable result links

**Out of scope**

- Live data refresh inside the app
- User accounts or a stored leaderboard
- Anything involving betting

## 4. Stack

| Purpose | Choice |
|---|---|
| Language | Python 3.11+ |
| Environment | `uv` (or `pip` + `requirements.txt` if `uv` is unavailable) |
| Storage | DuckDB, with transformations written as `.sql` files |
| Data frames | pandas |
| Modeling | scikit-learn (`GaussianMixture`, `Ridge`, `BayesianRidge`) |
| Optimization | OR-Tools (CP-SAT), with a local-search fallback |
| App | Streamlit + Plotly |
| Quality | pytest, ruff, mypy (non-strict), GitHub Actions |

Do not add dependencies outside this list without noting why in the PR description.

## 5. Repo layout

```
.
├── README.md                 # question, method, results, limitations, how to run
├── PROJECT_PLAN.md           # this file
├── Makefile                  # make data / features / models / app-data / test / app
├── pyproject.toml
├── config/
│   └── settings.yaml         # seasons, thresholds, budgets, cap figure + source + date
├── data/
│   ├── raw/                  # cached API responses and CSVs (gitignored)
│   └── warehouse.duckdb      # (gitignored)
├── sql/                      # staging and mart transformations
├── src/fivecap/
│   ├── ingest/               # one module per source
│   ├── features/             # player feature building
│   ├── archetypes.py
│   ├── lineup_model.py
│   ├── valuation.py
│   ├── optimizer.py
│   ├── evaluate.py
│   └── export.py             # writes app_data/
├── app/
│   ├── Home.py               # the game
│   └── pages/                # Player Explorer, Methods
├── app_data/                 # small parquet files the app reads (committed)
├── reports/                  # validation tables and figures (committed)
└── tests/
```

## 6. Data

### Sources

| Data | Source | Notes |
|---|---|---|
| Five-man lineup stats | `nba_api` → `LeagueDashLineups` (Advanced, per 100 possessions) | Pull per team per season to avoid row caps. Verify row counts. |
| Player box and advanced stats | `nba_api` → `LeagueDashPlayerStats` (Base and Advanced) | |
| Shot locations | `nba_api` → `LeagueDashPlayerShotLocations` or `ShotChartDetail` | Zone shares only; no need for raw coordinates in v1. |
| Play-type mix | `nba_api` → `SynergyPlayTypes` | Known to be flaky. Treat as optional features; the pipeline must run without them. |
| Height, weight, age | `nba_api` → `LeagueDashPlayerBioStats` | |
| Team records and net ratings | `nba_api` → `LeagueDashTeamStats` | For the net-rating-to-wins conversion. |
| Salaries | Basketball Reference contracts table | Preferred method is a manual CSV export placed in `data/raw/salaries/`. See rules below. |

### Seasons

- Lineup model training and testing: the five most recent completed regular seasons. The most recent one is the test season; the rest are training.
- Game: player ratings fit on all five seasons (recency-weighted), paired with current-season salaries.
- Player pool for the game: players with at least 500 minutes in the most recent completed season and a current-season salary. Rookies without NBA minutes are excluded; say so on the Methods page.

Put all of these in `config/settings.yaml`, not in code.

### Ingestion rules

- **Cache everything.** Every API response is written to `data/raw/` before parsing. Re-running ingestion must not re-hit the network when a cache file exists.
- **Rate limit.** Sleep between calls to stats.nba.com (start at 1 second) and retry with backoff. Set browser-like headers; the API rejects default ones.
- **Cloud IPs are often blocked by stats.nba.com.** If ingestion fails from your environment, stop and tell the owner so they can run `make data` locally and commit nothing but the cache location. Do not work around this with proxies.
- **Salaries:** do not write an aggressive scraper. If automated access is used at all, make a single cached request per page and respect the site's published rate limits. The default path is that the owner exports the table to CSV by hand.
- **Never generate synthetic data to stand in for real data.** Small hand-built fixtures are fine in `tests/` and must be clearly labelled as fixtures.

### Warehouse tables

- `stg_*`: one staging table per raw source, typed and renamed, nothing dropped
- `dim_player`: one row per player
- `fct_player_season`: one row per player-season with all features
- `fct_lineup_season`: one row per five-man lineup per season, with five player IDs (sorted), possessions, minutes, net rating
- `dim_salary`: player, season, salary
- `fct_team_season`: wins, losses, net rating

Player ID joins between NBA.com and Basketball Reference are done by normalized name plus team, with a manual override file (`config/player_id_overrides.csv`) for mismatches. Log every unmatched player; fail the build if more than 2% of the game pool is unmatched.

## 7. Modeling spec

### 7.1 Archetypes

- **Unit:** player-season, minimum 500 minutes.
- **Features:** shot-zone shares (rim, short mid-range, long mid-range, corner three, above-the-break three), three-point attempt rate, free-throw rate, share of made baskets that were assisted, usage, assist rate, offensive and defensive rebound rates, steal and block rates, height, weight, and play-type frequencies when available.
- **Method:** standardize, then fit a Gaussian mixture. Choose the number of components by BIC over a range of 5–12, and check stability across random seeds. Keep **soft memberships** (a probability vector per player-season), not hard labels.
- **Output:** a membership matrix, plus a human-readable name for each archetype. Draft names from the cluster centroids and top members, and flag them for the owner to review.

### 7.2 Lineup models

Target: lineup net rating per 100 possessions for a lineup-season. All fits are weighted by possessions.

- **Model 0 (floor):** predict the league mean.
- **Model A (sum of parts):** `net_rating = intercept + Σ player effects`. Ridge regression on player indicator columns. Player effects are pooled across training seasons with recency weights.
- **Model B (sum of parts + fit):** Model A plus archetype-pair features. For archetypes `a ≤ b`, the feature for a lineup is the sum over its ten player pairs of `m_i[a]·m_j[b] + m_i[b]·m_j[a]` (halve the diagonal case), where `m` is the membership vector. Use a separate ridge penalty for the fit coefficients.

Tune penalties with time-blocked cross-validation inside the training seasons only. Players in the test season who are absent from training get the mean effect of players with a similar archetype mix and minutes level.

### 7.3 Validation

- **Split:** by time only. Train on the earlier seasons, test on the most recent. Never split lineups randomly; the same players appear on both sides and it leaks.
- **Test set:** lineups with at least 100 possessions in the test season.
- **Metrics:** possession-weighted RMSE and R² for Models 0, A, and B.
- **The headline number:** the difference in weighted RMSE between A and B, with a 95% interval from a paired bootstrap over test lineups.
- **Expect low R².** Net rating over a few hundred possessions carries noise on the order of 10+ points per 100. A low R² is not a bug. Estimate a rough noise ceiling (for example from split-half reliability of lineup net rating) and report model performance relative to it.
- **Report the result whichever way it comes out.** If fit terms do not help, the README says so and discusses why. Do not tune until B wins.

Write the comparison table and one figure to `reports/`.

### 7.4 Uncertainty

Every lineup prediction shown in the app has an 80% interval. For v1, get it from a bootstrap over training lineups (refit, predict, take quantiles) or from `BayesianRidge`. On the test set, report the empirical coverage of the 80% interval.

### 7.5 Dollar value

- **Net rating to wins:** regress team wins on team net rating over the training seasons; use the slope.
- **Price of a win:** v1 uses a simple league-wide estimate, payroll above a minimum-salary roster divided by wins above a replacement-level team. State the assumptions on the Methods page.
- **Player value:** player effect → wins added over a starter's share of minutes → dollars. **Surplus** = value − salary.

This is deliberately simple. Label it as an approximation in the app.

### 7.6 Optimizer

Find the best five players under a budget according to Model B. With pair terms this is a quadratic knapsack problem.

- Pre-filter dominated players (worse rating and higher salary than several alternatives of similar archetype).
- Solve with CP-SAT using linearized pair products. If that is too slow on the filtered pool, fall back to simulated annealing with many restarts and report the best found.
- **Restrict the search to in-distribution lineups** (see pitfall 11.1). An unconstrained optimizer will find lineups the model has never seen anything like and exploit the extrapolation.
- Run offline for each difficulty level and store the top 10 lineups in `app_data/`. The app never solves live.

## 8. App spec

The app reads only from `app_data/`. No network calls, no model fitting at runtime. A prediction is a few dot products and must feel instant.

### Page 1: The game

- Difficulty selector (Easy / Standard / Hard) that sets the budget.
- Searchable player picker showing name, team, salary, and top archetype. Five slots.
- Payroll bar that fills toward the budget line and turns red when over. Over-budget lineups cannot be submitted.
- Live predicted net rating with its 80% interval, split into "talent" (Model A part) and "fit" (pair terms).
- A 5×5 grid of pairwise fit contributions, colored by sign with a diverging palette that is colorblind-safe.
- On submit: a score, the percentile against a precomputed sample of random valid lineups, and the gap to the optimal lineup, with a button to reveal it.
- A visible warning when the lineup is out of distribution.

### Page 2: Player explorer

Sortable table of the pool: rating, salary, estimated value, surplus, archetype mix. A scatter of salary against rating with surplus highlighted.

### Page 3: Methods

Plain-language explanation of the data, archetypes, both models, the validation result with the figure from `reports/`, and the limitations list. This page is what a hiring analyst will read most closely. Write it for that reader.

### Budgets

Derive from the salary data and store in config: Standard is the median across teams of the sum of their five highest salaries; Easy is the 75th percentile; Hard is the 25th. Also record the current league salary cap in config **with its source and retrieval date**; look it up from an official source, do not rely on memory.

## 9. Milestones

Complete in order. Commit at the end of each. Do not start a milestone until the previous one meets its acceptance criteria.

| # | Milestone | Acceptance criteria |
|---|---|---|
| 1 | Scaffold | Repo layout exists; `make test` runs and passes a trivial test; CI is green; ruff clean. |
| 2 | Ingestion | `make data` populates staging tables for all seasons from cache; row-count checks pass; ingestion is idempotent. |
| 3 | Warehouse | Mart tables built by SQL; player ID match rate ≥ 98% for the game pool; data tests pass (no duplicate keys, lineup rows have exactly five distinct players, possessions > 0). |
| 4 | Archetypes | Membership matrix written; BIC curve and cluster summaries in `reports/`; results stable across seeds; draft names flagged for owner review. |
| 5 | Lineup models + validation | Models 0, A, B fit; comparison table with bootstrap interval in `reports/`; interval coverage reported; no test-season data touches fitting or tuning. |
| 6 | Valuation + optimizer | Player values and surplus computed; top-10 lineups per difficulty stored; optimizer results are all in-distribution and within budget. |
| 7 | App | All three pages work locally from `app_data/` only; cold start under ~5 seconds; picking a player updates the prediction without a noticeable delay. |
| 8 | Ship | README complete with results and limitations; app deployed to Streamlit Community Cloud; fresh-clone reproduction instructions verified. |

**Stop and check in with the owner after milestones 4 and 5.** The archetype names and the validation result both shape how the app is framed.

## 10. Engineering standards

- Type hints on all public functions; docstrings that state inputs, outputs, and assumptions.
- No logic in notebooks. Notebooks, if any, only call into `src/` for exploration.
- All thresholds, seasons, and paths come from `config/settings.yaml`.
- Fixed random seeds, recorded in config.
- Tests cover: SQL transformations on fixture data, pair-feature construction (hand-checked small case), the time split (assert no leakage), budget and five-player constraints in the optimizer, and that the app's prediction function matches the model's.
- Each model artifact is saved with a small metadata file: training seasons, config hash, date, metrics.
- CI runs ruff, mypy, and pytest on every push.

## 11. Known pitfalls

1. **Extrapolation to absurd lineups.** The model has never seen five centers together, so its prediction for one is meaningless. Compute each candidate lineup's distance to the nearest observed lineup in archetype-composition space (the sum of the five membership vectors). Beyond the 95th percentile of observed nearest-neighbor distances, flag it in the app, widen the interval, and exclude it from the optimizer.
2. **Leakage.** Archetype features and player effects used to predict the test season must come from earlier seasons only.
3. **No opponent adjustment in v1.** Lineup-level aggregates average over whoever the lineup faced. Starters mostly face starters, which biases comparisons. State this as the main limitation; the stint-level stretch goal fixes it.
4. **Collinearity.** Teammates who always play together cannot be separated. Ridge handles the arithmetic but not the identification; note it on the Methods page.
5. **Small samples.** Most five-man lineups have very few possessions. Weight by possessions and never display raw lineup net rating as if it were reliable.
6. **Players who changed teams or roles.** Last season's archetype may not describe this season's role. Accept this in v1 and list it as a limitation.
7. **Salary edge cases.** Two-way contracts, mid-season signings, and players with multiple rows. Deduplicate to one current-season cap figure per player and log what was dropped.

## 12. Rules for the agent

- If a data pull fails or returns something unexpected, stop and report. Do not substitute invented or approximated data.
- Do not state basketball or salary-cap facts from memory in the README or app. Either compute them from the data or cite a source with a date.
- Do not change the research question, the validation design, or the stack without asking.
- Do not tune on the test season, and do not rerun the validation with different choices until Model B looks better.
- Keep limitations visible. Never remove a caveat to make the app read more confidently.
- Prefer the simplest implementation that meets the acceptance criteria. Stretch goals wait until v1 ships.
- At each check-in, summarize what was built, what the numbers say, and anything that looked wrong.

## 13. Open questions for the owner

1. Final project name and GitHub repo name.
2. Will you run `make data` locally if the API blocks the agent's environment, and export the salary CSV by hand?
3. Approve or rename the archetypes after milestone 4.
4. Should the game enforce any positional rule, or rely only on the out-of-distribution guard?
5. After milestone 5: how to frame the result on the Methods page and in the resume bullet.

## 14. Definition of done

- A stranger can clone the repo, follow the README, and reproduce the validation table.
- The deployed app loads quickly, the game is playable end to end, and every number shown has an uncertainty range or an "approximation" label where one applies.
- The README states the question, the method, the measured result with its interval, and the limitations, in under two screens.
- CI is green.
- The owner can explain every modeling choice in an interview. If something in the repo is too clever to explain simply, simplify it.