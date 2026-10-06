# Project write-up for the resume writer

*Status as of 2026-10-06. All 8 milestones are built. The code is public at
<https://github.com/waleedhasan1/five-under-the-cap>, and the app was verified to run from a fresh
clone. The live app link will be added here once the owner finishes the one-click Streamlit
Community Cloud deploy.*

## Instructions for the resume writer

- Use only the facts in this file. Every number here was produced by the code in this repo.
- **The main result is a null result, and the owner has approved this framing:** "Measured
  carefully, fit terms did not beat summed talent out of sample." Never turn it into a claim
  that fit was found to matter, or that the model is accurate. The selling point is the rigor.
- Do not invent metrics, users or accuracy figures. The **code repository is public** and may be
  linked: <https://github.com/waleedhasan1/five-under-the-cap>. Give a live app URL only once one
  is listed under "Links" below, and never say people use the app.
- Target role: Basketball Data Scientist (Utah Jazz). Readers are professional basketball
  analysts who value statistical honesty over impressive-sounding claims.

## Links

- Code: <https://github.com/waleedhasan1/five-under-the-cap>
- Live app: *not yet listed (owner to add after deploying)*

## One-line summary

An end-to-end NBA analytics project that tests whether player *fit* predicts lineup
performance beyond individual talent, built on a reproducible data pipeline and a
salary-cap lineup game as the front end.

## The question

Does adding role-pairing interaction terms to a lineup model predict next-season five-man
lineup net rating better than summing individual player ratings? Validation is a strict
time split: train on 2021-22 to 2024-25, test on 2025-26.

## What has been built (verified)

**Data pipeline (Python, DuckDB, SQL)**
- Ingests six NBA.com stats endpoint families via `nba_api` for five regular seasons
  (2021-22 to 2025-26): five-man lineups, player box/advanced/scoring stats, shot zones,
  Synergy play types, player bio, and team stats.
- Every API response is cached to disk before parsing; reruns make no network calls
  (a full rebuild from cache takes about 7 seconds). Rate limiting and retry with backoff.
- Lineups are pulled per team per season to avoid row caps: 87,566 raw lineup rows,
  85,299 lineup-seasons with possessions after cleaning.
- 2026-27 salaries from Basketball Reference, fetched with a single cached request after
  checking robots.txt and the site's published rate limit; parsed with a standard-library
  HTML parser (no new dependency).
- Warehouse marts written as SQL files: player-season features, player dimension, lineup
  facts, team facts, salaries.
- Cross-source player ID matching (NBA.com to Basketball Reference) by normalized name
  (accents, punctuation and suffixes removed) with team-code mapping, a tie-breaker on team,
  and a manual override file. Of 485 salary players, 430 matched by name, 1 needed a manual
  override, and the remaining 54 have no NBA minutes in the five seasons (rookies or players
  returning from abroad). Suspected mismatches are flagged with fuzzy matching.
- Automated data tests: unique keys on every table, exactly five distinct players per lineup,
  positive possessions, 30 teams per season, every lineup player present in the player table,
  no conflicting salary figures.

**Player archetypes (scikit-learn)**
- 24 style features per player-season (shot-zone shares, free-throw rate, assisted share,
  usage, assist and rebound rates, steal and block rates, height, weight, Synergy play-type
  shares).
- Standardize, then PCA (7 components, 81% of variance), then a Gaussian mixture with soft
  memberships. Fit on training seasons only (1,477 player-seasons with 500+ minutes) so no
  test-season information leaks into the archetype definitions.
- Number of archetypes chosen by BIC over 5–12 components and two covariance types: k = 6.
- Stability check across 10 random seeds: all 10 chose k = 6; label agreement with the
  reference fit (adjusted Rand index) was about 0.9 for 8 seeds and about 0.5 for 2.
- First attempt without PCA was rejected: BIC picked the edge of the search range and
  seeds disagreed (ARI about 0.49). Adding PCA fixed both. This is a good interview story
  about diagnosing an unstable model instead of tuning until it looked good.
- Archetype names: 3-and-D, post-up big, pass-first guard, self-creator,
  low-usage defensive forward, rim-running big.

**Engineering**
- Python 3.11, uv, DuckDB, pandas, scikit-learn, OR-Tools (CP-SAT), Plotly, Streamlit.
- Typed, documented code; ruff, mypy and pytest (33 tests, including SQL transformations
  run on hand-built fixture tables, a brute-force check of the optimizer, a test that the app's
  prediction matches the model's, and smoke tests of every app page); GitHub Actions CI config;
  Makefile targets for each stage (`make data / features / models / app-data / app`).
- Deployment-ready (milestone 8): public GitHub repo; a fresh clone with only the locked
  dependencies installed renders all three app pages without errors; configured for Streamlit
  Community Cloud (Python 3.11, `uv.lock`, no secrets). The app ships only 1.7 MB of precomputed
  data, so it runs without the raw data or the warehouse.
- All seasons, thresholds and seeds in one config file; model metadata records the config hash.

**Lineup models and validation (milestone 5)**
- Model 0 (league mean), Model A (possession-weighted ridge on player effects, recency-weighted),
  Model B (A plus 21 archetype-pair interaction features with a separate ridge penalty).
- Leakage-free design: time split (train 2021-22 to 2024-25, test 2025-26); penalties tuned by
  time-blocked CV inside training seasons; test-season archetype mixes taken from earlier seasons.
- Result on 262 test lineups: weighted RMSE A 13.31 vs B 13.30; difference 0.01 points per 100
  possessions, paired-bootstrap 95% CI −0.10 to +0.11. Both beat the league-mean baseline (13.92).
- Estimated a noise floor (RMSE ≈ 11.8; achievable R² ≈ 0.2) by separating sampling noise from
  true lineup spread, so results are judged against what is achievable.
- 80% prediction intervals were calibrated: 79% empirical coverage on the test season.
- Diagnosed a selection bias (lineups that reach 100 possessions outperform their talent) and
  documented it as a limitation instead of re-tuning on the test season.

**Player valuation and lineup optimizer (milestone 6)**
- Converted player effects to wins and dollars with every input computed from the data:
  2.28 wins per point of net rating (team regression), starters on court 52% of minutes,
  replacement level from low-sample players, and an estimated price of about $23.8M per win.
  Surplus value = value minus 2026-27 salary. Labelled as an approximation throughout.
- Formulated "best five players under a budget" as a quadratic knapsack and solved it with
  OR-Tools CP-SAT (linearized pair products), after pruning dominated players (347 → 114).
  Repeated solves with no-good cuts exclude lineup types the model has never seen
  (nearest-neighbour distance in archetype space). Top 10 lineups stored for each of three
  budgets derived from team payrolls (Easy $172.8M / Standard $151.0M / Hard $127.4M).
- Honest caveats recorded: solves hit a 20-second limit, so lineups are best found rather than
  proven optimal; star-heavy top lineups are extrapolations in talent.

**Front end: Streamlit web app (milestone 7)**
- Three pages, reading only small precomputed files (1.7 MB); no model fitting or network calls
  at runtime. Data loads in about 1.5 seconds; each lineup prediction takes about 5 ms.
- **The game:** choose a difficulty, search and pick five of 347 real players, watch payroll
  fill toward the budget, and get a live predicted net rating with an 80% range, split into
  talent and fit. Unusual lineups get a visible warning and a wider range. On submit: score,
  percentile against 5,000 random valid lineups, gap to the best lineup, and a reveal of the
  top 10.
- **Player explorer:** sortable table of every pool player (talent, salary, value, surplus,
  archetype mix) with an archetype filter.
- **Methods page:** plain-language write-up of the data, archetypes, models, validation result
  and limitations, with every number read from the pipeline's outputs.

**Data visualizations (Plotly, in the app)**
- Payroll bar that fills toward a budget line and turns red with a text warning when over.
- 5×5 heatmap of pairwise fit contributions using a colorblind-checked diverging palette
  (blue helps, red hurts, grey neutral) with values printed in each cell.
- Salary vs. talent scatter for the whole pool, coloured by surplus, with hover details.
- In-table bar charts of each player's archetype mix.
- Model comparison chart: test-season error for each model with 95% bootstrap intervals and
  the estimated noise floor.
- Archetype table with progress bars for group shares; plus BIC and model-comparison charts
  saved as interactive HTML in `reports/`.

## Not done yet (do not present as done)

- The live app link: the Streamlit Community Cloud deploy is the owner's step (it needs their
  account). Until a link appears under "Links", describe the app as built, not hosted.
- Not verified: a full rebuild of the raw data from a fresh clone (it re-downloads five seasons
  from NBA.com and needs a manually saved salary page). The fresh-clone check covered the app.
- Follow-up: fix the selection bias in the lineup test set (documented limitation).

## Suggested bullets

- Built a reproducible NBA data pipeline (Python, DuckDB, SQL) ingesting five seasons of
  NBA.com lineup, player and play-type data (85K+ lineup-seasons) with cached, rate-limited
  ingestion and automated data-quality tests.
- Matched players across NBA.com and Basketball Reference using normalized-name joins,
  team-code mapping and a manual override file, with fuzzy-match flags for suspected errors.
- Derived six data-driven player archetypes with PCA and a Gaussian mixture model (soft
  memberships), selecting model size by BIC and validating stability across random seeds;
  diagnosed and fixed an unstable first specification.
- Tested whether lineup "fit" predicts performance beyond individual talent with a leakage-free
  time-split validation; fit terms did not beat summed talent out of sample (ΔRMSE 0.01,
  95% CI −0.10 to +0.11), measured against an estimated noise floor that caps R² near 0.2.
- Built calibrated 80% prediction intervals (79% empirical coverage) and diagnosed a selection
  bias in the evaluation set, documenting it rather than tuning on the test season.
- Built an interactive Streamlit web app on top of the model: a salary-cap lineup game with
  live predictions and uncertainty ranges, a player value explorer, and a methods page;
  predictions run in about 5 ms from precomputed data. Code public on GitHub.
- Solved a budget-constrained lineup optimization (quadratic knapsack) with OR-Tools CP-SAT,
  excluding lineup types outside the training data to avoid exploiting model extrapolation.
- Designed Plotly visualizations for analysts and fans: a colorblind-safe pairwise fit heatmap,
  a salary-vs-talent surplus scatter, and model-error charts with bootstrap intervals and a
  noise-floor reference.

## Skills and keywords

Python, SQL, DuckDB, pandas, scikit-learn, PCA, Gaussian mixture models, BIC model selection,
ridge regression, bootstrap confidence intervals, time-blocked cross-validation,
prediction-interval calibration, quadratic knapsack optimization, OR-Tools CP-SAT, Streamlit,
Plotly, data visualization, colorblind-safe design, pytest, mypy, ruff, GitHub Actions, data
pipelines, entity resolution, NBA lineup analytics, player valuation.
