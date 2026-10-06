# Project write-up for the resume writer

*Status as of 2026-10-06. Milestones 1–4 of 8 are complete. Update this file as milestones land.*

## Instructions for the resume writer

- Use only the facts in this file. Every number here was produced by the code in this repo.
- **Do not claim a modeling result yet.** The main experiment (milestone 5: do "fit" terms
  beat a sum-of-talent baseline out of sample?) has not been run. Write bullets that describe
  what was built, not what it found. Once the result exists, the owner chooses how to frame it,
  whichever way it comes out.
- Do not invent metrics, users, accuracy figures or deployment status. The app is not built or
  deployed yet.
- Target role: Basketball Data Scientist (Utah Jazz). Readers are professional basketball
  analysts who value statistical honesty over impressive-sounding claims.

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
- Draft archetype names (pending owner review): movement shooter, post hub big,
  pass-first guard, self-creating scorer, low-usage defensive forward, rim-running big.

**Engineering**
- Python 3.11, uv, DuckDB, pandas, scikit-learn, Plotly, Streamlit (planned UI).
- Typed, documented code; ruff, mypy and pytest (16 tests, including SQL transformations
  run on hand-built fixture tables); GitHub Actions CI config; Makefile targets for each stage.
- All seasons, thresholds and seeds in one config file; model metadata records the config hash.

## Planned (not done; do not present as done)

- Milestone 5: Model 0 (league mean), Model A (ridge on player effects), Model B (A plus
  archetype-pair interaction terms); possession-weighted RMSE with a paired-bootstrap 95%
  interval on the A vs. B difference; 80% prediction intervals with measured coverage.
- Milestone 6: player dollar value and surplus; optimal lineups under a budget via CP-SAT
  (quadratic knapsack), restricted to in-distribution lineups.
- Milestone 7–8: Streamlit game (pick five players under a cap), player explorer, methods page;
  deployment to Streamlit Community Cloud.

## Suggested bullets (safe to use now)

- Built a reproducible NBA data pipeline (Python, DuckDB, SQL) ingesting five seasons of
  NBA.com lineup, player and play-type data (85K+ lineup-seasons) with cached, rate-limited
  ingestion and automated data-quality tests.
- Matched players across NBA.com and Basketball Reference using normalized-name joins,
  team-code mapping and a manual override file, with fuzzy-match flags for suspected errors.
- Derived six data-driven player archetypes with PCA and a Gaussian mixture model (soft
  memberships), selecting model size by BIC and validating stability across random seeds;
  diagnosed and fixed an unstable first specification.
- Designed a leakage-free time-split validation to test whether lineup "fit" adds predictive
  value beyond individual talent (in progress).

## Skills and keywords

Python, SQL, DuckDB, pandas, scikit-learn, PCA, Gaussian mixture models, BIC model selection,
ridge regression (planned), bootstrap confidence intervals (planned), time-series cross-validation
(planned), optimization with OR-Tools (planned), Streamlit (planned), pytest, mypy, ruff,
GitHub Actions, data pipelines, entity resolution, NBA lineup analytics.
