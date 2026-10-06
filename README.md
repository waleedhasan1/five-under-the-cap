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
| 4. Archetypes | done (PCA + mixture, k=6); draft names awaiting owner review |
| 5. Lineup models + validation | run; awaiting owner review of the result |
| 6–8 | not started |

## Results

Not yet available. The validation result will be reported here whichever way it comes out.

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

To be written with the results. Known up front:

- Mid-range shots are one zone (no short/long split) because the zone endpoint does not separate them.
- Salaries are a single snapshot of the Basketball Reference contracts table; players missing from it
  (for example, unsigned at that date) are not in the game pool.
- no opponent adjustment in v1
(lineup aggregates average over whoever the lineup faced).
