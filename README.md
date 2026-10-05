# Five Under the Cap

**Question:** Does player *fit* matter beyond individual talent? Does adding
role-pairing interaction terms to a lineup model predict next-season lineup
net rating better than summing individual player ratings?

**Status:** work in progress. See [`project_plan.md`](project_plan.md) for the full plan.

| Milestone | State |
|---|---|
| 1. Scaffold | done |
| 2. Ingestion | code written; needs a run from a network that can reach stats.nba.com |
| 3–8 | not started |

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

### Data you provide by hand

Salaries come from the Basketball Reference contracts table. Export it to CSV
and place it in `data/raw/salaries/` (the file is gitignored).

## Limitations

To be written with the results. Known up front: no opponent adjustment in v1
(lineup aggregates average over whoever the lineup faced).
