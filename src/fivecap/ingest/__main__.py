"""``make data``: fill the raw cache and build ``stg_*`` tables for every season."""

from __future__ import annotations

import logging
import sys

import duckdb
import pandas as pd

from fivecap.config import all_seasons, load_settings, path
from fivecap.ingest.nba_stats import pull_season, pulls
from fivecap.ingest.salaries import read_salary_csvs
from fivecap.ingest.staging import check_counts, to_staging

log = logging.getLogger("fivecap.ingest")


def main() -> int:
    """Run ingestion. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = load_settings()
    seasons = all_seasons(s)
    raw_dir = path("raw_dir")
    warehouse = path("warehouse")
    warehouse.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(warehouse)) as con:
        for p in pulls(s["play_types"]):
            frames = []
            try:
                for season in seasons:
                    log.info("%s %s", p.table, season)
                    frames.append(
                        pull_season(
                            p,
                            season,
                            raw_dir=raw_dir,
                            season_type=s["seasons"]["season_type"],
                            ingest=s["ingest"],
                        )
                    )
            except Exception:
                if p.optional:
                    log.exception("Optional pull %s failed; continuing without it", p.table)
                    continue
                log.exception("Pull %s failed. Stopping: do not substitute data.", p.table)
                return 1
            n = to_staging(con, p.table, pd.concat(frames, ignore_index=True))
            log.info("stg_%s: %d rows", p.table, n)

        salaries = read_salary_csvs(path("salary_dir"))
        if salaries is None:
            log.warning("No salary CSV in %s; stg_salary not built", path("salary_dir"))
        else:
            log.info("stg_salary: %d rows", to_staging(con, "salary", salaries))

        problems = check_counts(
            con,
            seasons=seasons,
            min_players=s["ingest"]["min_players_per_season"],
            min_teams=s["ingest"]["min_teams_per_season"],
            min_lineups_per_team=s["ingest"]["min_lineups_per_team"],
        )
    for msg in problems:
        log.error("Row-count check failed: %s", msg)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
