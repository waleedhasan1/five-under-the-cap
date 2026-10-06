"""Build the mart tables from ``stg_*`` with the SQL files in ``sql/``, then run data tests."""

from __future__ import annotations

import difflib
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

from fivecap.config import ROOT, all_seasons, load_settings, path

log = logging.getLogger("fivecap.warehouse")

SQL_DIR = ROOT / "sql"

# Tables and the columns that must be unique together.
KEYS: dict[str, list[str]] = {
    "dim_player": ["player_id"],
    "fct_player_season": ["season", "player_id"],
    "fct_lineup_season": ["season", "team_id", "p1", "p2", "p3", "p4", "p5"],
    "fct_team_season": ["season", "team_id"],
    "dim_salary": ["bbref_id", "season"],
}


def prepare_inputs(con: duckdb.DuckDBPyConnection, overrides_csv: Path) -> None:
    """Load the manual ID overrides and stand in empty tables for optional sources.

    Synergy play types are optional (PROJECT_PLAN.md section 6); when the pull failed the
    play-type features are simply absent.
    """
    overrides = pd.read_csv(overrides_csv, dtype={"bbref_id": "string", "nba_player_id": "Int64"})
    con.register("_overrides", overrides)
    con.execute("CREATE OR REPLACE TABLE stg_player_id_overrides AS SELECT * FROM _overrides")
    con.unregister("_overrides")
    con.execute(
        "CREATE TABLE IF NOT EXISTS stg_player_play_types "
        "(season VARCHAR, player_id BIGINT, variant VARCHAR, poss BIGINT)"
    )


def build(con: duckdb.DuckDBPyConnection, sql_dir: Path = SQL_DIR) -> None:
    """Run every ``.sql`` file in ``sql_dir`` in filename order."""
    for f in sorted(sql_dir.glob("*.sql")):
        log.info("running %s", f.name)
        con.execute(f.read_text())


def check(
    con: duckdb.DuckDBPyConnection,
    *,
    seasons: list[str],
    n_teams: int,
    salary_season: str,
    pool_season: str,
    pool_min_minutes: float,
    max_unmatched_share: float,
    report_dir: Path,
) -> list[str]:
    """Data tests for the marts (PROJECT_PLAN.md milestone 3).

    Returns:
        Failure messages; empty means every test passed. Diagnostics are logged.
    """
    fails: list[str] = []

    def one(sql: str) -> int:
        row = con.execute(sql).fetchone()
        assert row is not None
        return int(row[0] or 0)

    for table, key in KEYS.items():
        cols = ", ".join(key)
        dupes = one(
            f"SELECT count(*) FROM (SELECT {cols} FROM {table} GROUP BY ALL HAVING count(*) > 1)"
        )
        if dupes:
            fails.append(f"{table}: {dupes} duplicate keys on ({cols})")

    bad = one(
        "SELECT count(*) FROM fct_lineup_season "
        "WHERE len(list_distinct([p1, p2, p3, p4, p5])) <> 5 "
        "OR p1 IS NULL OR p2 IS NULL OR p3 IS NULL OR p4 IS NULL OR p5 IS NULL"
    )
    if bad:
        fails.append(f"fct_lineup_season: {bad} rows without five distinct players")
    if one("SELECT count(*) FROM fct_lineup_season WHERE possessions <= 0"):
        fails.append("fct_lineup_season: rows with possessions <= 0")

    for s in seasons:
        n = one(f"SELECT count(*) FROM fct_team_season WHERE season = '{s}'")
        if n != n_teams:
            fails.append(f"fct_team_season: {n} teams in {s}")

    orphans = one(
        "SELECT count(*) FROM (SELECT season, unnest([p1, p2, p3, p4, p5]) AS player_id "
        "FROM fct_lineup_season) AS l ANTI JOIN fct_player_season USING (season, player_id)"
    )
    if orphans:
        fails.append(f"fct_lineup_season: {orphans} lineup slots with no player-season row")

    multi = one("SELECT count(*) FROM int_salary_dedup WHERE distinct_salaries > 1")
    if multi:
        fails.append(f"salary: {multi} player-seasons with conflicting salary figures")
    for name, teams, rows in con.execute(
        "SELECT player_name, teams, source_rows FROM int_salary_dedup "
        f"WHERE season = '{salary_season}' AND source_rows > 1 ORDER BY player_name"
    ).fetchall():
        log.info("salary dedup: %s had %d rows (%s); kept one", name, rows, ", ".join(teams))
    for name, method in con.execute(
        "SELECT player_name, match_method FROM map_player_id "
        "WHERE match_method IN ('ambiguous', 'unmatched') ORDER BY 2, 1"
    ).fetchall():
        log.info("salary player %s: %s", method, name)

    # Game pool match rate. A pool player with no salary row is either missing from the
    # contracts table (e.g. unsigned when it was saved) or a failed ID match. A failed match
    # is suspected when an unmatched salary row has a near-identical name.
    pool = con.execute(
        f"""
        SELECT p.player_id, p.player_name, p.team_abbreviation, norm_name(p.player_name) AS key
        FROM fct_player_season AS p
        ANTI JOIN (SELECT player_id FROM dim_salary WHERE season = '{salary_season}')
            USING (player_id)
        WHERE p.season = '{pool_season}' AND p.minutes >= {pool_min_minutes}
        """
    ).fetchdf()
    n_pool = one(
        f"SELECT count(*) FROM fct_player_season "
        f"WHERE season = '{pool_season}' AND minutes >= {pool_min_minutes}"
    )
    unmatched_keys = [
        r[0]
        for r in con.execute(
            "SELECT norm_name(player_name) FROM map_player_id WHERE player_id IS NULL"
        ).fetchall()
    ]
    pool["suspected_match"] = [
        ", ".join(difflib.get_close_matches(k, unmatched_keys, n=3, cutoff=0.8))
        for k in pool["key"]
    ]
    pool.drop(columns="key").to_csv(report_dir / "pool_players_without_salary.csv", index=False)
    suspected = pool[pool["suspected_match"] != ""]
    for r in suspected.itertuples():
        log.warning("possible ID mismatch: %s ~ %s", r.player_name, r.suspected_match)
    log.info(
        "game pool: %d players with >= %s minutes in %s; %d have no %s salary row "
        "(listed in reports/pool_players_without_salary.csv); %d suspected ID mismatches",
        n_pool,
        pool_min_minutes,
        pool_season,
        len(pool),
        salary_season,
        len(suspected),
    )
    if len(suspected) / n_pool > max_unmatched_share:
        fails.append(
            f"{len(suspected)} suspected ID mismatches in a pool of {n_pool} "
            f"(limit {max_unmatched_share:.0%}); add rows to config/player_id_overrides.csv"
        )
    return fails


def main() -> int:
    """Build the warehouse marts and run the data tests. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = load_settings()
    with duckdb.connect(str(path("warehouse"))) as con:
        prepare_inputs(con, path("player_id_overrides"))
        build(con)
        fails = check(
            con,
            seasons=all_seasons(s),
            n_teams=s["ingest"]["min_teams_per_season"],
            salary_season=s["seasons"]["salary_season"],
            pool_season=s["seasons"]["test"],
            pool_min_minutes=s["thresholds"]["game_pool_min_minutes"],
            max_unmatched_share=s["thresholds"]["max_unmatched_share"],
            report_dir=path("reports"),
        )
    for f in fails:
        log.error("data test failed: %s", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
