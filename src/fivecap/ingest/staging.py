"""Write raw frames to DuckDB ``stg_*`` tables and run row-count checks."""

from __future__ import annotations

import re

import duckdb
import pandas as pd


def snake(name: str) -> str:
    """Snake-case a column name: ``Restricted Area_FGA`` -> ``restricted_area_fga``."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def to_staging(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Replace ``stg_<table>`` with ``df`` (columns snake-cased, nothing dropped).

    Returns:
        Rows written.
    """
    df = df.rename(columns=snake)
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise ValueError(f"stg_{table}: column names collide after renaming: {dupes}")
    con.register("_staging_df", df)
    con.execute(f"CREATE OR REPLACE TABLE stg_{table} AS SELECT * FROM _staging_df")
    con.unregister("_staging_df")
    return len(df)


def check_counts(
    con: duckdb.DuckDBPyConnection,
    *,
    seasons: list[str],
    min_players: int,
    min_teams: int,
    min_lineups_per_team: int,
) -> list[str]:
    """Row-count checks for the staging tables. Returns a list of failures (empty = pass)."""
    problems: list[str] = []

    def per_season(sql: str) -> dict[str, int]:
        return dict(con.execute(sql).fetchall())

    players = per_season("SELECT season, count(*) FROM stg_player_base GROUP BY 1")
    team_rows = per_season("SELECT season, count(*) FROM stg_team_base GROUP BY 1")
    lineup_teams = per_season(
        f"SELECT season, count(*) FROM (SELECT season, request_team_id FROM stg_lineups "
        f"GROUP BY 1, 2 HAVING count(*) >= {min_lineups_per_team}) GROUP BY 1"
    )
    for s in seasons:
        if players.get(s, 0) < min_players:
            problems.append(
                f"{s}: {players.get(s, 0)} players in stg_player_base (< {min_players})"
            )
        if team_rows.get(s, 0) != min_teams:
            problems.append(f"{s}: {team_rows.get(s, 0)} teams in stg_team_base (!= {min_teams})")
        if lineup_teams.get(s, 0) != min_teams:
            problems.append(f"{s}: lineups present for {lineup_teams.get(s, 0)} teams")
    return problems
