"""Ingestion tests. All payloads here are small hand-built FIXTURES, not real data."""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from fivecap.ingest.cache import fetch_cached
from fivecap.ingest.nba_stats import Pull, pull_season
from fivecap.ingest.parse import first_result_set
from fivecap.ingest.staging import check_counts, snake, to_staging

FIXTURE_FLAT = {
    "resultSets": [
        {"name": "Players", "headers": ["PLAYER_ID", "MIN"], "rowSet": [[1, 600.0], [2, 20.5]]}
    ]
}
FIXTURE_SHOT_ZONES = {
    "resultSets": {
        "name": "ShotLocations",
        "headers": [
            {
                "name": "SHOT_CATEGORY",
                "columnsToSkip": 1,
                "columnSpan": 2,
                "columnNames": ["Restricted Area", "Corner 3"],
            },
            {
                "name": "columns",
                "columnSpan": 1,
                "columnNames": ["PLAYER_ID", "FGM", "FGA", "FGM", "FGA"],
            },
        ],
        "rowSet": [[1, 3, 5, 1, 4]],
    }
}
INGEST = {"sleep_seconds": 0, "max_retries": 2, "backoff_factor": 2.0, "timeout_seconds": 1}


def test_fetch_cached_writes_then_never_refetches(tmp_path: Path) -> None:
    calls = []

    def fetch() -> dict[str, Any]:
        calls.append(1)
        return FIXTURE_FLAT

    f = tmp_path / "x" / "a.json"
    assert fetch_cached(f, fetch, sleep=lambda _: None) == FIXTURE_FLAT
    assert fetch_cached(f, fetch, sleep=lambda _: None) == FIXTURE_FLAT
    assert len(calls) == 1 and f.exists()


def test_fetch_cached_retries_with_backoff_then_raises(tmp_path: Path) -> None:
    waits: list[float] = []

    def fetch() -> dict[str, Any]:
        raise ConnectionError("blocked")

    with pytest.raises(ConnectionError):
        fetch_cached(tmp_path / "a.json", fetch, sleep_seconds=1, max_retries=2, sleep=waits.append)
    assert waits == [1, 2, 4]
    assert not (tmp_path / "a.json").exists()


def test_parse_flat_and_grouped_headers() -> None:
    assert list(first_result_set(FIXTURE_FLAT).columns) == ["PLAYER_ID", "MIN"]
    zones = first_result_set(FIXTURE_SHOT_ZONES)
    assert list(zones.columns) == [
        "PLAYER_ID",
        "Restricted Area_FGM",
        "Restricted Area_FGA",
        "Corner 3_FGM",
        "Corner 3_FGA",
    ]
    assert zones.iloc[0]["Corner 3_FGA"] == 4


def test_snake() -> None:
    assert snake("Restricted Area_FGA") == "restricted_area_fga"
    assert snake("FG3A_FREQUENCY") == "fg3a_frequency"


def test_pull_season_per_team_is_idempotent(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def factory(endpoint: str, params: dict[str, Any]):  # type: ignore[no-untyped-def]
        def fetch() -> dict[str, Any]:
            calls.append(params)
            return FIXTURE_FLAT

        return fetch

    p = Pull("lineups", "LeagueDashLineups", {"group_quantity": 5}, per_team=True)
    kw = dict(raw_dir=tmp_path, season_type="Regular Season", ingest=INGEST, fetch_factory=factory)
    df = pull_season(p, "2024-25", **kw)  # type: ignore[arg-type]
    assert len(calls) == 30  # one request per team
    assert len(df) == 60 and df["request_team_id"].nunique() == 30
    assert (df["season"] == "2024-25").all()
    pull_season(p, "2024-25", **kw)  # type: ignore[arg-type]
    assert len(calls) == 30  # second run reads the cache only


def test_staging_and_row_count_checks() -> None:
    con = duckdb.connect()
    to_staging(con, "player_base", pd.DataFrame({"season": ["s1"] * 3, "PLAYER_ID": [1, 2, 3]}))
    to_staging(con, "team_base", pd.DataFrame({"season": ["s1"] * 2, "TEAM_ID": [1, 2]}))
    to_staging(con, "lineups", pd.DataFrame({"season": ["s1"] * 2, "request_team_id": [1, 2]}))
    assert con.execute("SELECT player_id FROM stg_player_base").fetchall() == [(1,), (2,), (3,)]
    kw = dict(seasons=["s1"], min_lineups_per_team=1)
    assert check_counts(con, min_players=3, min_teams=2, **kw) == []  # type: ignore[arg-type]
    problems = check_counts(con, min_players=4, min_teams=2, **kw)  # type: ignore[arg-type]
    assert len(problems) == 1 and "players" in problems[0]
