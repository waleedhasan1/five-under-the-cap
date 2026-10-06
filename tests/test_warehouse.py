"""SQL transformation tests on small hand-built FIXTURE tables (not real data)."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from fivecap.warehouse import build, prepare_inputs

S = "2025-26"


def _stage(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    con.register("_df", df)
    con.execute(f"CREATE TABLE stg_{name} AS SELECT * FROM _df")
    con.unregister("_df")


@pytest.fixture
def con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    players = pd.DataFrame(
        {
            "season": S,
            "player_id": [1, 2, 3, 4, 5, 6],
            "player_name": [
                "Ana Ñúñez Jr.",
                "Bo Two",
                "Cy Three",
                "Di Four",
                "Ed Five",
                "Ron Smith",
            ],
            "team_id": 100,
            "team_abbreviation": ["PHX", "PHX", "PHX", "PHX", "PHX", "BKN"],
            "age": 25.0,
            "gp": 10,
            "min": 600.0,
            "fga": 100,
            "fg3a": 40,
            "fta": 20,
            "stl": 5,
            "blk": 2,
        }
    )
    _stage(c, "player_base", players)
    _stage(
        c,
        "player_advanced",
        players[["season", "player_id"]].assign(
            poss=1000, usg_pct=0.2, ast_pct=0.1, oreb_pct=0.05, dreb_pct=0.15
        ),
    )
    _stage(c, "player_scoring", players[["season", "player_id"]].assign(pct_ast_fgm=0.5))
    _stage(
        c,
        "player_bio",
        players[["season", "player_id"]].assign(player_height_inches=78, player_weight="210"),
    )
    _stage(
        c,
        "player_shot_zones",
        players[["season", "player_id"]].assign(
            restricted_area_fga=30.0,
            in_the_paint_non_ra_fga=10.0,
            mid_range_fga=20.0,
            corner_3_fga=10.0,
            above_the_break_3_fga=30.0,
        ),
    )
    # Player 1 was traded: two Synergy rows for Isolation that must be summed.
    _stage(
        c,
        "player_play_types",
        pd.DataFrame(
            {
                "season": S,
                "player_id": [1, 1, 1],
                "variant": ["Isolation", "Isolation", "Spotup"],
                "poss": [10, 20, 30],
            }
        ),
    )
    _stage(
        c,
        "lineups",
        pd.DataFrame(
            {
                "season": S,
                "team_id": 100,
                "team_abbreviation": "PHX",
                "gp": 5,
                "group_id": ["-5-3-1-4-2-", "-1-2-3-4-6-"],
                "min": [50.0, 1.0],
                "poss": [100, 0],
                "net_rating": [5.0, 0.0],
                "off_rating": 110.0,
                "def_rating": 105.0,
            }
        ),
    )
    _stage(
        c,
        "team_base",
        pd.DataFrame(
            {"season": S, "team_id": [100], "team_name": ["Fixture"], "gp": 82, "w": 50, "l": 32}
        ),
    )
    _stage(
        c,
        "team_advanced",
        pd.DataFrame(
            {
                "season": S,
                "team_id": [100],
                "net_rating": 4.0,
                "off_rating": 115.0,
                "def_rating": 111.0,
                "pace": 99.0,
                "poss": 8000,
            }
        ),
    )
    _stage(
        c,
        "salary",
        pd.DataFrame(
            {
                "player": ["Ana Nunez", "Bo Two", "Bo Two", "Ronnie Smith"],
                "bbref_id": ["nunezan01", "twobo01", "twobo01", "smithro01"],
                "team": ["PHO", "PHO", "BRK", "BRK"],
                "2026_27": ["$1,000,000", "$2,000,000", "$2,000,000", "$3,000,000"],
                "2027_28": [None, "$2,100,000", "$2,100,000", None],
            }
        ),
    )
    overrides = tmp_path / "overrides.csv"
    overrides.write_text("bbref_id,nba_player_id,note\nsmithro01,6,fixture\n")
    prepare_inputs(c, overrides)
    build(c)
    return c


def test_lineups_sorted_ids_and_zero_possessions_dropped(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute("SELECT p1, p2, p3, p4, p5, possessions FROM fct_lineup_season").fetchall()
    assert rows == [(1, 2, 3, 4, 5, 100)]


def test_player_features(con: duckdb.DuckDBPyConnection) -> None:
    r = con.execute(
        "SELECT zone_rim + zone_paint + zone_mid + zone_corner3 + zone_atb3, zone_rim, fg3a_rate,"
        " stl_per100, weight_lb, pt_isolation, pt_spotup FROM fct_player_season WHERE player_id = 1"
    ).fetchone()
    assert r == pytest.approx((1.0, 0.3, 0.4, 0.5, 210.0, 0.5, 0.5))


def test_salary_dedup_team_codes_and_matching(con: duckdb.DuckDBPyConnection) -> None:
    sal = con.execute(
        "SELECT player_id, season, salary, teams, source_rows FROM dim_salary ORDER BY 1, 2"
    ).fetchall()
    assert sal == [
        (1, "2026-27", 1_000_000, ["PHX"], 1),  # accents and "Jr." normalized away
        (2, "2026-27", 2_000_000, ["BKN", "PHX"], 2),  # two source rows, one kept
        (2, "2027-28", 2_100_000, ["BKN", "PHX"], 2),
        (6, "2026-27", 3_000_000, ["BKN"], 1),  # matched only through the override
    ]
    methods = dict(con.execute("SELECT bbref_id, match_method FROM map_player_id").fetchall())
    assert methods == {"nunezan01": "name", "twobo01": "name", "smithro01": "override"}


def test_team_season(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute("SELECT wins, losses, net_rating FROM fct_team_season").fetchone() == (
        50,
        32,
        4.0,
    )
