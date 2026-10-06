"""Pull stats.nba.com endpoints through ``nba_api`` into the raw cache."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from nba_api.stats import endpoints
from nba_api.stats.static import teams

from fivecap.ingest.cache import fetch_cached
from fivecap.ingest.parse import first_result_set

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pull:
    """One stats.nba.com request family that becomes one staging table.

    Attributes:
        table: Staging table name, without the ``stg_`` prefix.
        endpoint: ``nba_api`` endpoint class name.
        params: Fixed keyword arguments for the endpoint.
        per_team: Request once per team (avoids row caps on large endpoints).
        optional: Failures are logged and skipped instead of stopping the run.
        variants: Extra parameter sets, each requested separately and tagged.
    """

    table: str
    endpoint: str
    params: dict[str, Any]
    per_team: bool = False
    optional: bool = False
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)


def pulls(play_types: list[str]) -> list[Pull]:
    """The requests listed in PROJECT_PLAN.md section 6."""
    return [
        Pull(
            "lineups",
            "LeagueDashLineups",
            {
                "group_quantity": 5,
                "measure_type_detailed_defense": "Advanced",
                "per_mode_detailed": "Per100Possessions",
            },
            per_team=True,
        ),
        Pull(
            "player_base",
            "LeagueDashPlayerStats",
            {"measure_type_detailed_defense": "Base", "per_mode_detailed": "Totals"},
        ),
        Pull(
            "player_advanced",
            "LeagueDashPlayerStats",
            {"measure_type_detailed_defense": "Advanced", "per_mode_detailed": "Totals"},
        ),
        Pull(
            "player_scoring",
            "LeagueDashPlayerStats",
            {"measure_type_detailed_defense": "Scoring", "per_mode_detailed": "Totals"},
        ),
        Pull(
            "player_shot_zones",
            "LeagueDashPlayerShotLocations",
            {"distance_range": "By Zone", "per_mode_detailed": "Totals"},
        ),
        Pull("player_bio", "LeagueDashPlayerBioStats", {"per_mode_simple": "Totals"}),
        Pull(
            "team_base",
            "LeagueDashTeamStats",
            {"measure_type_detailed_defense": "Base", "per_mode_detailed": "Totals"},
        ),
        Pull(
            "team_advanced",
            "LeagueDashTeamStats",
            {"measure_type_detailed_defense": "Advanced", "per_mode_detailed": "Totals"},
        ),
        Pull(
            "player_play_types",
            "SynergyPlayTypes",
            {
                "player_or_team_abbreviation": "P",
                "type_grouping_nullable": "offensive",
                "per_mode_simple": "Totals",
            },
            optional=True,
            variants={pt: {"play_type_nullable": pt} for pt in play_types},
        ),
    ]


def _requests(pull: Pull, season: str) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(endpoint params, tags)`` for every request a pull needs in a season."""
    variants = pull.variants or {"": {}}
    team_ids: list[int | None] = [t["id"] for t in teams.get_teams()] if pull.per_team else [None]
    for variant, extra in variants.items():
        for team_id in team_ids:
            params = {**pull.params, **extra}
            tags: dict[str, Any] = {}
            if team_id is not None:
                params["team_id_nullable"] = team_id
                tags["request_team_id"] = team_id
            if variant:
                tags["variant"] = variant
            yield params, tags


def cache_path(raw_dir: Path, pull: Pull, season: str, tags: dict[str, Any]) -> Path:
    """Deterministic cache location for one request."""
    suffix = "_".join(str(v) for v in tags.values())
    name = f"{season}_{suffix}.json" if suffix else f"{season}.json"
    return raw_dir / "nba_stats" / pull.table / name


def pull_season(
    pull: Pull,
    season: str,
    *,
    raw_dir: Path,
    season_type: str,
    ingest: dict[str, Any],
    fetch_factory: Callable[[str, dict[str, Any]], Callable[[], dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    """Fetch (or read from cache) every request for one pull and season.

    Returns:
        One frame with the endpoint's columns plus ``season`` and any request tags.
    """
    factory = fetch_factory or _nba_api_fetch(ingest["timeout_seconds"])
    frames = []
    for params, tags in _requests(pull, season):
        full = {**params, "season": season, "season_type_all_star": season_type}
        payload = fetch_cached(
            cache_path(raw_dir, pull, season, tags),
            factory(pull.endpoint, full),
            sleep_seconds=ingest["sleep_seconds"],
            max_retries=ingest["max_retries"],
            backoff_factor=ingest["backoff_factor"],
        )
        df = first_result_set(payload)
        for k, v in tags.items():
            df[k] = v
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "season", season)
    return out


def _nba_api_fetch(timeout: int) -> Callable[[str, dict[str, Any]], Callable[[], dict[str, Any]]]:
    def factory(endpoint: str, params: dict[str, Any]) -> Callable[[], dict[str, Any]]:
        def fetch() -> dict[str, Any]:
            cls = getattr(endpoints, endpoint)
            resp = cls(**params, timeout=timeout)
            data: dict[str, Any] = json.loads(resp.get_json())
            return data

        return fetch

    return factory
