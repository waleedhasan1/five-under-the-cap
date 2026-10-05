"""Turn stats.nba.com JSON payloads into data frames."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _flatten_headers(headers: Any) -> list[str]:
    """Column names for a result set.

    Most endpoints give a flat list. Shot-location endpoints give two header rows:
    a group row (zone names, each spanning several columns after ``columnsToSkip``)
    and a leaf row. Those become ``<zone>_<leaf>``.
    """
    if headers and isinstance(headers[0], str):
        return list(headers)
    groups, leaves = headers[0], headers[-1]["columnNames"]
    skip = groups.get("columnsToSkip", 0)
    span = groups["columnSpan"]
    names = list(leaves[:skip])
    for i, leaf in enumerate(leaves[skip:]):
        names.append(f"{groups['columnNames'][i // span]}_{leaf}")
    return names


def result_sets(payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """All result sets in a payload, keyed by name."""
    raw: Any = payload["resultSets"] if "resultSets" in payload else payload["resultSet"]
    sets = raw if isinstance(raw, list) else [raw]
    return {
        s["name"]: pd.DataFrame(s["rowSet"], columns=_flatten_headers(s["headers"])) for s in sets
    }


def first_result_set(payload: dict[str, Any]) -> pd.DataFrame:
    """The first (usually only) result set in a payload."""
    return next(iter(result_sets(payload).values()))
