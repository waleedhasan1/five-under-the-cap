"""Read hand-exported Basketball Reference salary CSVs from ``data/raw/salaries/``.

Either the owner exports the contracts table by hand, or a single cached copy of the page
is saved to ``data/raw/salaries/`` and converted with ``html_to_csv`` (PROJECT_PLAN.md
section 6). Never more than one request per page.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def read_salary_csvs(salary_dir: Path) -> pd.DataFrame | None:
    """Concatenate every CSV in ``salary_dir``, keeping all columns as text.

    Returns:
        The raw rows with a ``source_file`` column, or ``None`` if no CSV is present.
        Header cleanup and deduplication happen in the warehouse step (milestone 3).
    """
    files = sorted(salary_dir.glob("*.csv"))
    if not files:
        return None
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str)
        df["source_file"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


class _ContractsParser(HTMLParser):
    """Collect cells of the ``player-contracts`` table by their ``data-stat`` attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.season_labels: dict[str, str] = {}  # data-stat "y1" -> "2026-27"
        self.rows: list[dict[str, str]] = []
        self._row: dict[str, str] | None = None
        self._stat: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        stat = a.get("data-stat")
        if tag == "th" and stat and stat.startswith("y") and a.get("aria-label"):
            self.season_labels.setdefault(stat, a["aria-label"] or "")
        if tag == "tr":
            self._row = {}
        elif tag in ("td", "th") and self._row is not None and stat:
            self._stat = stat
            self._row[stat] = ""
        elif tag == "a" and self._row is not None and self._stat == "player":
            href = a.get("href") or ""
            self._row["bbref_id"] = href.rsplit("/", 1)[-1].removesuffix(".html")

    def handle_data(self, data: str) -> None:
        if self._row is not None and self._stat:
            self._row[self._stat] += data

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._stat = None
        elif tag == "tr" and self._row is not None:
            if self._row.get("player") and self._row.get("ranker", "").strip().isdigit():
                self.rows.append(self._row)
            self._row = None


def parse_contracts_html(html: str) -> pd.DataFrame:
    """Parse the Basketball Reference ``/contracts/players.html`` table.

    Returns:
        One row per table row with ``player``, ``bbref_id``, ``team``, ``remain_gtd`` and
        one text column per salary season (e.g. ``2026-27``), values as shown on the page.
        Repeated header rows inside the table are dropped; nothing else is.
    """
    p = _ContractsParser()
    p.feed(html)
    df = pd.DataFrame(p.rows).rename(columns={"team_id": "team", **p.season_labels})
    keep = ["player", "bbref_id", "team", *p.season_labels.values(), "remain_gtd"]
    return df[[c for c in keep if c in df.columns]]


def html_to_csv(html_file: Path) -> Path:
    """Write ``<html_file>.csv`` next to a cached contracts page and return its path."""
    out = html_file.with_suffix(".csv")
    parse_contracts_html(html_file.read_text()).to_csv(out, index=False)
    return out
