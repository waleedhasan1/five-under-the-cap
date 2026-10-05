"""Read hand-exported Basketball Reference salary CSVs from ``data/raw/salaries/``.

No scraping: the owner exports the contracts table by hand (PROJECT_PLAN.md section 6).
"""

from __future__ import annotations

import logging
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
