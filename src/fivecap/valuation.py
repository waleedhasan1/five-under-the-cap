"""Player dollar values and surplus (PROJECT_PLAN.md 7.5). Deliberately simple; an approximation.

Chain: player effect (points per 100 vs. average) -> change in team net rating over a
starter's share of possessions, relative to a replacement-level player -> wins (slope of
team wins on team net rating) -> dollars (league price of a win).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ValuationInputs:
    """Every number the valuation uses, all computed from the project's data."""

    wins_per_point: float  # slope of team wins on team net rating
    wins_at_zero: float  # intercept of that regression
    replacement_effect: float  # effect of a replacement-level player (points per 100)
    average_effect: float  # possession-weighted mean player effect
    starter_share: float  # fraction of team possessions a starter is on court
    min_salary: float
    roster_size: float
    league_payroll: float
    league_wins: float
    n_teams: int

    @property
    def replacement_team_wins(self) -> float:
        """Wins for a team of five replacement-level players in every slot."""
        nr = 5 * (self.replacement_effect - self.average_effect)
        return self.wins_at_zero + self.wins_per_point * nr

    @property
    def price_per_win(self) -> float:
        """Payroll above a minimum-salary league divided by wins above a replacement league."""
        spend = self.league_payroll - self.n_teams * self.roster_size * self.min_salary
        wins = self.league_wins - self.n_teams * self.replacement_team_wins
        return spend / wins

    def as_dict(self) -> dict[str, float]:
        return asdict(self) | {
            "replacement_team_wins": self.replacement_team_wins,
            "price_per_win": self.price_per_win,
        }


def wins_regression(team_seasons: pd.DataFrame) -> tuple[float, float]:
    """``(slope, intercept)`` of wins on net rating across team-seasons."""
    fit = stats.linregress(team_seasons["net_rating"], team_seasons["wins"])
    return float(fit.slope), float(fit.intercept)


def minimum_salary(salaries: pd.Series, quantile: float) -> float:
    """Most common salary in the bottom ``quantile`` of the table (the league-minimum tier).

    Lower one-off figures (partial or non-guaranteed deals) are ignored by taking the mode.
    """
    low = salaries[salaries <= salaries.quantile(quantile)]
    return float(low.mode().iloc[0])


def player_values(effects: np.ndarray, salary: np.ndarray, v: ValuationInputs) -> pd.DataFrame:
    """Wins added over a starter's share of possessions, dollar value and surplus.

    Returns:
        Frame with ``wins_added``, ``value`` and ``surplus`` (value minus salary).
    """
    wins = v.wins_per_point * v.starter_share * (effects - v.replacement_effect)
    value = wins * v.price_per_win
    return pd.DataFrame({"wins_added": wins, "value": value, "surplus": value - salary})
