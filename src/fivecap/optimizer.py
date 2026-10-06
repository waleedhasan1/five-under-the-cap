"""Best five players under a budget according to Model B (PROJECT_PLAN.md 7.6).

With pair terms this is a quadratic knapsack. It is solved exactly with CP-SAT using
linearized pair products. Out-of-distribution lineups are excluded by solving repeatedly:
each solution is checked, then cut off with a no-good constraint, until enough
in-distribution lineups are found.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

log = logging.getLogger(__name__)


def prefilter(salary: np.ndarray, rating: np.ndarray, group: np.ndarray, count: int) -> np.ndarray:
    """Mask of players kept after removing dominated ones.

    A player is dominated when at least ``count`` players with the same top archetype cost no
    more and rate at least as high (strictly better on one). This is a heuristic: pair terms
    depend on the full membership vector, not only the top archetype.
    """
    keep = np.ones(len(salary), dtype=bool)
    for i in range(len(salary)):
        same = group == group[i]
        better = (
            same
            & (salary <= salary[i])
            & (rating >= rating[i])
            & ((salary < salary[i]) | (rating > rating[i]))
        )
        keep[i] = better.sum() < count
    return keep


@dataclass
class Solution:
    players: tuple[int, ...]  # indices into the arrays passed to ``top_lineups``
    score: float
    optimal: bool


def top_lineups(
    effects: np.ndarray,
    pair_value: np.ndarray,
    salary: np.ndarray,
    budget: float,
    *,
    n_top: int,
    accept: Callable[[tuple[int, ...]], bool],
    max_rounds: int,
    seconds: float,
    scale: float,
    seed: int,
) -> tuple[list[Solution], int]:
    """Best ``n_top`` accepted lineups of five, in descending score.

    Args:
        effects: ``(n,)`` per-player score terms.
        pair_value: ``(n, n)`` symmetric pair contributions; a lineup scores
            ``sum(effects) + sum_{i<j} pair_value[i, j]``.
        salary: ``(n,)`` salaries; total must be ``<= budget``.
        accept: Called on every solution; rejected ones are cut and skipped.
        max_rounds: Solve at most this many times.
        seconds: Time limit per solve.
        scale: Objective multiplier before rounding to integers.

    Returns:
        ``(solutions, rejected_count)``.
    """
    n = len(effects)
    m = cp_model.CpModel()
    x = [m.new_bool_var(f"x{i}") for i in range(n)]
    m.add(sum(x) == 5)
    m.add(sum(int(round(s)) * xi for s, xi in zip(salary, x, strict=True)) <= int(budget))
    terms = [int(round(scale * e)) * xi for e, xi in zip(effects, x, strict=True)]
    for i in range(n):
        for j in range(i + 1, n):
            q = int(round(scale * pair_value[i, j]))
            if q == 0:
                continue
            y = m.new_bool_var(f"y{i}_{j}")
            m.add_implication(y, x[i])
            m.add_implication(y, x[j])
            m.add_bool_or([y, x[i].Not(), x[j].Not()])
            terms.append(q * y)
    m.maximize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = 8

    found: list[Solution] = []
    rejected = 0
    for _ in range(max_rounds):
        status = solver.solve(m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        chosen = tuple(i for i in range(n) if solver.value(x[i]))
        score = float(effects[list(chosen)].sum()) + sum(
            float(pair_value[a, b]) for k, a in enumerate(chosen) for b in chosen[k + 1 :]
        )
        if accept(chosen):
            found.append(Solution(chosen, score, status == cp_model.OPTIMAL))
        else:
            rejected += 1
        m.add(sum(x[i] for i in chosen) <= 4)
        if len(found) == n_top:
            break
    if any(not s.optimal for s in found):
        log.warning("some solves hit the time limit; those lineups are best found, not proven")
    return found, rejected
