"""Lineup prediction used by both the offline export and the app.

The app must reproduce Model B exactly with a few dot products (PROJECT_PLAN.md section 8),
so the pair terms are rewritten as a symmetric matrix ``G``: for players ``i`` and ``j`` the
fit contribution is ``m_i^T G m_j``, and a lineup's fit is the sum over its ten pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fivecap.lineup_model import pair_index


def pair_matrix(pair_coef: np.ndarray, k: int) -> np.ndarray:
    """Symmetric ``G`` with ``G[a, a] = coef(a, a)`` and ``G[a, b] = G[b, a] = coef(a, b)``.

    Then ``sum_{i<j} m_i^T G m_j`` equals ``pair_features(m) @ pair_coef``.
    """
    g = np.zeros((k, k))
    for c, (a, b) in zip(pair_coef, pair_index(k), strict=True):
        g[a, b] = g[b, a] = c
    return g


def pair_grid(m5: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``5 x 5`` matrix of pairwise fit contributions ``m_i^T G m_j`` (diagonal zero)."""
    grid = m5 @ g @ m5.T
    np.fill_diagonal(grid, 0.0)
    return grid


def lineup_fit(m: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Raw fit for lineups. ``m`` is ``(n, 5, k)``; returns ``(n,)`` (sum over the ten pairs)."""
    full = np.einsum("nia,ab,njb->nij", m, g, m)
    return (full.sum(axis=(1, 2)) - np.einsum("nii->n", full)) / 2.0


@dataclass
class GameModel:
    """Everything needed to score a lineup of pool players.

    Attributes:
        effects: ``(n_players,)`` Model B player effects for the pool.
        memberships: ``(n_players, k)`` archetype memberships for the pool.
        intercept: Model B intercept.
        g: ``(k, k)`` pair matrix.
        fit_const: Fit of the mean training lineup; subtracted so fit 0 means "average fit".
        boot_effects, boot_intercept, boot_g, boot_fit_const: bootstrap refits (``R`` rows).
        tau2_resid: Unexplained lineup variance from CV (points^2 per 100).
        z: Normal quantile for the interval (1.2816 for 80%).
        observed: ``(n_obs, k)`` archetype compositions of observed lineups.
        ood_threshold: Nearest-neighbour distance beyond which a lineup is out of distribution.
        ood_multiplier: Interval widening for out-of-distribution lineups.
    """

    effects: np.ndarray
    memberships: np.ndarray
    intercept: float
    g: np.ndarray
    fit_const: float
    boot_effects: np.ndarray
    boot_intercept: np.ndarray
    boot_g: np.ndarray
    boot_fit_const: np.ndarray
    tau2_resid: float
    z: float
    observed: np.ndarray
    ood_threshold: float
    ood_multiplier: float

    def predict(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Score lineups given as pool row indices, shape ``(n, 5)``.

        Returns:
            Dict of ``(n,)`` arrays: ``talent``, ``fit``, ``total``, ``low``, ``high``,
            ``ood_distance`` and ``ood`` (bool). The interval is for the lineup's underlying
            net rating (model uncertainty plus unexplained lineup variance), not for a noisy
            observed sample.
        """
        idx = np.atleast_2d(idx)
        m = self.memberships[idx]
        talent = self.intercept + self.effects[idx].sum(axis=1)
        fit = lineup_fit(m, self.g) - self.fit_const
        boot = (
            self.boot_intercept[None, :]
            + self.boot_effects[:, idx].sum(axis=2).T
            + np.stack([lineup_fit(m, gb) for gb in self.boot_g], axis=1)
            - self.boot_fit_const[None, :]
        )
        half = self.z * np.sqrt(boot.var(axis=1) + self.tau2_resid)
        dist = self.ood_distance(m.sum(axis=1))
        ood = dist > self.ood_threshold
        half = np.where(ood, half * self.ood_multiplier, half)
        total = talent + fit
        return {
            "talent": talent,
            "fit": fit,
            "total": total,
            "low": total - half,
            "high": total + half,
            "ood_distance": dist,
            "ood": ood,
        }

    def ood_distance(self, comps: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """Euclidean distance from each composition to the nearest observed composition."""
        out = np.empty(len(comps))
        for s in range(0, len(comps), chunk):
            c = comps[s : s + chunk]
            d2 = ((c[:, None, :] - self.observed[None, :, :]) ** 2).sum(axis=2)
            out[s : s + chunk] = np.sqrt(d2.min(axis=1))
        return out
