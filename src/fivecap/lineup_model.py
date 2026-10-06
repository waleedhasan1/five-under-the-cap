"""Lineup models: 0 (league mean), A (sum of player effects), B (A plus archetype-pair fit terms).

PROJECT_PLAN.md 7.2. Target is lineup net rating per 100 possessions; every fit is weighted by
possessions times a recency weight. Model B's fit terms get their own ridge penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

PLAYER_COLS = ["p1", "p2", "p3", "p4", "p5"]


def pair_index(k: int) -> list[tuple[int, int]]:
    """Archetype pairs ``(a, b)`` with ``a <= b`` in feature-column order."""
    return [(a, b) for a in range(k) for b in range(a, k)]


def pair_features(m: np.ndarray) -> np.ndarray:
    """Archetype-pair fit features for lineups.

    Args:
        m: Membership array of shape ``(n_lineups, 5, k)``; each row of ``m[i]`` sums to 1.

    Returns:
        Array ``(n_lineups, k(k+1)/2)``. For ``a <= b`` the feature is the sum over the ten
        player pairs ``i < j`` of ``m_i[a] m_j[b] + m_i[b] m_j[a]``, halved when ``a == b``.
        Uses ``sum_{i != j} m_i m_j^T = S S^T - sum_i m_i m_i^T`` with ``S = sum_i m_i``.
    """
    s = m.sum(axis=1)
    p = np.einsum("na,nb->nab", s, s) - np.einsum("nia,nib->nab", m, m)
    k = m.shape[2]
    a, b = np.triu_indices(k)
    out = p[:, a, b]
    return np.where(a == b, out / 2.0, out)


@dataclass
class Memberships:
    """Archetype memberships per player-season, with a fallback for players never seen.

    Attributes:
        table: Index ``(season, player_id)``, one column per archetype.
        fallback: Membership used when a player has no eligible season.
    """

    table: pd.DataFrame
    fallback: np.ndarray

    @property
    def k(self) -> int:
        return self.table.shape[1]

    def lookup(self, player_ids: np.ndarray, season: str, *, prior_only: bool) -> np.ndarray:
        """Membership vectors for players as of ``season``.

        With ``prior_only`` the player's latest season strictly before ``season`` is used
        (no information from ``season`` itself); otherwise the same season is used.
        Players with no eligible row get ``fallback``.
        """
        t = self.table.reset_index()
        t = t[t["season"] < season] if prior_only else t[t["season"] == season]
        latest = t.sort_values("season").groupby("player_id").last().drop(columns="season")
        out = latest.reindex(player_ids)
        missing = out.isna().any(axis=1).to_numpy()
        arr = out.to_numpy(dtype=float, copy=True)
        arr[missing] = self.fallback
        return arr


def lineup_memberships(df: pd.DataFrame, memb: Memberships, *, prior_only: bool) -> np.ndarray:
    """``(n_lineups, 5, k)`` memberships for lineups, looked up season by season."""
    out = np.empty((len(df), 5, memb.k))
    for season, idx in df.groupby("season").indices.items():
        ids = df[PLAYER_COLS].to_numpy()[idx]
        flat = memb.lookup(ids.ravel(), str(season), prior_only=prior_only)
        out[idx] = flat.reshape(len(idx), 5, memb.k)
    return out


def recency_weights(seasons: pd.Series, target_season: str, decay: float) -> np.ndarray:
    """``decay ** (age - 1)`` where age is how many seasons before ``target_season``."""
    order = sorted(set(seasons) | {target_season})
    pos = {s: i for i, s in enumerate(order)}
    age = pos[target_season] - seasons.map(pos).to_numpy()
    if (age <= 0).any():
        raise ValueError("training rows must come from seasons before the target season")
    return decay ** (age - 1).astype(float)


@dataclass
class LineupModel:
    """A fitted Model A (``with_fit=False``) or Model B (``with_fit=True``).

    ``predict`` returns talent and fit parts separately; their sum is the prediction.
    The fit part is centred on the mean training lineup, so 0 means "average fit".
    """

    with_fit: bool
    intercept: float
    player_effects: pd.Series  # indexed by player_id
    unseen_effect: float
    pair_coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pair_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def effects(self, player_ids: np.ndarray) -> np.ndarray:
        """Player effects, with ``unseen_effect`` for players absent from training."""
        e = self.player_effects.reindex(player_ids.ravel()).to_numpy(dtype=float, copy=True)
        e[np.isnan(e)] = self.unseen_effect
        return e.reshape(player_ids.shape)

    def predict(
        self, player_ids: np.ndarray, m: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Talent and fit parts for lineups.

        Args:
            player_ids: ``(n, 5)`` player IDs.
            m: ``(n, 5, k)`` memberships (ignored for Model A).

        Returns:
            ``(talent, fit)``, each of shape ``(n,)``.
        """
        talent = self.intercept + self.effects(player_ids).sum(axis=1)
        if not self.with_fit or m is None:
            return talent, np.zeros(len(player_ids))
        return talent, (pair_features(m) - self.pair_mean) @ self.pair_coef


def fit_lineup_model(
    train: pd.DataFrame,
    m_train: np.ndarray | None,
    *,
    target_season: str,
    alpha_player: float,
    alpha_fit: float | None,
    decay: float,
    unseen_quantile: float,
) -> LineupModel:
    """Fit Model A (``m_train`` is None) or Model B by possession-weighted ridge.

    Args:
        train: Lineup rows (``season``, ``p1..p5``, ``possessions``, ``net_rating``), all
            from seasons before ``target_season``.
        m_train: ``(n, 5, k)`` memberships for Model B, else None.
        target_season: Season being predicted; sets the recency weights.
        alpha_player: Ridge penalty on player effects.
        alpha_fit: Ridge penalty on pair coefficients (Model B only).
        decay: Recency weight multiplier per season of age.
        unseen_quantile: Players whose training possessions fall in this lower quantile
            define the effect given to players absent from training.
    """
    ids = train[PLAYER_COLS].to_numpy()
    players, inv = np.unique(ids, return_inverse=True)
    n = len(train)
    x_players = sparse.csr_matrix(
        (np.ones(5 * n), (np.repeat(np.arange(n), 5), inv.ravel())), shape=(n, len(players))
    )
    w = train["possessions"].to_numpy(float) * recency_weights(
        train["season"], target_season, decay
    )
    w = w / w.mean()

    with_fit = m_train is not None
    if m_train is not None:
        assert alpha_fit is not None
        f = pair_features(m_train)
        f_mean = np.average(f, axis=0, weights=w)
        scale = np.sqrt(alpha_player / alpha_fit)  # gives the pair block penalty alpha_fit
        x = sparse.hstack([x_players, sparse.csr_matrix((f - f_mean) * scale)]).tocsr()
    else:
        x = x_players

    ridge = Ridge(alpha=alpha_player, solver="sparse_cg", tol=1e-6, max_iter=5000)
    ridge.fit(x, train["net_rating"].to_numpy(float), sample_weight=w)
    effects = pd.Series(ridge.coef_[: len(players)], index=players)

    poss = pd.Series(np.repeat(train["possessions"].to_numpy(float), 5), index=ids.ravel())
    poss = poss.groupby(level=0).sum().reindex(players)
    low = poss <= poss.quantile(unseen_quantile)
    unseen = float(np.average(effects[low], weights=poss[low]))

    model = LineupModel(with_fit, float(ridge.intercept_), effects, unseen)
    if with_fit:
        model.pair_coef = ridge.coef_[len(players) :] * scale
        model.pair_mean = f_mean
    return model
