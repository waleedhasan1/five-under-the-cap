"""Game prediction, optimizer and valuation tests on small hand-built FIXTURES (not real data)."""

import itertools

import numpy as np
import pandas as pd
import pytest

from fivecap.game import GameModel, lineup_fit, pair_grid, pair_matrix
from fivecap.lineup_model import LineupModel, pair_features
from fivecap.optimizer import prefilter, top_lineups
from fivecap.valuation import ValuationInputs, minimum_salary, player_values

K = 3


def _fixture(n: int = 9, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 2, n), rng.dirichlet(np.ones(K), n), rng.normal(0, 1, K * (K + 1) // 2)


def test_pair_matrix_reproduces_pair_features() -> None:
    _, memb, coef = _fixture()
    m = memb[np.array([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])]
    g = pair_matrix(coef, K)
    np.testing.assert_allclose(lineup_fit(m, g), pair_features(m) @ coef)
    np.testing.assert_allclose(pair_grid(m[0], g).sum() / 2, pair_features(m[:1]) @ coef)


def test_app_prediction_matches_model() -> None:
    effects, memb, coef = _fixture()
    pair_mean = np.full(len(coef), 0.4)
    ids = np.arange(100, 109)
    model = LineupModel(True, 1.5, pd.Series(effects, index=ids), 0.0, coef, pair_mean)
    g = pair_matrix(coef, K)
    game = GameModel(
        effects=effects,
        memberships=memb,
        intercept=1.5,
        g=g,
        fit_const=float(pair_mean @ coef),
        boot_effects=np.tile(effects, (4, 1)),
        boot_intercept=np.full(4, 1.5),
        boot_g=np.stack([g] * 4),
        boot_fit_const=np.full(4, float(pair_mean @ coef)),
        tau2_resid=4.0,
        z=1.2816,
        observed=memb[:5].sum(axis=0, keepdims=True),
        ood_threshold=10.0,
        ood_multiplier=1.5,
    )
    idx = np.array([[0, 1, 2, 3, 4], [2, 4, 6, 7, 8]])
    talent, fit = model.predict(ids[idx], memb[idx])
    out = game.predict(idx)
    np.testing.assert_allclose(out["talent"], talent)
    np.testing.assert_allclose(out["fit"], fit)
    # identical bootstrap draws -> half-width comes only from tau2_resid
    np.testing.assert_allclose(out["high"] - out["total"], 1.2816 * 2.0)
    assert out["ood_distance"][0] == pytest.approx(0.0) and not out["ood"].any()


def _brute(effects, pv, salary, budget):  # type: ignore[no-untyped-def]
    res = []
    for c in itertools.combinations(range(len(effects)), 5):
        if salary[list(c)].sum() <= budget:
            res.append(
                (effects[list(c)].sum() + sum(pv[a, b] for a, b in itertools.combinations(c, 2)), c)
            )
    return sorted(res, reverse=True)


def test_optimizer_matches_brute_force_and_respects_constraints() -> None:
    effects, memb, coef = _fixture(n=10, seed=3)
    pv = memb @ pair_matrix(coef, K) @ memb.T
    salary = np.linspace(1e6, 10e6, 10)
    budget = 30e6
    reject = {
        _brute(effects, pv, salary, budget)[1][1]
    }  # pretend the 2nd-best is out of distribution
    sols, rejected = top_lineups(
        effects,
        pv,
        salary,
        budget,
        n_top=3,
        accept=lambda c: c not in reject,
        max_rounds=20,
        seconds=10,
        scale=1e5,
        seed=0,
    )
    expected = [c for _, c in _brute(effects, pv, salary, budget) if c not in reject][:3]
    assert [s.players for s in sols] == expected
    assert rejected == 1
    for s in sols:
        assert len(set(s.players)) == 5
        assert salary[list(s.players)].sum() <= budget


def test_prefilter_drops_only_dominated() -> None:
    salary = np.array([1, 1, 1, 5, 9.0])
    rating = np.array([3, 3, 3, 1, 2.0])
    group = np.array([0, 0, 0, 0, 1])
    # player 3 is beaten by three same-group players; player 4 is alone in its group
    np.testing.assert_array_equal(prefilter(salary, rating, group, count=3), [1, 1, 1, 0, 1])


def test_valuation_arithmetic() -> None:
    v = ValuationInputs(
        wins_per_point=2.5,
        wins_at_zero=41,
        replacement_effect=-2,
        average_effect=0,
        starter_share=0.7,
        min_salary=1e6,
        roster_size=15,
        league_payroll=5e9,
        league_wins=1230,
        n_teams=30,
    )
    assert v.replacement_team_wins == pytest.approx(41 + 2.5 * -10)
    assert v.price_per_win == pytest.approx((5e9 - 30 * 15 * 1e6) / (1230 - 30 * 16))
    out = player_values(np.array([-2.0, 2.0]), np.array([1e6, 1e6]), v)
    assert out["wins_added"].tolist() == pytest.approx([0.0, 2.5 * 0.7 * 4])
    assert out["surplus"].iloc[0] == pytest.approx(-1e6)


def test_minimum_salary_ignores_partial_deals() -> None:
    s = pd.Series([1e5, 2e5, 1.3e6, 1.3e6, 1.3e6] + [5e6] * 45)
    assert minimum_salary(s, 0.1) == 1.3e6
