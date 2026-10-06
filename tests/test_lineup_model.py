"""Lineup model tests on small hand-built FIXTURES (not real data)."""

import itertools

import numpy as np
import pandas as pd
import pytest

from fivecap.lineup_model import (
    Memberships,
    fit_lineup_model,
    pair_features,
    pair_index,
    recency_weights,
)


def test_pair_features_hand_checked() -> None:
    # Two archetypes. Players 0-2 are pure archetype 0, player 3 pure 1, player 4 half/half.
    m = np.array([[[1, 0], [1, 0], [1, 0], [0, 1], [0.5, 0.5]]], dtype=float)
    f = pair_features(m)[0]
    # (0,0): pairs among {0,1,2} -> 3, plus each of 0-2 with player 4 -> 3 * 0.5 = 1.5
    # (0,1): each of 0-2 with player 3 -> 3, each of 0-2 with player 4 -> 1.5,
    #        player 3 with player 4 -> 0.5            -> total 5.0
    # (1,1): player 3 with player 4 -> 0.5
    np.testing.assert_allclose(f, [4.5, 5.0, 0.5])
    assert pair_index(2) == [(0, 0), (0, 1), (1, 1)]
    assert f.sum() == pytest.approx(10.0)  # ten player pairs, memberships sum to 1


def test_pair_features_match_brute_force() -> None:
    rng = np.random.default_rng(1)
    m = rng.dirichlet(np.ones(4), size=(3, 5))
    brute = np.zeros((3, 10))
    for n in range(3):
        for col, (a, b) in enumerate(pair_index(4)):
            v = sum(
                m[n, i, a] * m[n, j, b] + m[n, i, b] * m[n, j, a]
                for i, j in itertools.combinations(range(5), 2)
            )
            brute[n, col] = v / 2 if a == b else v
    np.testing.assert_allclose(pair_features(m), brute)


def test_prior_only_lookup_never_uses_target_season() -> None:
    t = pd.DataFrame(
        {
            "season": ["2023-24", "2024-25", "2024-25"],
            "player_id": [1, 1, 2],
            "a0": [0.9, 0.2, 0.7],
            "a1": [0.1, 0.8, 0.3],
        }
    ).set_index(["season", "player_id"])
    memb = Memberships(t, fallback=np.array([0.5, 0.5]))
    prior = memb.lookup(np.array([1, 2]), "2024-25", prior_only=True)
    np.testing.assert_allclose(prior, [[0.9, 0.1], [0.5, 0.5]])  # player 2 has no prior season
    same = memb.lookup(np.array([1, 2]), "2024-25", prior_only=False)
    np.testing.assert_allclose(same, [[0.2, 0.8], [0.7, 0.3]])


def test_recency_weights_reject_target_season_rows() -> None:
    w = recency_weights(pd.Series(["2022-23", "2023-24"]), "2024-25", 0.5)
    np.testing.assert_allclose(w, [0.5, 1.0])
    with pytest.raises(ValueError):
        recency_weights(pd.Series(["2024-25"]), "2024-25", 0.5)


def test_model_a_recovers_player_effects() -> None:
    rng = np.random.default_rng(0)
    true = {p: e for p, e in zip(range(10), np.linspace(-5, 5, 10), strict=True)}
    rows = []
    for _ in range(400):
        five = sorted(rng.choice(10, 5, replace=False))
        rows.append(
            {
                "season": "2023-24",
                **{f"p{i + 1}": p for i, p in enumerate(five)},
                "possessions": 100,
                "net_rating": sum(true[p] for p in five),
            }
        )
    df = pd.DataFrame(rows)
    model = fit_lineup_model(
        df,
        None,
        target_season="2024-25",
        alpha_player=0.01,
        alpha_fit=None,
        decay=1.0,
        unseen_quantile=0.25,
    )
    est = model.player_effects - model.player_effects.mean()
    np.testing.assert_allclose(est.to_numpy(), np.linspace(-5, 5, 10), atol=0.05)


def test_time_split_has_no_leakage() -> None:
    from fivecap.evaluate import make_split

    df = pd.DataFrame(
        {"season": ["2022-23", "2023-24", "2024-25", "2024-25"], "possessions": [500, 500, 500, 50]}
    )
    sp = make_split(df, "2024-25", ["2022-23", "2023-24", "2024-25"], min_poss=100)
    assert set(sp.train["season"]) == {"2022-23", "2023-24"}
    assert len(sp.test) == 1 and (sp.test["season"] == "2024-25").all()


def test_unseen_players_get_unseen_effect() -> None:
    from fivecap.lineup_model import LineupModel

    model = LineupModel(False, 1.0, pd.Series([2.0, -1.0], index=[10, 11]), unseen_effect=-0.5)
    talent, fit = model.predict(np.array([[10, 11, 12, 13, 14]]), None)
    assert talent[0] == pytest.approx(1.0 + 2.0 - 1.0 - 1.5)
    assert fit[0] == 0.0
