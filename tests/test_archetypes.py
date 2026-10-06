"""Archetype helpers on small hand-built FIXTURE frames (not real data)."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fivecap.archetypes import ArchetypeModel, fill_values_from, fit_gmm, prepare_features

FEATURES = ["usg_pct", "pt_cut", "pt_spotup"]


def test_play_type_gaps_are_zero_when_player_has_any_rows() -> None:
    df = pd.DataFrame(
        {
            "usg_pct": [0.2, np.nan, 0.3],
            "pt_cut": [0.4, np.nan, np.nan],  # row 0: listed in cut; row 2: no Synergy rows
            "pt_spotup": [np.nan, 0.5, np.nan],
        }
    )
    fills = {"usg_pct": 0.25, "pt_cut": 0.1, "pt_spotup": 0.2}
    x = prepare_features(df, FEATURES, fills)
    np.testing.assert_allclose(x, [[0.2, 0.4, 0.0], [0.25, 0.0, 0.5], [0.3, 0.1, 0.2]])


def test_fill_values_use_zero_filled_play_types() -> None:
    df = pd.DataFrame({"usg_pct": [0.1, 0.3], "pt_cut": [0.4, np.nan], "pt_spotup": [0.6, 1.0]})
    assert fill_values_from(df, FEATURES)["pt_cut"] == 0.2


def test_memberships_are_probability_vectors() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((60, 3)), columns=FEATURES)
    fills = fill_values_from(df, FEATURES)
    scaler = StandardScaler().fit(prepare_features(df, FEATURES, fills))
    cfg = {"n_init": 1, "reg_covar": 1e-3}
    pca = PCA(n_components=2).fit(scaler.transform(prepare_features(df, FEATURES, fills)))
    x = pca.transform(scaler.transform(prepare_features(df, FEATURES, fills)))
    gmm = fit_gmm(x, 3, "diag", 0, cfg)
    m = ArchetypeModel(FEATURES, fills, scaler, pca, gmm).memberships(df)
    assert m.shape == (60, 3)
    np.testing.assert_allclose(m.sum(axis=1), 1.0)
