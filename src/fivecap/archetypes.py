"""Player archetypes from a Gaussian mixture over player-season style features.

PROJECT_PLAN.md 7.1. The mixture is fit on training-season player-seasons with at least
``archetype_min_minutes`` minutes; soft memberships are then computed for every
player-season (low-minute memberships are noisier and flagged with ``in_fit_sample``).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from fivecap.config import ROOT, config_hash, load_settings, path

log = logging.getLogger("fivecap.archetypes")


@dataclass
class ArchetypeModel:
    """A fitted scaler + mixture and the feature columns it expects."""

    features: list[str]
    fill_values: dict[str, float]
    scaler: StandardScaler
    gmm: GaussianMixture

    def memberships(self, df: pd.DataFrame) -> np.ndarray:
        """Soft membership matrix (rows sum to 1) for player-seasons in ``df``."""
        x = prepare_features(df, self.features, self.fill_values)
        return self.gmm.predict_proba(self.scaler.transform(x))


def prepare_features(
    df: pd.DataFrame, features: list[str], fill_values: dict[str, float]
) -> np.ndarray:
    """Feature matrix with missing play-type shares set to 0 and other gaps filled.

    A player missing from a Synergy play-type list gets 0 for that type when they appear
    in at least one other list. Anything still missing (no Synergy rows at all, or no
    field-goal attempts) takes the training value in ``fill_values``.
    """
    x = df[features].astype(float).copy()
    pt = [c for c in features if c.startswith("pt_")]
    if pt:
        has_any = x[pt].notna().any(axis=1)
        x.loc[has_any, pt] = x.loc[has_any, pt].fillna(0.0)
    return x.fillna(fill_values).to_numpy()


def fill_values_from(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Training medians used to fill gaps (play types computed after the 0-fill)."""
    x = df[features].astype(float).copy()
    pt = [c for c in features if c.startswith("pt_")]
    if pt:
        has_any = x[pt].notna().any(axis=1)
        x.loc[has_any, pt] = x.loc[has_any, pt].fillna(0.0)
    return {c: float(x[c].median()) for c in features}


def fit_gmm(x: np.ndarray, k: int, cov: str, seed: int, cfg: dict[str, Any]) -> GaussianMixture:
    """One Gaussian mixture fit with the configured restarts and covariance floor."""
    return GaussianMixture(
        n_components=k,
        covariance_type=cov,  # type: ignore[arg-type]
        n_init=cfg["n_init"],
        reg_covar=cfg["reg_covar"],
        random_state=seed,
    ).fit(x)


def bic_grid(x: np.ndarray, cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """BIC for every (covariance type, k) in the configured grid."""
    lo, hi = cfg["k_range"]
    rows = []
    for cov in cfg["covariance_types"]:
        for k in range(lo, hi + 1):
            g = fit_gmm(x, k, cov, seed, cfg)
            rows.append(
                {"covariance_type": cov, "k": k, "bic": g.bic(x), "converged": g.converged_}
            )
            log.info("BIC %s k=%d: %.0f", cov, k, rows[-1]["bic"])
    return pd.DataFrame(rows)


def stability(
    x: np.ndarray, k: int, cov: str, cfg: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit with several seeds: hard-label agreement with the reference fit, and BIC choice.

    Returns:
        (per-seed adjusted Rand index at the chosen k, per-seed BIC-optimal k)
    """
    ref = fit_gmm(x, k, cov, seed, cfg).predict(x)
    lo, hi = cfg["k_range"]
    ari, best_k = [], []
    for s in range(seed + 1, seed + 1 + cfg["stability_seeds"]):
        ari.append(
            {"seed": s, "ari": adjusted_rand_score(ref, fit_gmm(x, k, cov, s, cfg).predict(x))}
        )
        bics = {kk: fit_gmm(x, kk, cov, s, cfg).bic(x) for kk in range(lo, hi + 1)}
        best_k.append({"seed": s, "best_k": min(bics, key=bics.__getitem__)})
    return pd.DataFrame(ari), pd.DataFrame(best_k)


def describe(
    df: pd.DataFrame, m: np.ndarray, features: list[str], x_std: np.ndarray, top_n: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membership-weighted centroids (raw units and z-scores) and top members per archetype."""
    summary, top = [], []
    for a in range(m.shape[1]):
        w = m[:, a]
        raw = (
            df[features].astype(float).fillna(df[features].median()).to_numpy() * w[:, None]
        ).sum(0) / w.sum()
        z = (x_std * w[:, None]).sum(0) / w.sum()
        order = np.argsort(-np.abs(z))[:3]
        draft = ", ".join(f"{'high' if z[i] > 0 else 'low'} {features[i]}" for i in order)
        summary.append(
            {"archetype": a, "share": w.mean(), "draft_descriptor": draft}
            | {f: v for f, v in zip(features, raw, strict=True)}
            | {f"z_{f}": v for f, v in zip(features, z, strict=True)}
        )
        idx = np.argsort(-w)[:top_n]
        for rank, i in enumerate(idx, 1):
            r = df.iloc[i]
            top.append(
                {
                    "archetype": a,
                    "rank": rank,
                    "player_name": r["player_name"],
                    "season": r["season"],
                    "membership": w[i],
                }
            )
    return pd.DataFrame(summary), pd.DataFrame(top)


def bic_figure(bic: pd.DataFrame) -> go.Figure:
    """BIC against k, one line per covariance type (lower is better)."""
    fig = go.Figure()
    for cov, g in bic.groupby("covariance_type"):
        fig.add_trace(go.Scatter(x=g["k"], y=g["bic"], mode="lines+markers", name=str(cov)))
    fig.update_layout(
        title="Archetype mixture: BIC by number of components (lower is better)",
        xaxis_title="components (k)",
        yaxis_title="BIC",
        template="plotly_white",
    )
    return fig


def main() -> int:
    """Fit archetypes, write memberships to the warehouse and diagnostics to ``reports/``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = load_settings()
    cfg = s["archetypes"]
    seed = s["random_seed"]
    reports = path("reports")

    with duckdb.connect(str(path("warehouse"))) as con:
        all_ps = con.execute("SELECT * FROM fct_player_season ORDER BY season, player_id").fetchdf()
        pt_present = [c for c in cfg["play_type_features"] if c in all_ps.columns]
        features = cfg["features"] + pt_present
        if len(pt_present) < len(cfg["play_type_features"]):
            log.warning(
                "play-type features missing; fitting without %s",
                sorted(set(cfg["play_type_features"]) - set(pt_present)),
            )

        train = all_ps[
            all_ps["season"].isin(s["seasons"]["train"])
            & (all_ps["minutes"] >= s["thresholds"]["archetype_min_minutes"])
        ].reset_index(drop=True)
        fills = fill_values_from(train, features)
        no_pt = int(train[pt_present].isna().all(axis=1).sum()) if pt_present else 0
        log.info("fit sample: %d player-seasons; %d without any play-type rows", len(train), no_pt)

        scaler = StandardScaler().fit(prepare_features(train, features, fills))
        x = scaler.transform(prepare_features(train, features, fills))

        bic = bic_grid(x, cfg, seed)
        best = bic.sort_values("bic").iloc[0]
        k, cov = int(best["k"]), str(best["covariance_type"])
        log.info("chosen: k=%d, covariance=%s", k, cov)

        ari, best_k = stability(x, k, cov, cfg, seed)
        log.info(
            "stability: ARI mean %.2f (min %.2f); BIC-best k by seed %s",
            ari["ari"].mean(),
            ari["ari"].min(),
            best_k["best_k"].value_counts().to_dict(),
        )

        model = ArchetypeModel(features, fills, scaler, fit_gmm(x, k, cov, seed, cfg))
        m_train = model.memberships(train)
        summary, top = describe(train, m_train, features, x)

        m_all = model.memberships(all_ps)
        out = all_ps[["season", "player_id", "player_name", "minutes"]].copy()
        out["in_fit_sample"] = all_ps["season"].isin(s["seasons"]["train"]) & (
            all_ps["minutes"] >= s["thresholds"]["archetype_min_minutes"]
        )
        for a in range(k):
            out[f"arch_{a}"] = m_all[:, a]
        con.register("_m", out)
        con.execute("CREATE OR REPLACE TABLE fct_archetype_membership AS SELECT * FROM _m")
        con.unregister("_m")

    models = ROOT / "data" / "models"
    models.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models / "archetypes.joblib")
    bic.to_csv(reports / "archetype_bic.csv", index=False)
    bic_figure(bic).write_html(reports / "archetype_bic.html", include_plotlyjs="cdn")
    summary.round(4).to_csv(reports / "archetype_summary.csv", index=False)
    top.round(3).to_csv(reports / "archetype_top_members.csv", index=False)
    pd.concat([ari, best_k.drop(columns="seed")], axis=1).round(3).to_csv(
        reports / "archetype_stability.csv", index=False
    )
    meta = {
        "training_seasons": s["seasons"]["train"],
        "n_player_seasons": len(train),
        "features": features,
        "k": k,
        "covariance_type": cov,
        "bic": float(best["bic"]),
        "stability_ari_mean": float(ari["ari"].mean()),
        "stability_ari_min": float(ari["ari"].min()),
        "bic_best_k_by_seed": best_k["best_k"].value_counts().sort_index().to_dict(),
        "seed": seed,
        "config_hash": config_hash(),
        "date": date.today().isoformat(),
    }
    (reports / "archetypes_meta.json").write_text(json.dumps(meta, indent=2, default=int))
    return 0


if __name__ == "__main__":
    sys.exit(main())
