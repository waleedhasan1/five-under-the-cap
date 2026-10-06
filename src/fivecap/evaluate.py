"""Out-of-sample comparison of lineup Models 0, A and B (PROJECT_PLAN.md 7.3 and 7.4).

The test season never touches fitting or tuning: penalties and the recency decay are chosen
by time-blocked CV in which each validation season is predicted from earlier seasons only.
"""

from __future__ import annotations

import itertools
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
from plotly.subplots import make_subplots
from scipy import stats

from fivecap.config import ROOT, config_hash, load_settings, path
from fivecap.lineup_model import (
    PLAYER_COLS,
    LineupModel,
    Memberships,
    fit_lineup_model,
    lineup_memberships,
)

log = logging.getLogger("fivecap.evaluate")


def wrmse(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Weighted root mean squared error."""
    return float(np.sqrt(np.average((y - pred) ** 2, weights=w)))


def wr2(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Weighted R^2 against the weighted mean of ``y``."""
    sst = np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    return float(1 - np.average((y - pred) ** 2, weights=w) / sst)


@dataclass
class Split:
    """Training rows (all seasons before ``target``) and evaluation rows for one target."""

    target: str
    train: pd.DataFrame
    test: pd.DataFrame


def make_split(
    lineups: pd.DataFrame, target: str, train_seasons: list[str], min_poss: int
) -> Split:
    """Time split: train on ``train_seasons`` before ``target``; evaluate ``target`` lineups.

    Raises:
        ValueError: if any training row is from ``target`` or later.
    """
    train = lineups[lineups["season"].isin([s for s in train_seasons if s < target])]
    test = lineups[(lineups["season"] == target) & (lineups["possessions"] >= min_poss)]
    if (train["season"] >= target).any():
        raise ValueError("leakage: training rows from the target season or later")
    return Split(target, train.reset_index(drop=True), test.reset_index(drop=True))


def fit_on(split: Split, memb: Memberships, p: dict[str, Any], cfg: dict[str, Any]) -> LineupModel:
    """Fit Model A (``p['alpha_fit']`` is None) or Model B on a split's training rows."""
    m = None
    if p.get("alpha_fit") is not None:
        m = lineup_memberships(split.train, memb, prior_only=False)
    return fit_lineup_model(
        split.train,
        m,
        target_season=split.target,
        alpha_player=p["alpha_player"],
        alpha_fit=p.get("alpha_fit"),
        decay=p["decay"],
        unseen_quantile=cfg["unseen_player_quantile"],
    )


def predict_on(model: LineupModel, split: Split, memb: Memberships) -> np.ndarray:
    """Predictions for a split's evaluation rows using prior-season memberships only."""
    m = lineup_memberships(split.test, memb, prior_only=True) if model.with_fit else None
    talent, fit = model.predict(split.test[PLAYER_COLS].to_numpy(), m)
    return talent + fit


def tune(
    folds: list[Split], memb: Memberships, cfg: dict[str, Any], with_fit: bool
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Grid search by mean validation weighted RMSE across the CV folds."""
    fits = cfg["alpha_fit_grid"] if with_fit else [None]
    rows = []
    for decay, ap, af in itertools.product(
        cfg["recency_decay_grid"], cfg["alpha_player_grid"], fits
    ):
        p = {"decay": decay, "alpha_player": ap, "alpha_fit": af}
        scores = []
        for f in folds:
            pred = predict_on(fit_on(f, memb, p, cfg), f, memb)
            scores.append(
                wrmse(f.test["net_rating"].to_numpy(), pred, f.test["possessions"].to_numpy())
            )
        rows.append(
            p
            | {f"rmse_{f.target}": s for f, s in zip(folds, scores, strict=True)}
            | {"rmse_mean": float(np.mean(scores))}
        )
    grid = pd.DataFrame(rows).sort_values("rmse_mean")
    best = grid.iloc[0]
    params = {
        "decay": float(best["decay"]),
        "alpha_player": float(best["alpha_player"]),
        "alpha_fit": None if not with_fit else float(best["alpha_fit"]),
    }
    return params, grid


def noise_model(lineups: pd.DataFrame, n_bins: int = 20) -> dict[str, float]:
    """Split lineup net-rating variance into true spread and sampling noise.

    Lineup net rating over ``n`` possessions has variance ``tau2 + c / n``. Bin lineups by
    possessions, take the possession-weighted variance in each bin and regress it on the
    bin's mean ``1/n``. ``c`` is the per-possession noise scale; ``tau2`` the spread of
    true lineup quality. Used in place of split-half reliability, which needs game-level
    lineup data this project does not have.
    """
    d = lineups[lineups["possessions"] >= 10].copy()
    d["bin"] = pd.qcut(d["possessions"], n_bins, duplicates="drop")
    rows = []
    for _, g in d.groupby("bin", observed=True):
        w = g["possessions"].to_numpy(float)
        y = g["net_rating"].to_numpy(float)
        rows.append((np.average(1 / w), np.average((y - np.average(y, weights=w)) ** 2, weights=w)))
    inv_n, var = np.array(rows).T
    fit = stats.linregress(inv_n, var)
    return {"tau2": float(fit.intercept), "c": float(fit.slope), "r2": float(fit.rvalue**2)}


def paired_bootstrap(
    y: np.ndarray, preds: dict[str, np.ndarray], w: np.ndarray, reps: int, seed: int
) -> pd.DataFrame:
    """Weighted RMSE per model for ``reps`` resamples of the test lineups (same draws for all)."""
    rng = np.random.default_rng(seed)
    out = {k: np.empty(reps) for k in preds}
    for r in range(reps):
        i = rng.integers(0, len(y), len(y))
        for k, p in preds.items():
            out[k][r] = wrmse(y[i], p[i], w[i])
    return pd.DataFrame(out)


def comparison_figure(table: pd.DataFrame, boot: pd.DataFrame, floor: float) -> go.Figure:
    """Left: weighted RMSE by model with the noise floor. Right: bootstrap of RMSE(A) - RMSE(B)."""
    fig = make_subplots(
        cols=2,
        subplot_titles=(
            "Test-season weighted RMSE (lower is better)",
            "Bootstrap: RMSE(A) − RMSE(B)",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=table["model"],
            y=table["wrmse"],
            marker_color="#4C78A8",
            text=table["wrmse"].round(2),
            textposition="outside",
            showlegend=False,
        ),
        1,
        1,
    )
    fig.add_hline(
        y=floor,
        line_dash="dash",
        line_color="#888",
        row=1,
        col=1,
        annotation_text=f"noise floor ≈ {floor:.1f}",
        annotation_position="bottom right",
    )
    diff = boot["A"] - boot["B"]
    fig.add_trace(go.Histogram(x=diff, nbinsx=50, marker_color="#4C78A8", showlegend=False), 1, 2)
    fig.add_vline(x=0, line_color="#333", row=1, col=2)
    for q in (0.025, 0.975):
        fig.add_vline(
            x=float(diff.quantile(q)), line_dash="dot", line_color="#E45756", row=1, col=2
        )
    fig.update_layout(template="plotly_white", height=420, width=1000)
    fig.update_yaxes(title_text="points per 100 possessions", row=1, col=1)
    fig.update_xaxes(title_text="points per 100 (positive = fit terms help)", row=1, col=2)
    return fig


def main() -> int:
    """Tune on training seasons, evaluate once on the test season, write ``reports/``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = load_settings()
    cfg = s["lineup_model"]
    seed = s["random_seed"]
    train_seasons, test_season = s["seasons"]["train"], s["seasons"]["test"]
    min_poss = s["thresholds"]["test_min_possessions"]
    reports, models = path("reports"), ROOT / "data" / "models"

    with duckdb.connect(str(path("warehouse")), read_only=True) as con:
        lineups = con.execute("SELECT * FROM fct_lineup_season").fetchdf()
        mt = con.execute("SELECT * FROM fct_archetype_membership").fetchdf()
    arch = [c for c in mt.columns if c.startswith("arch_")]
    fit_rows = mt[mt["in_fit_sample"]]
    fallback = np.average(fit_rows[arch].to_numpy(), axis=0, weights=fit_rows["minutes"])
    memb = Memberships(mt.set_index(["season", "player_id"])[arch], fallback)

    # --- Tuning: time-blocked CV inside the training seasons only.
    folds = [make_split(lineups, v, train_seasons, min_poss) for v in cfg["cv_validation_seasons"]]
    params_a, grid_a = tune(folds, memb, cfg, with_fit=False)
    params_b, grid_b = tune(folds, memb, cfg, with_fit=True)
    log.info("tuned A: %s; tuned B: %s", params_a, params_b)
    pd.concat([grid_a.assign(model="A"), grid_b.assign(model="B")]).round(4).to_csv(
        reports / "lineup_model_cv.csv", index=False
    )

    # Unexplained lineup variance for Model B, from CV residuals (training seasons only).
    noise = noise_model(lineups[lineups["season"].isin(train_seasons)])
    resid_terms = []
    for f in folds:
        r = f.test["net_rating"].to_numpy() - predict_on(fit_on(f, memb, params_b, cfg), f, memb)
        n = f.test["possessions"].to_numpy(float)
        resid_terms.append(np.average(r**2 - noise["c"] / n, weights=n))
    tau2_resid = max(0.0, float(np.mean(resid_terms)))
    log.info("noise model %s; unexplained lineup variance %.2f", noise, tau2_resid)

    # --- The one test-season evaluation.
    test = make_split(lineups, test_season, train_seasons, min_poss)
    y = test.test["net_rating"].to_numpy(float)
    w = test.test["possessions"].to_numpy(float)
    model_a, model_b = fit_on(test, memb, params_a, cfg), fit_on(test, memb, params_b, cfg)
    league_mean = float(np.average(test.train["net_rating"], weights=test.train["possessions"]))
    preds = {
        "0": np.full(len(y), league_mean),
        "A": predict_on(model_a, test, memb),
        "B": predict_on(model_b, test, memb),
    }

    boot = paired_bootstrap(y, preds, w, cfg["bootstrap_test_reps"], seed)
    diff = boot["A"] - boot["B"]
    floor = float(np.sqrt(np.average(noise["c"] / w, weights=w)))
    table = pd.DataFrame(
        [
            {
                "model": k,
                "wrmse": wrmse(y, p, w),
                "wr2": wr2(y, p, w),
                "wrmse_ci_low": float(boot[k].quantile(0.025)),
                "wrmse_ci_high": float(boot[k].quantile(0.975)),
            }
            for k, p in preds.items()
        ]
    )
    var_y = float(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
    headline = {
        "rmse_a_minus_b": wrmse(y, preds["A"], w) - wrmse(y, preds["B"], w),
        "ci95_low": float(diff.quantile(0.025)),
        "ci95_high": float(diff.quantile(0.975)),
        "share_of_bootstrap_b_better": float((diff > 0).mean()),
        "noise_floor_wrmse": floor,
        "max_achievable_wr2_approx": 1 - floor**2 / var_y,
        "n_test_lineups": len(y),
        "test_possessions": float(w.sum()),
        "unseen_player_slots": int(
            (~np.isin(test.test[PLAYER_COLS].to_numpy(), model_a.player_effects.index)).sum()
        ),
    }
    log.info("results:\n%s\nheadline: %s", table.round(3).to_string(index=False), headline)

    # --- 80% predictive interval for Model B: bootstrap over training lineups for model
    # uncertainty, plus unexplained lineup variance and possession-level noise.
    rng = np.random.default_rng(seed)
    boot_preds = np.empty((cfg["bootstrap_fit_reps"], len(y)))
    for r in range(cfg["bootstrap_fit_reps"]):
        i = rng.integers(0, len(test.train), len(test.train))
        resampled = Split(test.target, test.train.iloc[i].reset_index(drop=True), test.test)
        boot_preds[r] = predict_on(fit_on(resampled, memb, params_b, cfg), resampled, memb)
    var_model = boot_preds.var(axis=0)
    z = stats.norm.ppf(0.5 + cfg["interval"] / 2)
    half = z * np.sqrt(var_model + tau2_resid + noise["c"] / w)
    covered = np.abs(y - preds["B"]) <= half
    coverage = {
        "nominal": cfg["interval"],
        "empirical_unweighted": float(covered.mean()),
        "empirical_possession_weighted": float(np.average(covered, weights=w)),
        "median_half_width": float(np.median(half)),
        "median_model_only_half_width": float(np.median(z * np.sqrt(var_model))),
        "tau2_resid": tau2_resid,
    }
    log.info("interval coverage: %s", coverage)

    # --- Write outputs.
    table.round(4).to_csv(reports / "lineup_model_comparison.csv", index=False)
    comparison_figure(table, boot, floor).write_html(
        reports / "lineup_model_comparison.html", include_plotlyjs="cdn"
    )
    models.mkdir(parents=True, exist_ok=True)
    joblib.dump({"A": model_a, "B": model_b}, models / "lineup_models_train.joblib")
    meta = {
        "training_seasons": train_seasons,
        "test_season": test_season,
        "test_min_possessions": min_poss,
        "params_a": params_a,
        "params_b": params_b,
        "headline": headline,
        "interval": coverage,
        "noise_model": noise,
        "metrics": table.to_dict(orient="records"),
        "seed": seed,
        "config_hash": config_hash(),
        "date": date.today().isoformat(),
    }
    (reports / "lineup_models_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
