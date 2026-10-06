"""Fit the game model on all five seasons and write the small files the app reads to ``app_data/``.

PROJECT_PLAN.md milestone 6. The app reads only ``app_data/``: no network, no fitting.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import date
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from fivecap.config import ROOT, all_seasons, config_hash, load_settings, path
from fivecap.evaluate import Split, fit_on
from fivecap.game import GameModel, pair_matrix
from fivecap.lineup_model import (
    PLAYER_COLS,
    LineupModel,
    Memberships,
    lineup_memberships,
)
from fivecap.optimizer import prefilter, top_lineups
from fivecap.valuation import ValuationInputs, minimum_salary, player_values, wins_regression

log = logging.getLogger("fivecap.export")


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    """First column of the first row of a query, as a float."""
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"query returned no rows: {sql}")
    return float(row[0])


def fit_const(model: LineupModel) -> float:
    """Fit of the mean training lineup, so a fit of 0 means "average fit"."""
    return float(model.pair_mean @ model.pair_coef)


def observed_compositions(lineups: pd.DataFrame, memb: Memberships, min_poss: int) -> np.ndarray:
    """Archetype compositions (sum of the five membership vectors) of observed lineups."""
    obs = lineups[lineups["possessions"] >= min_poss].reset_index(drop=True)
    return lineup_memberships(obs, memb, prior_only=False).sum(axis=1)


def ood_threshold(comps: np.ndarray, quantile: float) -> float:
    """``quantile`` of leave-one-out nearest-neighbour distances among observed compositions."""
    dists = np.empty(len(comps))
    for s in range(0, len(comps), 1024):
        c = comps[s : s + 1024]
        d2 = ((c[:, None, :] - comps[None, :, :]) ** 2).sum(axis=2)
        d2[np.arange(len(c)), np.arange(s, s + len(c))] = np.inf
        dists[s : s + len(c)] = np.sqrt(d2.min(axis=1))
    return float(np.quantile(dists, quantile))


def random_valid_lineups(
    game: GameModel, salary: np.ndarray, budget: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Predicted net ratings of ``n`` random in-budget, in-distribution lineups."""
    scores: list[np.ndarray] = []
    have = 0
    while have < n:
        idx = np.array([rng.choice(len(salary), 5, replace=False) for _ in range(4 * n)])
        idx = idx[salary[idx].sum(axis=1) <= budget]
        out = game.predict(idx)
        keep = out["total"][~out["ood"]]
        scores.append(keep)
        have += len(keep)
    return np.concatenate(scores)[:n]


def main() -> int:
    """Build the game model, valuation, optimal lineups and app files."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = load_settings()
    gcfg, vcfg, lcfg = s["game"], s["valuation"], s["lineup_model"]
    seed = s["random_seed"]
    seasons = all_seasons(s)
    latest, salary_season = s["seasons"]["test"], s["seasons"]["salary_season"]
    out_dir = path("app_data")
    out_dir.mkdir(exist_ok=True)
    meta5 = json.loads((path("reports") / "lineup_models_meta.json").read_text())
    params_b = meta5["params_b"]
    names = yaml.safe_load((ROOT / "config" / "archetype_names.yaml").read_text())

    with duckdb.connect(str(path("warehouse")), read_only=True) as con:
        lineups = con.execute("SELECT * FROM fct_lineup_season").fetchdf()
        mt = con.execute("SELECT * FROM fct_archetype_membership").fetchdf()
        teams = con.execute("SELECT * FROM fct_team_season").fetchdf()
        pool = con.execute(
            f"""
            SELECT p.player_id, p.player_name, p.team_abbreviation AS last_team, p.minutes,
                   s.salary, array_to_string(s.teams, '/') AS contract_team
            FROM fct_player_season AS p
            JOIN dim_salary AS s ON s.player_id = p.player_id AND s.season = '{salary_season}'
            WHERE p.season = '{latest}' AND p.minutes >= {s["thresholds"]["game_pool_min_minutes"]}
            ORDER BY p.player_id
            """
        ).fetchdf()
        sal_all = con.execute(
            f"SELECT salary FROM int_salary_dedup WHERE season = '{salary_season}'"
        ).fetchdf()["salary"]
        roster = scalar(
            con,
            f"SELECT median(n) FROM (SELECT team, count(DISTINCT bbref_id) AS n "
            f"FROM int_salary_long WHERE season = '{salary_season}' GROUP BY team)",
        )
        team_top5 = con.execute(
            f"""
            SELECT team, sum(salary) AS top5 FROM (
                SELECT team, salary,
                       row_number() OVER (PARTITION BY team ORDER BY salary DESC) AS rk
                FROM (SELECT DISTINCT bbref_id, team, salary FROM int_salary_long
                      WHERE season = '{salary_season}')
            ) WHERE rk <= 5 GROUP BY team
            """
        ).fetchdf()
        starter_share = scalar(
            con,
            f"""
            SELECT avg(share) FROM (
                SELECT p.min / t.min AS share,
                       row_number() OVER (PARTITION BY p.team_id ORDER BY p.min DESC) AS rk
                FROM stg_player_base AS p JOIN stg_team_base AS t USING (season, team_id)
                WHERE p.season = '{latest}'
            ) WHERE rk <= {vcfg["starters_per_team"]}
            """,
        )

    arch = [c for c in mt.columns if c.startswith("arch_")]
    k = len(arch)
    fit_rows = mt[mt["in_fit_sample"]]
    memb = Memberships(
        mt.set_index(["season", "player_id"])[arch],
        np.average(fit_rows[arch].to_numpy(), axis=0, weights=fit_rows["minutes"]),
    )

    # --- Game model: Model B refit on all five seasons, predicting the salary season.
    all_split = Split(
        salary_season,
        lineups[lineups["season"].isin(seasons)].reset_index(drop=True),
        lineups.iloc[:0],
    )
    model = fit_on(all_split, memb, params_b, lcfg)
    ids = pool["player_id"].to_numpy()
    missing = ~np.isin(ids, model.player_effects.index)
    if missing.any():
        raise RuntimeError(f"{missing.sum()} pool players have no fitted effect")
    pool_m = memb.lookup(ids, latest, prior_only=False)

    rng = np.random.default_rng(seed)
    reps = gcfg["bootstrap_reps"]
    boot_e, boot_i, boot_g, boot_c = (
        np.empty((reps, len(ids))),
        np.empty(reps),
        np.empty((reps, k, k)),
        np.empty(reps),
    )
    for r in range(reps):
        i = rng.integers(0, len(all_split.train), len(all_split.train))
        b = fit_on(
            Split(salary_season, all_split.train.iloc[i].reset_index(drop=True), all_split.test),
            memb,
            params_b,
            lcfg,
        )
        boot_e[r], boot_i[r] = b.effects(ids), b.intercept
        boot_g[r], boot_c[r] = pair_matrix(b.pair_coef, k), fit_const(b)
    log.info("bootstrap refits done (%d)", reps)

    comps = observed_compositions(lineups, memb, gcfg["ood_min_possessions"])
    threshold = ood_threshold(comps, gcfg["ood_quantile"])
    game = GameModel(
        effects=model.effects(ids),
        memberships=pool_m,
        intercept=model.intercept,
        g=pair_matrix(model.pair_coef, k),
        fit_const=fit_const(model),
        boot_effects=boot_e,
        boot_intercept=boot_i,
        boot_g=boot_g,
        boot_fit_const=boot_c,
        tau2_resid=meta5["interval"]["tau2_resid"],
        z=float(stats.norm.ppf(0.5 + lcfg["interval"] / 2)),
        observed=comps,
        ood_threshold=threshold,
        ood_multiplier=gcfg["ood_interval_multiplier"],
    )
    log.info("OOD threshold %.3f from %d observed lineups", threshold, len(comps))

    # --- Valuation.
    train_poss = (
        pd.Series(
            np.repeat(all_split.train["possessions"].to_numpy(float), 5),
            index=all_split.train[PLAYER_COLS].to_numpy().ravel(),
        )
        .groupby(level=0)
        .sum()
    )
    slope, intercept = wins_regression(teams[teams["season"].isin(s["seasons"]["train"])])
    v = ValuationInputs(
        wins_per_point=slope,
        wins_at_zero=intercept,
        replacement_effect=model.unseen_effect,
        average_effect=float(
            np.average(model.player_effects.reindex(train_poss.index), weights=train_poss)
        ),
        starter_share=starter_share,
        min_salary=minimum_salary(sal_all, vcfg["min_salary_quantile"]),
        roster_size=roster,
        league_payroll=float(sal_all.sum()),
        league_wins=float(
            teams[teams["season"].isin(s["seasons"]["train"])]
            .groupby("season")["wins"]
            .sum()
            .mean()
        ),
        n_teams=int(teams["team_id"].nunique()),
    )
    log.info("valuation inputs: %s", {kk: round(vv, 3) for kk, vv in v.as_dict().items()})
    values = player_values(game.effects, pool["salary"].to_numpy(float), v)

    # --- Budgets from the salary data.
    budgets = {d: float(team_top5["top5"].quantile(q)) for d, q in gcfg["budget_quantiles"].items()}
    log.info("budgets: %s", {d: round(b / 1e6, 2) for d, b in budgets.items()})

    # --- Optimizer per difficulty.
    salary = pool["salary"].to_numpy(float)
    top_arch = pool_m.argmax(axis=1)
    keep = prefilter(salary, game.effects, top_arch, gcfg["dominance_count"])
    kept = np.flatnonzero(keep)
    log.info("optimizer pool: %d of %d players after removing dominated ones", len(kept), len(ids))
    pv = pool_m[kept] @ game.g @ pool_m[kept].T
    eff = game.effects[kept] + (game.intercept - game.fit_const) / 5  # constant spread over 5 slots

    def accept(c: tuple[int, ...]) -> bool:
        return not bool(game.predict(kept[list(c)][None, :])["ood"][0])

    opt_rows, random_rows = [], []
    for d, budget in budgets.items():
        sols, rejected = top_lineups(
            eff,
            pv,
            salary[kept],
            budget,
            n_top=gcfg["top_n_lineups"],
            accept=accept,
            max_rounds=gcfg["max_solver_rounds"],
            seconds=gcfg["solver_seconds_per_round"],
            scale=gcfg["objective_scale"],
            seed=seed,
        )
        log.info(
            "%s: %d lineups found, %d out-of-distribution solutions rejected",
            d,
            len(sols),
            rejected,
        )
        for rank, sol in enumerate(sols, 1):
            idx = kept[list(sol.players)]
            pred = game.predict(idx[None, :])
            if not np.isclose(pred["total"][0], sol.score, atol=1e-6):
                raise RuntimeError("optimizer score disagrees with the game model")
            opt_rows.append(
                {
                    "difficulty": d,
                    "rank": rank,
                    "proven_optimal": sol.optimal,
                    "payroll": float(salary[idx].sum()),
                    **{f"p{j + 1}": int(ids[x]) for j, x in enumerate(idx)},
                    **{kk: float(vv[0]) for kk, vv in pred.items()},
                }
            )
        scores = random_valid_lineups(game, salary, budget, gcfg["random_lineups"], rng)
        random_rows.append(pd.DataFrame({"difficulty": d, "total": scores}))

    # --- Write app_data/.
    players = pool.assign(
        effect=game.effects,
        wins_added=values["wins_added"],
        value=values["value"],
        surplus=values["surplus"],
        top_archetype=[names["names"][a] for a in top_arch],
        **{a: pool_m[:, j] for j, a in enumerate(arch)},
    )
    players.to_parquet(out_dir / "players.parquet", index=False)
    pd.DataFrame(boot_e, columns=[str(x) for x in ids]).to_parquet(
        out_dir / "boot_effects.parquet", index=False
    )
    pd.DataFrame(
        {"intercept": boot_i, "fit_const": boot_c}
        | {f"g_{a}_{b}": boot_g[:, a, b] for a in range(k) for b in range(k)}
    ).to_parquet(out_dir / "boot_model.parquet", index=False)
    pd.DataFrame(comps, columns=arch).to_parquet(
        out_dir / "observed_compositions.parquet", index=False
    )
    pd.DataFrame(opt_rows).to_parquet(out_dir / "optimal_lineups.parquet", index=False)
    pd.concat(random_rows).to_parquet(out_dir / "random_lineup_scores.parquet", index=False)
    for f in [
        "lineup_model_comparison.csv",
        "archetype_summary.csv",
        "archetype_top_members.csv",
        "archetype_bic.csv",
        "archetype_stability.csv",
        "lineup_models_meta.json",
        "archetypes_meta.json",
    ]:
        shutil.copy(path("reports") / f, out_dir / f)
    game_json: dict[str, Any] = {
        "salary_season": salary_season,
        "ratings_through": latest,
        "archetype_names": {f"arch_{a}": n for a, n in names["names"].items()},
        "intercept": game.intercept,
        "fit_const": game.fit_const,
        "g": game.g.tolist(),
        "tau2_resid": game.tau2_resid,
        "z": game.z,
        "interval": lcfg["interval"],
        "ood_threshold": threshold,
        "ood_multiplier": game.ood_multiplier,
        "budgets": budgets,
        "salary_cap": s["salary_cap"],
        "valuation": v.as_dict(),
        "pool_size": len(ids),
        "optimizer_pool_size": int(len(kept)),
        "params_b": params_b,
        "config_hash": config_hash(),
        "date": date.today().isoformat(),
    }
    (out_dir / "game.json").write_text(json.dumps(game_json, indent=2))
    log.info("wrote app_data/ (%d players)", len(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
