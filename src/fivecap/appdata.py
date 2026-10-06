"""Load ``app_data/`` into the objects the Streamlit app uses. No network, no fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fivecap.config import ROOT
from fivecap.game import GameModel

APP_DATA = ROOT / "app_data"


@dataclass
class AppData:
    """Everything the three app pages read."""

    game: GameModel
    players: pd.DataFrame  # one row per pool player, aligned with ``game`` arrays
    meta: dict[str, Any]  # app_data/game.json
    optimal: pd.DataFrame
    random_scores: pd.DataFrame
    comparison: pd.DataFrame
    validation_meta: dict[str, Any]
    archetype_summary: pd.DataFrame
    archetype_top: pd.DataFrame
    archetype_bic: pd.DataFrame
    archetype_meta: dict[str, Any]

    @property
    def archetype_cols(self) -> list[str]:
        return list(self.meta["archetype_names"])

    def archetype_name(self, col: str) -> str:
        return str(self.meta["archetype_names"][col])


def load(root: Path = APP_DATA) -> AppData:
    """Read every app file and assemble the game model."""
    meta = json.loads((root / "game.json").read_text())
    players = pd.read_parquet(root / "players.parquet")
    arch = list(meta["archetype_names"])
    k = len(arch)
    boot = pd.read_parquet(root / "boot_model.parquet")
    boot_e = pd.read_parquet(root / "boot_effects.parquet")
    boot_e = boot_e[[str(x) for x in players["player_id"]]]
    game = GameModel(
        effects=players["effect"].to_numpy(float),
        memberships=players[arch].to_numpy(float),
        intercept=float(meta["intercept"]),
        g=np.array(meta["g"]),
        fit_const=float(meta["fit_const"]),
        boot_effects=boot_e.to_numpy(float),
        boot_intercept=boot["intercept"].to_numpy(float),
        boot_g=np.stack(
            [boot[[f"g_{a}_{b}" for b in range(k)]].to_numpy(float) for a in range(k)], axis=1
        ),
        boot_fit_const=boot["fit_const"].to_numpy(float),
        tau2_resid=float(meta["tau2_resid"]),
        z=float(meta["z"]),
        observed=pd.read_parquet(root / "observed_compositions.parquet")[arch].to_numpy(float),
        ood_threshold=float(meta["ood_threshold"]),
        ood_multiplier=float(meta["ood_multiplier"]),
    )
    return AppData(
        game=game,
        players=players,
        meta=meta,
        optimal=pd.read_parquet(root / "optimal_lineups.parquet"),
        random_scores=pd.read_parquet(root / "random_lineup_scores.parquet"),
        comparison=pd.read_csv(root / "lineup_model_comparison.csv"),
        validation_meta=json.loads((root / "lineup_models_meta.json").read_text()),
        archetype_summary=pd.read_csv(root / "archetype_summary.csv"),
        archetype_top=pd.read_csv(root / "archetype_top_members.csv"),
        archetype_bic=pd.read_csv(root / "archetype_bic.csv"),
        archetype_meta=json.loads((root / "archetypes_meta.json").read_text()),
    )
