"""Smoke tests: every app page runs from app_data/ without errors, and the game scores a lineup."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
pytestmark = pytest.mark.skipif(
    not (APP.parent / "app_data" / "game.json").exists(), reason="app_data/ not built"
)


@pytest.mark.parametrize("page", ["Home.py", "pages/1_Player_Explorer.py", "pages/2_Methods.py"])
def test_page_runs(page: str) -> None:
    at = AppTest.from_file(str(APP / page), default_timeout=60).run()
    assert not at.exception


def test_game_scores_a_full_lineup() -> None:
    from fivecap.appdata import load

    cheapest = load().players.sort_values("salary").index[:5].tolist()  # always under budget
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    at.multiselect[0].set_value(cheapest).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"Predicted net rating", "Talent", "Fit"} <= set(labels)
    at.button[0].click().run()
    assert not at.exception
    assert "Percentile" in [m.label for m in at.metric]
