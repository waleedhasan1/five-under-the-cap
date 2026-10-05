from fivecap.config import all_seasons, config_hash, load_settings


def test_settings_load_and_seasons_are_ordered() -> None:
    s = load_settings()
    seasons = all_seasons(s)
    assert len(seasons) == 5
    assert seasons == sorted(seasons)
    assert seasons[-1] == s["seasons"]["test"]
    assert s["seasons"]["test"] not in s["seasons"]["train"]


def test_config_hash_is_stable() -> None:
    assert config_hash() == config_hash()
    assert len(config_hash()) == 12
