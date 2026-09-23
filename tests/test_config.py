import os

import pytest

from osuai.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_preset_loads():
    cfg = load_config(os.path.join(ROOT, "configs", "rtx2060.toml"))
    assert cfg.train.amp and cfg.obs.height == 96


def test_overrides_and_validation():
    cfg = load_config(None, ["train.batch_size=128", "input.keys=['a', 's']", "vis.enabled=false"])
    assert cfg.train.batch_size == 128 and cfg.input.keys == ["a", "s"] and not cfg.vis.enabled
    with pytest.raises(KeyError):
        load_config(None, ["train.nope=1"])
    with pytest.raises(ValueError):
        load_config(None, ["obs.height=90"])
