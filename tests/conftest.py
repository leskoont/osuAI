import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from osuai.config import load_config


@pytest.fixture
def small_cfg(tmp_path):
    """Маленькая конфигурация для быстрых тестов на CPU."""
    return load_config(None, [
        "obs.height=32", "obs.width=32", "obs.stack=2",
        "model.channels=16", "model.blocks=2", "model.stem_channels=8",
        "train.device='cpu'", "train.batch_size=16", "train.replay_capacity=5000",
        "train.learning_starts=0", "train.prefetch=1",
        f"train.checkpoint='{tmp_path / 'm.pt'}'", f"log.dir='{tmp_path / 'logs'}'",
        "vis.enabled=False", "tosu.enabled=False",
    ])
