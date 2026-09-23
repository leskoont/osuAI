import os

import numpy as np

from osuai.config import load_config
from osuai.runner import Runner, pretrain
from osuai.sim import SimEnv


def test_sim_judgement_rules():
    cfg = load_config(None, ["obs.height=48", "obs.width=64"])
    env = SimEnv(cfg, seed=0)
    n = env.notes[0]
    env.t = n.t - 500  # слишком рано: клик игнорируется (note lock)
    env.act(n.x, n.y, 1)
    assert env.counters.h300 + env.counters.h100 + env.counters.h50 == 0
    env.act(n.x, n.y, 0)
    env.t = n.t - 5
    env.act(0.0, 0.0, 1)  # мимо круга
    assert not n.judged
    env.act(n.x, n.y, 0)
    env.t = n.t
    env.act(n.x, n.y, 1)
    assert n.hit and env.counters.h300 == 1 and env.counters.combo == 1
    # следующая нота пропущена → промах и сброс комбо
    m = env.notes[1]
    env.t = m.t + cfg.sim.hit_window_ms + 1
    env.act(0.0, 0.0, 0)
    assert m.judged and not m.hit and env.counters.miss >= 1 and env.counters.combo == 0


def test_sim_autopilot_is_perfect():
    cfg = load_config(None, ["obs.height=48", "obs.width=64"])
    env = SimEnv(cfg, seed=3)
    while not env._ended:
        env.observe()
        env.human_step()
    c = env.counters
    assert c.miss == 0 and c.h300 == len(env.notes)


def _cfg(small_cfg, tmp_path):
    small_cfg.train.throttle_actor = True
    small_cfg.train.learning_starts = 200
    small_cfg.train.replay_ratio = 0.1
    small_cfg.sim.map_seconds = 4.0
    small_cfg.train.demo_path = str(tmp_path / "demo.npz")
    return small_cfg


def test_runner_record_pretrain_train_play(small_cfg, tmp_path):
    cfg = _cfg(small_cfg, tmp_path)
    demo = str(tmp_path / "demo.npz")
    Runner(cfg, "record", sim=True, out=demo, max_steps=600).run()
    data = np.load(demo)
    assert len(data["actions"]) == 600
    assert data["rewards"].max() >= cfg.reward.hit300 * 0.99  # автопилот попадает

    cfg.train.pretrain_updates = 20
    pretrain(cfg)
    assert os.path.exists(cfg.train.checkpoint)

    Runner(cfg, "train", sim=True, max_steps=500).run()
    assert os.path.exists(os.path.join(cfg.log.dir, "episodes.csv"))
    Runner(cfg, "play", sim=True, max_steps=100).run()
