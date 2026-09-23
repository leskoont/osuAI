import numpy as np
import torch

from osuai.actions import ActionSpace, playfield_region
from osuai.agent import EpsilonSchedule, InferenceEngine, explore
from osuai.model import QMapNet


def test_model_shapes_and_dueling(small_cfg):
    net = QMapNet(small_cfg.obs, small_cfg.model)
    frames = torch.randint(0, 255, (3, 2, 32, 32), dtype=torch.uint8)
    q = net(frames, torch.tensor([0.0, 1.0, 0.0]))
    space = ActionSpace.for_obs(32, 32)
    assert q.shape == (3, space.n) == (3, 2 * 8 * 8)
    assert torch.isfinite(q).all()


def test_engine_matches_model(small_cfg):
    eng = InferenceEngine(small_cfg, torch.device("cpu"))
    frames = np.random.default_rng(0).integers(0, 255, (2, 32, 32), dtype=np.uint8)
    a, q = eng.act(frames, 1.0, want_q=True)
    assert a == int(np.argmax(q))
    # загрузка весов на месте
    sd = {k: torch.zeros_like(v) for k, v in eng.model.state_dict().items()}
    eng.load_state_dict(sd)
    _, q0 = eng.act(frames, 0.0, want_q=True)
    assert np.allclose(q0, 0.0)


def test_action_space_roundtrip():
    sp = ActionSpace(24, 32)
    for a in (0, 5, sp.cells - 1, sp.cells, sp.n - 1):
        cy, cx, p = sp.decode(a)
        assert sp.encode(cy, cx, p) == a
        x, y, p2 = sp.to_norm(a)
        assert sp.from_norm(x, y, p2) == a
    assert sp.from_norm(-1, 2, 1) == sp.encode(sp.grid_h - 1, 0, 1)


def test_playfield_region_1440p():
    l, t, r, b = playfield_region((0, 0, 2560, 1440), 0, 0, 8)
    # scale = 3: playfield 1536x1152, по центру, сдвиг вниз на 24 px
    assert (r - l, b - t) == (1536, 1152)
    assert l == (2560 - 1536) // 2
    assert t == (1440 - 1152) // 2 + 24


def test_explore_and_schedule():
    sp = ActionSpace(8, 8)
    rng = np.random.default_rng(0)
    assert explore(7, 0.0, sp, rng, None, 0.5, 0.3) == (7, "greedy")
    kinds = {explore(7, 1.0, sp, rng, 3, 0.5, 0.3)[1] for _ in range(100)}
    assert kinds == {"guided", "random"}
    e = EpsilonSchedule(1.0, 0.1, 100)
    assert e(0) == 1.0 and abs(e(50) - 0.55) < 1e-9 and abs(e(1000) - 0.1) < 1e-9
