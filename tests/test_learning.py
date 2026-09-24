"""Сквозные проверки, что обучение действительно учит (на CPU, за секунды)."""
import numpy as np
import torch

from osuai.actions import ActionSpace
from osuai.learner import Learner
from osuai.replay import ReplayBuffer


def _bandit_buffer(cfg, n, rng, demo=False):
    """Контекстный бандит: яркий квадрат в случайной ячейке; награда 1 за нажатие именно там."""
    sp = ActionSpace.for_obs(cfg.obs.height, cfg.obs.width)
    buf = ReplayBuffer(n + 10, (cfg.obs.height, cfg.obs.width), cfg.obs.stack, cfg.train.n_step,
                       cfg.train.gamma, sp.cells, gap=0)
    targets = []
    for _ in range(n):
        cy, cx = rng.integers(sp.grid_h), rng.integers(sp.grid_w)
        frame = np.zeros((cfg.obs.height, cfg.obs.width), np.uint8)
        frame[cy * 4:cy * 4 + 4, cx * 4:cx * 4 + 4] = 255
        if demo:
            a = sp.encode(cy, cx, 1)
        else:
            a = int(rng.integers(sp.n)) if rng.random() < 0.5 else sp.encode(cy, cx, int(rng.integers(2)))
        r = 1.0 if a == sp.encode(cy, cx, 1) else 0.0
        buf.add(frame, a, r, first=True, terminal=True)  # каждый шаг — отдельный эпизод
        targets.append((frame, sp.encode(cy, cx, 1)))
    return buf, targets


def _accuracy(learner, targets):
    net = learner.online.eval()
    frames = torch.from_numpy(np.stack([np.stack([f] * learner.cfg.obs.stack) for f, _ in targets]))
    with torch.no_grad():
        q = net(frames, torch.zeros(len(targets)))
    net.train()
    pred = q.argmax(1).numpy()
    return float(np.mean(pred == np.array([a for _, a in targets])))


def _train(learner, steps):
    learner.make_prefetcher()
    done = 0
    while done < steps:
        done += learner.step_once()
    learner.prefetcher.stop()


def test_dqn_learns_contextual_bandit(small_cfg):
    torch.manual_seed(0)
    small_cfg.train.lr = 1e-3
    small_cfg.train.batch_size = 64
    rng = np.random.default_rng(0)
    buf, _ = _bandit_buffer(small_cfg, 4000, rng)
    _, test = _bandit_buffer(small_cfg, 200, np.random.default_rng(1))
    learner = Learner(small_cfg, torch.device("cpu"), buf)
    before = _accuracy(learner, test)
    _train(learner, 600)
    after = _accuracy(learner, test)
    assert after > 0.8 > before, (before, after)


def test_dqfd_pretrain_from_demos(small_cfg, tmp_path):
    torch.manual_seed(0)
    small_cfg.train.lr = 1e-3
    rng = np.random.default_rng(0)
    demos, _ = _bandit_buffer(small_cfg, 2000, rng, demo=True)
    _, test = _bandit_buffer(small_cfg, 200, np.random.default_rng(1))
    learner = Learner(small_cfg, torch.device("cpu"), None, demos)
    _train(learner, 300)
    assert _accuracy(learner, test) > 0.8
    path = learner.save()
    l2 = Learner(small_cfg, torch.device("cpu"), None, demos)
    assert l2.load(path) and l2.updates == learner.updates
    assert _accuracy(l2, test) == _accuracy(learner, test)


def test_update_budget_starts_after_checkpoint(small_cfg):
    buf = ReplayBuffer(100, (32, 32), 2, 3, 0.9, 64)
    l1 = Learner(small_cfg, torch.device("cpu"), buf)
    l1.updates, l1.env_steps = 5000, 0  # как после pretrain
    l1.save()
    l2 = Learner(small_cfg, torch.device("cpu"), buf)
    l2.tc.learning_starts, l2.tc.replay_ratio = 100, 0.5
    assert l2.load()
    l2.env_steps = 300
    assert l2.allowed_updates() - l2.updates == 100  # (300 - 100) * 0.5, а не -4900
