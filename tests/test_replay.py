import numpy as np
import pytest

from osuai.replay import ReplayBuffer

H, W, S, CELLS = 4, 4, 3, 8


def make(cap=100, n=3, gamma=0.5, gap=0):
    return ReplayBuffer(cap, (H, W), S, n, gamma, CELLS, gap=gap)


def fill(buf, episodes, reward=1.0, terminal_last=True):
    """episodes — список длин; кадр шага = его глобальный номер (mod 256)."""
    g = 0
    for length in episodes:
        for t in range(length):
            frame = np.full((H, W), g % 256, np.uint8)
            buf.add(frame, action=(t % 2) * CELLS + t % CELLS, reward=reward, first=(t == 0),
                    terminal=terminal_last and t == length - 1)
            g += 1


def test_stack_repeats_first_frame_at_episode_start():
    buf = make()
    fill(buf, [5, 5])
    st = buf.stack_at(np.array([0, 1, 4, 5, 6, 7]))
    ids = st[:, :, 0, 0]
    assert ids[0].tolist() == [0, 0, 0]
    assert ids[1].tolist() == [0, 0, 1]
    assert ids[2].tolist() == [2, 3, 4]
    assert ids[3].tolist() == [5, 5, 5]   # начало второго эпизода — не смешиваем с первым
    assert ids[4].tolist() == [5, 5, 6]
    assert ids[5].tolist() == [5, 6, 7]


def test_nstep_returns_and_done():
    buf = make(n=3, gamma=0.5)
    fill(buf, [5, 10])
    out = buf.gather(np.array([0, 2, 3, 4, 5]))
    # шаг 0: r0 + .5 r1 + .25 r2, эпизод продолжается
    assert out["ret"][0] == pytest.approx(1.75)
    assert out["done"][0] == 0
    # шаг 2: r2 + .5 r3 + .25 r4, шаг 4 терминальный
    assert out["ret"][1] == pytest.approx(1.75)
    assert out["done"][1] == 1
    # шаг 3: r3 + .5 r4, конец эпизода
    assert out["ret"][2] == pytest.approx(1.5)
    assert out["done"][2] == 1
    # шаг 4: последний шаг эпизода
    assert out["ret"][3] == pytest.approx(1.0)
    assert out["done"][3] == 1
    assert out["done"][4] == 0
    assert out["next_obs"][4, -1, 0, 0] == 8


def test_truncated_episode_counts_as_done():
    buf = make(n=3)
    fill(buf, [4, 6], terminal_last=False)
    out = buf.gather(np.array([2]))
    assert out["done"][0] == 1
    assert out["ret"][0] == pytest.approx(1.5)


def test_prev_press():
    buf = make()
    fill(buf, [5])
    p = buf.prev_press_at(np.array([0, 1, 2, 3]))
    # действия: t=0 без нажатия, t=1 с нажатием, t=2 без, ...
    assert p.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_sample_range_after_wraparound():
    buf = make(cap=50, n=3, gap=5)
    fill(buf, [37, 40])  # total=77 > cap
    lo, hi = buf.valid_range()
    assert lo == 77 - 50 + (S - 1) + 5
    assert hi == 77 - 3
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = buf.sample(16, rng)
        # в стеке никогда не должно оказаться перезаписанных кадров из «будущего»
        ids = out["obs"][:, :, 0, 0].astype(int)
        assert (np.diff(ids, axis=1) >= 0).all()


def test_gather_into_preallocated_views():
    buf = make()
    fill(buf, [20])
    spec = buf.batch_spec(8)
    out = {k: np.zeros(shape, dt) for k, (shape, dt) in spec.items()}
    views = {k: v[2:6] for k, v in out.items()}
    buf.gather(np.array([3, 4, 5, 6]), views)
    assert out["obs"][2, -1, 0, 0] == 3
    assert out["obs"][0].sum() == 0


def test_save_load_roundtrip(tmp_path):
    buf = make(cap=30)
    fill(buf, [20, 25])  # переполнение: сохраняются последние 30
    path = str(tmp_path / "r.npz")
    buf.save(path)
    b2 = make(cap=100)
    assert b2.load(path) == 30
    idx = np.array([5, 10, 20])
    a = buf.gather(idx + buf.total - 30)
    b = b2.gather(idx)
    for k in ("obs", "ret", "done", "action"):
        np.testing.assert_array_equal(a[k], b[k])
