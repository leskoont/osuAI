"""Главный цикл: актор (фиксированная частота) + фоновый learner + запись демонстраций.

Режимы:
    train    — играет и учится онлайн (ε-greedy + подсказки детектора, DQfD при наличии демо);
    play     — только играет лучшей политикой из чекпоинта;
    record   — записывает игру человека (или автопилота в симуляторе) как демонстрации;
    pretrain — офлайн-обучение на демонстрациях (без игры).
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import torch

from . import system
from .actions import ActionSpace
from .agent import EpsilonSchedule, InferenceEngine, explore
from .config import Config
from .learner import Learner, load_checkpoint
from .metrics import Metrics
from .model import resolve_device
from .replay import ReplayBuffer
from .reward import CircleDetector, CreditAssigner, HitTracker, PendingStep, RewardShaper, teacher_action


def _demo_files(pattern: str) -> list[str]:
    if not pattern:
        return []
    files: list[str] = []
    for part in pattern.split(","):
        files += sorted(glob.glob(part.strip()))
    return files


def load_demos(cfg: Config, space: ActionSpace) -> ReplayBuffer | None:
    files = _demo_files(cfg.train.demo_path)
    if not files:
        return None
    sizes = [len(np.load(f, mmap_mode="r")["actions"]) for f in files]
    demos = ReplayBuffer(sum(sizes) + 1, (cfg.obs.height, cfg.obs.width), cfg.obs.stack,
                         cfg.train.n_step, cfg.train.gamma, space.cells, gap=0)
    for f in files:
        n = demos.load(f)
        print(f"[demos] {f}: {n} шагов")
    return demos


class Runner:
    def __init__(self, cfg: Config, mode: str, sim: bool = False, out: str = "", max_steps: int = 0):
        assert mode in ("train", "play", "record")
        self.cfg = cfg
        self.mode = mode
        self.max_steps = max_steps  # 0 = без ограничения
        self.sim = sim
        self.out = out
        self.space = ActionSpace.for_obs(cfg.obs.height, cfg.obs.width)
        self.rng = np.random.default_rng(cfg.train.seed)
        system.tune(cfg.system)
        self.device = resolve_device(cfg.train.device)
        print(f"[osuai] режим {mode}{' (симулятор)' if sim else ''}, устройство: "
              f"{system.describe_device(self.device)}")

        if sim:
            from .sim import SimEnv
            self.env = SimEnv(cfg, seed=cfg.train.seed)
        else:
            from .env import OsuEnv
            self.env = OsuEnv(cfg)

        obs_shape = (cfg.obs.height, cfg.obs.width)
        self.buffer: ReplayBuffer | None = None
        self.learner: Learner | None = None
        self.engine: InferenceEngine | None = None
        if mode == "train":
            self.buffer = ReplayBuffer(cfg.train.replay_capacity, obs_shape, cfg.obs.stack,
                                       cfg.train.n_step, cfg.train.gamma, self.space.cells)
            if cfg.train.save_replay and os.path.exists(cfg.train.replay_path):
                print(f"[replay] загружено {self.buffer.load(cfg.train.replay_path)} шагов")
            demos = load_demos(cfg, self.space)
            self.learner = Learner(cfg, self.device, self.buffer, demos)
            if self.learner.load():
                print(f"[learner] чекпоинт {cfg.train.checkpoint}: {self.learner.updates} апдейтов, "
                      f"{self.learner.env_steps} шагов")
            self.engine = InferenceEngine(cfg, self.device)
        elif mode == "play":
            self.engine = InferenceEngine(cfg, self.device)
            if os.path.exists(cfg.train.checkpoint):
                self.engine.load_state_dict(load_checkpoint(cfg.train.checkpoint, cfg, self.device)["model"])
                print(f"[play] загружен {cfg.train.checkpoint}")
            else:
                print(f"[play] ВНИМАНИЕ: {cfg.train.checkpoint} не найден — играет необученная сеть")
        else:  # record
            self.buffer = ReplayBuffer(cfg.train.replay_capacity, obs_shape, cfg.obs.stack,
                                       cfg.train.n_step, cfg.train.gamma, self.space.cells, gap=0)

        sink = self.buffer.add if self.buffer is not None else (lambda *a: None)
        self.credit = CreditAssigner(cfg.reward, sink)
        self.tracker = HitTracker()
        self.detector = CircleDetector(cfg.detector) if cfg.detector.enabled else None
        self.shaper = RewardShaper(cfg.reward, self.env.aspect)
        self.eps = EpsilonSchedule(cfg.agent.eps_start, cfg.agent.eps_end, cfg.agent.eps_decay_steps)
        self.metrics = Metrics(cfg.log.dir, cfg.log.tensorboard)
        self.vis = None
        if cfg.vis.enabled:
            from .vis import Visualizer
            self.vis = Visualizer(cfg.vis.hz, cfg.vis.scale, cfg.vis.show_qmap)
        self.hotkeys = None
        if not sim:
            from .hotkeys import Hotkeys
            self.hotkeys = Hotkeys(cfg.hotkeys)

        self.env_steps = self.learner.env_steps if self.learner else 0
        self.session_steps = 0
        self.paused = not sim  # как и раньше: старт на паузе, 'p' — начать
        self.training = True
        self.running = True
        self._reset_episode_stats()
        self.episode_open = False
        self.stack = np.zeros((cfg.obs.stack, *obs_shape), np.uint8)
        self.prev_press = 0
        self.first = True
        self.last_fid = 0
        self.weights_version = 0
        self.last_sync = 0.0
        self.status_reason = ""

    # ------------------------------------------------------------------ episodes
    def _reset_episode_stats(self) -> None:
        self.ep = {"steps": 0, "reward": 0.0, "h300": 0, "h100": 0, "h50": 0, "miss": 0,
                   "max_combo": 0}

    def _start_episode(self, gray: np.ndarray) -> None:
        self.stack[:] = gray  # как и replay: кадры «до начала» = первый кадр эпизода
        self.prev_press = 0
        self.first = True
        self.tracker.reset()
        if self.detector is not None:
            self.detector.reset()
        counters = self.env.game_counters()
        if counters is not None:
            self.tracker.delta(counters)  # базовая линия счётчиков
        self._reset_episode_stats()
        self.episode_open = True

    def _end_episode(self) -> None:
        self.env.release()
        self.credit.flush(terminal=True)
        self.episode_open = False
        if self.ep["steps"] > 30:
            eps = self._current_eps()
            self.metrics.episode(self.env_steps, self.learner.updates if self.learner else 0, self.ep, eps)
            e = self.ep
            print(f"\n[эпизод] шагов {e['steps']}, награда {e['reward']:.2f}, 300/100/50/miss "
                  f"{e['h300']}/{e['h100']}/{e['h50']}/{e['miss']}, точность {e['accuracy']:.2f}%, "
                  f"макс. комбо {e['max_combo']}")

    def _current_eps(self) -> float:
        if self.mode == "record":
            return 0.0
        if self.mode == "play":
            return self.cfg.agent.eval_eps
        return self.eps(self.env_steps)

    # ------------------------------------------------------------------ controls
    def _handle_hotkeys(self) -> None:
        if self.hotkeys is None:
            return
        for name in self.hotkeys.poll():
            if name == "pause":
                self.paused = not self.paused
                print("\n[osuai] " + ("пауза" if self.paused else "работаем"))
                if not self.paused:
                    self.env.focus()
            elif name == "training" and self.learner is not None:
                self.training = not self.training
                (self.learner.pause_event.clear if self.training else self.learner.pause_event.set)()
                print(f"\n[osuai] обучение {'включено' if self.training else 'выключено'}")
            elif name == "save" and self.learner is not None:
                self.learner.save_requested.set()  # сохраняет сам поток learner — без гонок с апдейтом
            elif name == "focus":
                self.env.focus()
            elif name == "quit":
                self.running = False

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        if self.learner is not None:
            self.learner.start()
        if self.hotkeys is not None:
            h = self.cfg.hotkeys
            print(f"Управление: '{h.pause}' пауза/старт, '{h.training}' обучение вкл/выкл, "
                  f"'{h.save}' сохранить, '{h.focus}' фокус на osu!, '{h.quit}' выход")
            print("Агент на паузе. Откройте карту в osu! и нажмите пауза-клавишу.")
        period = 1.0 / self.cfg.agent.hz
        next_t = time.perf_counter()
        last_print = time.perf_counter()
        try:
            while self.running:
                self._handle_hotkeys()
                active, reason = (False, "пауза") if self.paused else self.env.status()
                self.status_reason = reason
                if not active:
                    if self.episode_open:
                        self._end_episode()
                    self._idle_vis()
                    last_print = self._maybe_print(last_print)
                    if self.env.realtime:
                        time.sleep(0.02)
                    next_t = time.perf_counter()
                    continue

                t_step = time.perf_counter()
                frame = self.env.observe(self.last_fid)
                if frame is None:
                    continue
                self.last_fid = frame.fid
                if not self.episode_open:
                    self._start_episode(frame.gray)
                else:
                    self.stack[:-1] = self.stack[1:]
                    self.stack[-1] = frame.gray
                dets = self.detector.detect(frame.color) if self.detector is not None else []

                want_q = self.vis is not None and self.vis.due()
                qmap = None
                t_inf = time.perf_counter()
                if self.mode == "record":
                    nx, ny, press, onset = self.env.human_step()
                    action = self.space.from_norm(nx, ny, press)
                    kind = "human"
                    infer = 0.0
                else:
                    greedy, qmap = self.engine.act(self.stack, self.prev_press, want_q)
                    infer = time.perf_counter() - t_inf
                    guide = teacher_action(dets, self.space, self.prev_press) if dets else None
                    action, kind = explore(greedy, self._current_eps(), self.space, self.rng, guide,
                                           self.cfg.agent.guided_prob, self.cfg.agent.explore_press_prob)
                    nx, ny, press = self.space.to_norm(action)
                    onset = self.env.act(nx, ny, press)

                scale = self.cfg.reward.shaping_scale * (1.0 if self.env.has_events
                                                         else self.cfg.reward.no_tosu_shaping_scale)
                reward = self.shaper(dets, nx, ny, onset, scale)
                self.credit.push(PendingStep(frame.gray, action, reward, self.first, onset))
                counters = self.env.game_counters()
                if counters is not None:
                    d = self.tracker.delta(counters)
                    reward += self.credit.apply_events(d)
                    self.ep["h300"] += d.h300
                    self.ep["h100"] += d.h100
                    self.ep["h50"] += d.h50
                    self.ep["miss"] += d.miss
                    self.ep["max_combo"] = max(self.ep["max_combo"], counters.combo)
                self.ep["steps"] += 1
                self.ep["reward"] += reward
                self.first = False
                self.prev_press = press
                self.env_steps += 1
                self.session_steps += 1
                if self.max_steps and self.session_steps >= self.max_steps:
                    self.running = False

                if self.learner is not None:
                    self.learner.env_steps = self.env_steps
                    self._sync_weights()
                    self._throttle()
                if want_q:
                    self._push_vis(frame.gray, qmap, nx, ny, press, dets)
                self.metrics.step(time.perf_counter() - t_step, infer, reward, kind)
                last_print = self._maybe_print(last_print)

                if self.env.realtime:
                    next_t += period
                    now = time.perf_counter()
                    if next_t < now - period:  # отстали больше чем на шаг — не «догоняем» пачкой
                        next_t = now
                    system.sleep_until(next_t)
        except KeyboardInterrupt:
            print("\n[osuai] прервано")
        finally:
            self.shutdown()

    def _sync_weights(self) -> None:
        now = time.perf_counter()
        if now - self.last_sync >= self.cfg.agent.weight_sync_seconds:
            self.last_sync = now
            self.weights_version = self.learner.pull_weights(self.engine, self.weights_version)

    def _throttle(self) -> None:
        """Для симулятора: не даём актору убегать от learner (держим replay_ratio)."""
        if not self.cfg.train.throttle_actor or not self.training:
            return
        while self.learner.allowed_updates() - self.learner.updates > 4 * self.cfg.train.replay_ratio + 8:
            if not self.learner.thread.is_alive():
                raise RuntimeError("Поток обучения завершился с ошибкой")
            time.sleep(0.001)

    # ------------------------------------------------------------------ ui
    def _status_lines(self) -> list[str]:
        s = self.metrics.summary()
        lines = [f"steps {self.env_steps}  eps {self._current_eps():.3f}  "
                 f"step {s['step_ms_p50']:.1f}/{s['step_ms_p99']:.1f}ms  infer {s['infer_ms_p50']:.2f}ms"]
        if self.learner is not None:
            m = self.learner.last_metrics
            lines.append(f"upd {self.learner.updates} ({self.learner.updates_per_sec:.0f}/s)  "
                         f"loss {m.get('loss', 0):.4f}  q {m.get('q', 0):.3f}  "
                         f"buf {len(self.buffer)}  train {'on' if self.training else 'off'}")
        e = self.ep
        lines.append(f"ep: 300 {e['h300']} 100 {e['h100']} 50 {e['h50']} miss {e['miss']} "
                     f"r {e['reward']:.1f}  tosu {'yes' if self.env.has_events else 'no'}")
        if self.status_reason:
            lines.append(f"inactive: {self.status_reason}")
        return lines

    def _maybe_print(self, last: float) -> float:
        now = time.perf_counter()
        if now - last < self.cfg.log.print_every_seconds:
            return last
        lines = self._status_lines()
        print("\r" + " | ".join(lines)[:230].ljust(230), end="", flush=True)
        if self.learner is not None and self.learner.last_metrics:
            self.metrics.scalars({f"train/{k}": v for k, v in self.learner.last_metrics.items()},
                                 self.env_steps)
        return now

    def _push_vis(self, gray, qmap, nx, ny, press, dets) -> None:
        from .vis import VisSnapshot
        q = qmap.reshape(2, self.space.grid_h, self.space.grid_w) if qmap is not None else None
        self.vis.push(VisSnapshot(gray, q, (nx, ny), press, dets, self._status_lines()))

    def _idle_vis(self) -> None:
        if self.vis is not None and self.vis.due():
            from .vis import VisSnapshot
            self.vis.push(VisSnapshot(self.stack[-1].copy(), None, None, 0, [], self._status_lines()))

    # ------------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        print("\n[osuai] завершение...")
        try:
            if self.episode_open:
                self._end_episode()
        finally:
            self.env.close()
        if self.learner is not None:
            self.learner.stop()
            print(f"[osuai] модель сохранена: {self.learner.save()}")
            if self.cfg.train.save_replay and self.buffer is not None and len(self.buffer):
                self.buffer.save(self.cfg.train.replay_path)
                print(f"[osuai] replay сохранён: {self.cfg.train.replay_path}")
        if self.mode == "record" and self.buffer is not None and len(self.buffer):
            out = self.out or "demos/demo.npz"
            self.buffer.save(out)
            print(f"[osuai] демонстрации ({len(self.buffer)} шагов) сохранены: {out}")
        if self.vis is not None:
            self.vis.stop()
        self.metrics.close()
        system.restore(self.cfg.system)


def pretrain(cfg: Config) -> None:
    """Офлайн-обучение на демонстрациях (DQfD: TD + large-margin)."""
    system.tune(cfg.system)
    device = resolve_device(cfg.train.device)
    space = ActionSpace.for_obs(cfg.obs.height, cfg.obs.width)
    print(f"[pretrain] устройство: {system.describe_device(device)}")
    demos = load_demos(cfg, space)
    if demos is None:
        raise SystemExit(f"Нет демонстраций по пути train.demo_path={cfg.train.demo_path!r}")
    learner = Learner(cfg, device, None, demos)
    if learner.load():
        print(f"[pretrain] продолжаем с {cfg.train.checkpoint} ({learner.updates} апдейтов)")
    learner.make_prefetcher()
    start = learner.updates
    target = start + cfg.train.pretrain_updates
    t0 = time.perf_counter()
    last = t0
    try:
        while learner.updates < target:
            stepped = learner.step_once()
            now = time.perf_counter()
            if now - last > 2.0:
                m = learner.flush_metrics()
                rate = (learner.updates - start) / max(now - t0, 1e-6)
                print(f"\r[pretrain] {learner.updates}/{target}  loss {m.get('loss', 0):.4f}  "
                      f"td {m.get('td', 0):.4f}  margin {m.get('margin', 0):.4f}  {rate:.0f} upd/s",
                      end="", flush=True)
                last = now
            if stepped and learner.updates % cfg.train.save_every_updates == 0:
                learner.save()
    except KeyboardInterrupt:
        print("\n[pretrain] прервано")
    finally:
        learner.prefetcher.stop()
        print(f"\n[pretrain] сохранено: {learner.save()}")
        system.restore(cfg.system)
