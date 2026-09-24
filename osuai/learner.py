"""Обучение: Double-DQN (dueling, n-step, Huber) + DQfD large-margin loss на демонстрациях.

Оптимизации под RTX 2060 / i5-10400:
* fp16 autocast + GradScaler (tensor cores Turing; bf16 на Turing не поддерживается);
* fused Adam, foreach-клиппинг и foreach-lerp для target-сети — минимум запусков ядер;
* отдельный low-priority CUDA-поток: обучение не задерживает инференс актора;
* батчи собираются фоновым потоком в pinned-память и едут на GPU как uint8
  (в 4 раза меньше трафика PCIe, нормализация /255 — уже на GPU);
* метрики копятся на GPU и читаются редко — нет синхронизации CPU↔GPU на каждом шаге.

Исходная версия «училась» на target = action * reward — это просто стягивало все выходы
к нулю (курсор в угол, клавиши не нажимаются), никакого RL там не было.
"""

from __future__ import annotations

import copy
import os
import queue
import threading
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from .actions import ActionSpace
from .config import Config
from .model import build_model
from .replay import ReplayBuffer

class Prefetcher:
    """Фоновая сборка батчей прямо в переиспользуемые pinned-буферы.

    Слот возвращается в пул только после того, как его H2D-копия на GPU завершилась (CUDA event)."""

    def __init__(self, fill_fn, spec: dict, device: torch.device, depth: int):
        self.fill_fn = fill_fn
        self.device = device
        self.cuda = device.type == "cuda"
        self.ready: queue.Queue = queue.Queue()
        self.free: queue.Queue = queue.Queue()
        for _ in range(max(1, depth) + 1):
            tensors = {k: torch.from_numpy(np.empty(shape, dtype)) for k, (shape, dtype) in spec.items()}
            if self.cuda:
                tensors = {k: v.pin_memory() for k, v in tensors.items()}
            arrays = {k: v.numpy() for k, v in tensors.items()}
            self.free.put((tensors, arrays, None))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="prefetch", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                tensors, arrays, event = self.free.get(timeout=0.1)
            except queue.Empty:
                continue
            if event is not None:
                event.synchronize()  # предыдущая H2D-копия из этого слота завершена
            if not self.fill_fn(arrays):
                self.free.put((tensors, arrays, None))
                time.sleep(0.05)
                continue
            self.ready.put((tensors, arrays))

    def get(self, timeout: float = 0.5):
        try:
            tensors, arrays = self.ready.get(timeout=timeout)
        except queue.Empty:
            return None
        if self.cuda:
            gpu = {k: v.to(self.device, non_blocking=True) for k, v in tensors.items()}
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        else:
            gpu = {k: v.clone() for k, v in tensors.items()}
            event = None
        self.free.put((tensors, arrays, event))
        return gpu

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


class Learner:
    def __init__(self, cfg: Config, device: torch.device, replay: ReplayBuffer | None,
                 demos: ReplayBuffer | None = None):
        self.cfg = cfg
        self.tc = cfg.train
        self.device = device
        self.cuda = device.type == "cuda"
        self.space = ActionSpace.for_obs(cfg.obs.height, cfg.obs.width)
        self.replay = replay
        self.demos = demos
        self.amp = self.tc.amp and self.cuda
        self.online = build_model(cfg.obs, cfg.model, device, channels_last=self.tc.channels_last)
        if self.amp:
            self.online.input_dtype = torch.float16
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.forward = self.online
        if self.tc.compile:
            self.forward = torch.compile(self.online, mode="max-autotune-no-cudagraphs")
        self.opt = torch.optim.Adam(self.online.parameters(), lr=self.tc.lr, eps=self.tc.adam_eps,
                                    fused=self.cuda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.params = list(self.online.parameters())
        self.target_params = list(self.target.parameters())
        self.gamma_n = self.tc.gamma ** self.tc.n_step
        self.stream = torch.cuda.Stream(device=device, priority=0) if self.cuda else None
        self.rng = np.random.default_rng(self.tc.seed + 1)

        self.updates = 0
        self.env_steps = 0  # обновляется актором
        self.weights_version = 0
        self._weights_lock = threading.Lock()
        # Двойной буфер весов для актора: learner пишет в неактивный без блокировки,
        # под lock только переключается индекс → актор никогда не ждёт шаг обучения.
        sdt = torch.float16 if self.cuda else None
        self._staging = [[p.detach().clone().to(sdt or p.dtype) for p in self.params] for _ in range(2)]
        self._active = 0
        self._metric_sum = torch.zeros(4, device=device)  # loss, td, margin, q
        self._metric_n = 0
        self._metric_lock = threading.Lock()
        self.save_requested = threading.Event()
        self.last_metrics: dict[str, float] = {}
        self.prefetcher: Prefetcher | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.updates_per_sec = 0.0
        self._skipped = 0.0  # апдейты, «прощённые» после долгой паузы обучения
        self._upd_base = 0  # счётчики на момент загрузки чекпоинта
        self._env_base = 0
        self.publish_weights()  # актор стартует с теми же весами, что и learner

    # ------------------------------------------------------------------ batches
    def batch_spec(self) -> dict:
        buf = self.replay if self.replay is not None else self.demos
        spec = buf.batch_spec(self.tc.batch_size)
        spec["demo"] = ((self.tc.batch_size,), np.float32)
        return spec

    def fill_batch(self, out: dict[str, np.ndarray]) -> bool:
        """Заполняет батч: онлайн-переходы + доля демонстраций (DQfD)."""
        bs = self.tc.batch_size
        n_demo = 0
        if self.demos is not None and self.demos.can_sample():
            n_demo = bs if self.replay is None else int(round(bs * self.tc.demo_ratio))
        n_online = bs - n_demo
        if n_online and (self.replay is None or not self.replay.can_sample()):
            return False
        if n_online:
            self.replay.sample(n_online, self.rng, {k: v[:n_online] for k, v in out.items()})
        if n_demo:
            self.demos.sample(n_demo, self.rng, {k: v[n_online:] for k, v in out.items()})
        out["demo"][:n_online] = 0.0
        out["demo"][n_online:] = 1.0
        return True

    def make_prefetcher(self) -> Prefetcher:
        self.prefetcher = Prefetcher(self.fill_batch, self.batch_spec(), self.device, self.tc.prefetch)
        return self.prefetcher

    # ------------------------------------------------------------------ update
    def update(self, batch: dict[str, torch.Tensor]) -> None:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            q = self.forward(batch["obs"], batch["press"]).float()
            with torch.no_grad():
                q_next_online = self.forward(batch["next_obs"], batch["next_press"]).float()
                q_next_target = self.target(batch["next_obs"], batch["next_press"]).float()
        act = batch["action"]
        q_sa = q.gather(1, act[:, None]).squeeze(1)
        with torch.no_grad():
            a_star = q_next_online.argmax(dim=1, keepdim=True)
            q_next = q_next_target.gather(1, a_star).squeeze(1)
            y = batch["ret"] + self.gamma_n * (1.0 - batch["done"]) * q_next
        td = F.huber_loss(q_sa, y, delta=self.tc.huber_delta, reduction="none")
        loss = td.mean()
        demo = batch["demo"]
        margin_loss = torch.zeros((), device=self.device)
        if self.demos is not None and self.tc.margin_weight > 0:
            margins = torch.full_like(q, self.tc.margin)
            margins.scatter_(1, act[:, None], 0.0)
            per = (q + margins).amax(dim=1) - q_sa
            margin_loss = (per * demo).sum() / demo.sum().clamp_min(1.0)
            loss = loss + self.tc.margin_weight * margin_loss

        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.params, self.tc.grad_clip, foreach=True)
        self.scaler.step(self.opt)
        self.scaler.update()
        with torch.no_grad():
            torch._foreach_lerp_(self.target_params, self.params, self.tc.target_tau)
            with self._metric_lock:
                self._metric_sum += torch.stack([loss.detach(), td.detach().mean(),
                                                 margin_loss.detach(), q_sa.detach().mean()])
                self._metric_n += 1
        self.updates += 1

    def flush_metrics(self) -> dict[str, float]:
        """Читает накопленные метрики (одна синхронизация CPU↔GPU на N апдейтов).
        Вызывается из потока learner; актор читает готовый `last_metrics` без ожидания GPU."""
        with self._metric_lock:
            if not self._metric_n:
                return self.last_metrics
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    vals = (self._metric_sum / self._metric_n).tolist()
                    self._metric_sum.zero_()
            else:
                vals = (self._metric_sum / self._metric_n).tolist()
                self._metric_sum.zero_()
            self._metric_n = 0
        self.last_metrics = dict(zip(("loss", "td", "margin", "q"), vals))
        return self.last_metrics

    # ------------------------------------------------------------------ weights sharing
    @torch.no_grad()
    def publish_weights(self) -> None:
        """Вызывается только из потока learner (или до его старта)."""
        nxt = 1 - self._active
        if self.cuda:
            # веса могли быть изменены на default stream (инициализация, load) — упорядочиваем
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                torch._foreach_copy_(self._staging[nxt], self.params)
            self.stream.synchronize()
        else:
            torch._foreach_copy_(self._staging[nxt], self.params)
        with self._weights_lock:
            self._active = nxt
            self.weights_version += 1

    def pull_weights(self, engine, known_version: int) -> int:
        if self.weights_version == known_version:
            return known_version
        with self._weights_lock:
            engine.load_params(self._staging[self._active])
            return self.weights_version

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict:
        return {
            "model": self.online.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "updates": self.updates,
            "env_steps": self.env_steps,
            "obs": asdict(self.cfg.obs),
            "model_cfg": asdict(self.cfg.model),
        }

    def save(self, path: str | None = None) -> str:
        """Вызывать из потока learner (или после stop()); из UI — через save_requested."""
        path = path or self.tc.checkpoint
        if self.stream is not None:
            self.stream.synchronize()  # веса пишутся в потоке learner — дождаться последнего шага
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)  # атомарно: при сбое во время записи старый файл цел
        return path

    def load(self, path: str | None = None) -> bool:
        path = path or self.tc.checkpoint
        if not os.path.exists(path):
            return False
        sd = load_checkpoint(path, self.cfg, self.device)
        self.online.load_state_dict(sd["model"])
        self.target.load_state_dict(sd.get("target", sd["model"]))
        if "opt" in sd:
            try:
                self.opt.load_state_dict(sd["opt"])
            except ValueError as e:
                print(f"[learner] состояние оптимизатора не загружено: {e}")
        if "scaler" in sd and self.amp:
            self.scaler.load_state_dict(sd["scaler"])
        self.updates = int(sd.get("updates", 0))
        self.env_steps = int(sd.get("env_steps", 0))
        self._upd_base, self._env_base = self.updates, self.env_steps
        self.publish_weights()
        return True

    # ------------------------------------------------------------------ thread
    def allowed_updates(self) -> float:
        """Бюджет апдейтов этой сессии: replay_ratio от шагов, сделанных после загрузки
        (иначе апдейты предобучения «съедали» бы бюджет, и онлайн-обучение долго стояло бы)."""
        steps = self.env_steps - self._env_base - self.tc.learning_starts
        return self._upd_base + steps * self.tc.replay_ratio - self._skipped

    def _forgive_backlog(self) -> None:
        """После долгой паузы обучения не «догоняем» тысячами апдейтов подряд."""
        backlog = self.allowed_updates() - self.updates
        if backlog > 2000:
            self._skipped += backlog - 2000

    def start(self) -> None:
        self.make_prefetcher()
        self.thread = threading.Thread(target=self._run, name="learner", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        t_rate, n_rate = time.perf_counter(), self.updates
        last_save = self.updates
        last_publish = time.perf_counter()
        while not self.stop_event.is_set():
            self._forgive_backlog()
            if self.pause_event.is_set() or self.updates >= self.allowed_updates():
                if self.save_requested.is_set():
                    self.save_requested.clear()
                    print(f"\n[learner] сохранено: {self.save()}")
                time.sleep(0.005)
                continue
            self.step_once()
            now = time.perf_counter()
            if now - last_publish > 0.25:
                self.publish_weights()
                last_publish = now
            if now - t_rate > 2.0:
                self.updates_per_sec = (self.updates - n_rate) / (now - t_rate)
                t_rate, n_rate = now, self.updates
                self.flush_metrics()  # синхронизация с GPU — здесь, а не в потоке актора
            if self.updates - last_save >= self.tc.save_every_updates or self.save_requested.is_set():
                path = self.save()
                if self.save_requested.is_set():
                    self.save_requested.clear()
                    print(f"\n[learner] сохранено: {path}")
                last_save = self.updates

    def step_once(self) -> bool:
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.default_stream(self.device))  # load()/init на default
            with torch.cuda.stream(self.stream):
                batch = self.prefetcher.get()
                if batch is None:
                    return False
                self.update(batch)
        else:
            batch = self.prefetcher.get()
            if batch is None:
                return False
            self.update(batch)
        return True

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.prefetcher is not None:
            self.prefetcher.stop()


def load_checkpoint(path: str, cfg: Config, device: torch.device) -> dict:
    sd = torch.load(path, map_location=device, weights_only=True)
    if sd.get("obs") and (sd["obs"] != asdict(cfg.obs) or sd.get("model_cfg") != asdict(cfg.model)):
        raise ValueError(
            f"Чекпоинт {path} создан с другими obs/model настройками: "
            f"obs={sd['obs']} model={sd.get('model_cfg')}")
    return sd
