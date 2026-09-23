"""Конфигурация проекта: dataclass-секции, загрузка из TOML и переопределения из CLI.

Пример:
    cfg = load_config("configs/rtx2060.toml", overrides=["train.batch_size=128"])
"""

from __future__ import annotations

import ast
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class CaptureConfig:
    backend: str = "auto"            # auto | dxcam | mss
    region: str = "playfield"        # playfield | window | monitor | manual
    manual_region: list[int] = field(default_factory=list)  # [left, top, right, bottom] для region="manual"
    window_title: str = "osu!"
    process_name: str = "osu!.exe"
    monitor: int = 1                 # для region="monitor" (нумерация mss: 1 = первый монитор)
    target_fps: int = 144            # частота захвата (с запасом над agent.hz, чтобы кадр всегда был свежим)
    # поля вокруг playfield 512x384 (osu!px): 64/48 → ровно 640x480 (4:3) и круги у края
    # поля (радиус до ~50 osu!px) не обрезаются
    margin_x: float = 64.0
    margin_y: float = 48.0
    playfield_y_offset: float = 8.0  # osu!stable сдвигает playfield вниз на ~8 osu!px (в масштабе 480p)
    dxcam_output: int = -1           # -1 = определить автоматически по положению окна


@dataclass
class ObsConfig:
    height: int = 96                 # кратно 4 (сеть уменьшает в 4 раза)
    width: int = 128
    stack: int = 4                   # количество последовательных кадров во входе сети
    detector_scale: int = 2          # детектор работает на цветном кадре (H*scale, W*scale)


@dataclass
class ModelConfig:
    channels: int = 64
    blocks: int = 4                  # residual-блоки с дилатациями 1,2,4,8,...
    stem_channels: int = 32


@dataclass
class AgentConfig:
    hz: float = 60.0                 # частота принятия решений
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_decay_steps: int = 150_000
    eval_eps: float = 0.0            # epsilon в режиме play
    explore_press_prob: float = 0.3  # вероятность нажатия при случайном действии
    guided_prob: float = 0.5         # доля exploration-шагов, где действие подсказывает детектор
    weight_sync_seconds: float = 1.0 # как часто актор забирает свежие веса у learner


@dataclass
class TrainConfig:
    device: str = "auto"             # auto | cuda | cpu
    amp: bool = True                 # fp16 autocast + GradScaler (tensor cores Turing)
    channels_last: bool = True
    cuda_graphs: bool = True         # CUDA Graph для инференса актора
    compile: bool = False            # torch.compile для learner (на Windows требует triton-windows)
    batch_size: int = 64
    lr: float = 2.5e-4
    adam_eps: float = 1.5e-4
    gamma: float = 0.95
    n_step: int = 3
    huber_delta: float = 1.0
    grad_clip: float = 10.0
    target_tau: float = 0.005        # polyak-усреднение target-сети на каждом апдейте
    replay_capacity: int = 300_000   # кадров (96x128 uint8 = 12 KiB/кадр → ~3.7 GB)
    learning_starts: int = 5_000
    replay_ratio: float = 0.5        # апдейтов на один шаг среды
    throttle_actor: bool = False     # актор ждёт learner, если тот отстал (для симулятора)
    prefetch: int = 3                # батчей в очереди предзагрузки
    demo_path: str = ""              # записанные демонстрации (npz) для DQfD
    demo_ratio: float = 0.25         # доля демо-переходов в батче при онлайн-обучении
    margin: float = 0.8              # large-margin loss DQfD
    margin_weight: float = 1.0
    pretrain_updates: int = 20_000
    checkpoint: str = "models/osuai.pt"
    save_every_updates: int = 5_000
    save_replay: bool = False        # сохранять replay-буфер при выходе (большой файл)
    replay_path: str = "models/replay.npz"
    seed: int = 0


@dataclass
class RewardConfig:
    hit300: float = 1.0
    hit100: float = 0.4
    hit50: float = 0.15
    miss: float = -1.0
    slider_break: float = -0.5
    combo_tick: float = 0.05         # тики/концы слайдеров (рост комбо без новых попаданий)
    credit_window_steps: int = 12    # задержка коммита в replay для привязки награды к нажатию
    miss_lag_steps: int = 3
    shaping_scale: float = 1.0       # множитель эвристических наград (детектор); 0 = выключить
    proximity: float = 0.03
    click_good: float = 0.3
    click_spam: float = 0.1
    no_tosu_shaping_scale: float = 3.0  # если tosu недоступен — усиливаем эвристику


@dataclass
class DetectorConfig:
    enabled: bool = True
    brightness_threshold: int = 70
    min_radius: float = 0.015        # в долях ширины области захвата
    max_radius: float = 0.35
    disk_fill: float = 0.5           # заполненность bbox для «диска»


@dataclass
class InputConfig:
    keys: list[str] = field(default_factory=lambda: ["z", "x"])
    alternate: bool = True           # чередовать клавиши на каждом новом нажатии


@dataclass
class TosuConfig:
    enabled: bool = True
    url: str = "ws://127.0.0.1:24050/ws"
    gate_on_playing: bool = True     # действовать только во время игры (menu.state == 2)


@dataclass
class HotkeyConfig:
    pause: str = "p"
    training: str = "t"
    save: str = "s"
    focus: str = "f"
    quit: str = "q"


@dataclass
class SystemConfig:
    torch_threads: int = 4           # i5-10400: 6 ядер / 12 потоков, оставляем ядра для osu!
    cv2_threads: int = 2
    switch_interval: float = 0.001   # интервал переключения GIL (по умолчанию 5 мс — много для 60 Гц)
    process_priority: str = "above_normal"  # normal | above_normal | high
    timer_resolution_ms: int = 1


@dataclass
class VisConfig:
    enabled: bool = True
    hz: float = 15.0
    scale: int = 4
    show_qmap: bool = True


@dataclass
class LogConfig:
    dir: str = "logs"
    print_every_seconds: float = 2.0
    tensorboard: bool = False


@dataclass
class SimConfig:
    bpm_min: float = 100.0
    bpm_max: float = 170.0
    approach_ms: float = 600.0       # AR ~8
    circle_radius: float = 0.063     # в долях ширины (CS ~4)
    map_seconds: float = 40.0
    hit_window_ms: float = 150.0     # окно для 50 (300/100 — 1/3 и 2/3 окна)
    realtime: bool = False


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    input: InputConfig = field(default_factory=InputConfig)
    tosu: TosuConfig = field(default_factory=TosuConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    vis: VisConfig = field(default_factory=VisConfig)
    log: LogConfig = field(default_factory=LogConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def validate(self) -> None:
        if self.obs.height % 4 or self.obs.width % 4:
            raise ValueError("obs.height и obs.width должны быть кратны 4")
        if self.obs.stack < 1:
            raise ValueError("obs.stack >= 1")
        if self.train.n_step < 1:
            raise ValueError("train.n_step >= 1")
        if len(self.input.keys) < 1:
            raise ValueError("input.keys: нужна хотя бы одна клавиша")
        if self.capture.region == "manual" and len(self.capture.manual_region) != 4:
            raise ValueError("capture.manual_region: нужно [left, top, right, bottom]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(obj: Any, data: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"Неизвестный параметр конфига: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{path}{key} должен быть таблицей")
            _update_dataclass(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, _coerce(current, value, f"{path}{key}"))


def _coerce(current: Any, value: Any, name: str) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(current, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list) and not isinstance(value, list):
        raise TypeError(f"{name} должен быть списком")
    return value


def _parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"Переопределение должно иметь вид section.key=value: {item}")
    key, raw = item.split("=", 1)
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return key.strip().split("."), value


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "rb") as f:
            _update_dataclass(cfg, tomllib.load(f))
    for item in overrides or []:
        keys, value = _parse_override(item)
        nested: dict[str, Any] = {keys[-1]: value}
        for k in reversed(keys[:-1]):
            nested = {k: nested}
        _update_dataclass(cfg, nested)
    cfg.validate()
    return cfg
