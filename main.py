"""OsuAI — точка входа.

    python main.py train     [--config configs/rtx2060.toml] [--sim] [--set section.key=value ...]
    python main.py play      — играть обученной моделью
    python main.py record    --out demos/my_play.npz  — записать свою игру (демонстрации)
    python main.py pretrain  --set train.demo_path=demos/*.npz  — предобучение на демонстрациях
    python main.py benchmark — замер производительности конвейера на этом железе
"""

from __future__ import annotations

import argparse
import os
import signal
import sys


def _graceful_sigterm(*_):
    raise KeyboardInterrupt  # тот же корректный выход (отпустить клавиши, сохранить модель), что и Ctrl+C


DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "rtx2060.toml")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="OsuAI: RL-агент для osu!")
    p.add_argument("mode", choices=["train", "play", "record", "pretrain", "benchmark"])
    p.add_argument("--config", default=DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None,
                   help="TOML-конфиг (по умолчанию configs/rtx2060.toml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="переопределить параметр, напр. --set train.batch_size=128")
    p.add_argument("--sim", action="store_true", help="встроенный симулятор вместо игры")
    p.add_argument("--keys", nargs="+", help="клавиши osu!, напр. --keys z x")
    p.add_argument("--demos", help="демонстрации (glob), то же что --set train.demo_path=...")
    p.add_argument("--out", default="", help="файл для записи демонстраций (режим record)")
    p.add_argument("--no-vis", action="store_true", help="без окна визуализации")
    p.add_argument("--steps", type=int, default=0, help="остановиться после N шагов (0 = без ограничения)")
    args = p.parse_args(argv)
    signal.signal(signal.SIGTERM, _graceful_sigterm)

    from osuai.config import load_config

    overrides = list(args.overrides)
    if args.keys:
        overrides.append(f"input.keys={args.keys!r}")
    if args.demos:
        overrides.append(f"train.demo_path={args.demos!r}")
    if args.no_vis:
        overrides.append("vis.enabled=False")
    if args.sim:
        # симулятор быстрее реального времени; актор ждёт learner, чтобы держать replay_ratio
        overrides = ["train.throttle_actor=True", "tosu.enabled=False"] + overrides
    cfg = load_config(args.config, overrides)

    if args.mode == "benchmark":
        from osuai.bench import run
        run(cfg)
    elif args.mode == "pretrain":
        from osuai.runner import pretrain
        pretrain(cfg)
    else:
        from osuai.runner import Runner
        Runner(cfg, args.mode, sim=args.sim, out=args.out, max_steps=args.steps).run()


if __name__ == "__main__":
    sys.exit(main())
