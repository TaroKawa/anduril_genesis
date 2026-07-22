"""collector単体のスループットベンチ(learnerなし・キューなし)。

同一stage/seedでの最適化A/B比較用:
  uv run python -m genesis_rl.scripts.bench_collector --stage 2 --seed 26 --steps 300
  GENESIS_RL_KEEP_DEFAULT_DEVICE=1 uv run python -m genesis_rl.scripts.bench_collector ...
"""

from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    import os

    from ..config import load_config
    from ..training.orchestrator import _pick_gpu_index

    cfg = load_config(args.config, args.set)
    cfg.run.profile = True
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_pick_gpu_index(cfg.hw.collector_gpu, 0)))

    import torch

    torch.set_float32_matmul_precision("high")

    from ..training.collector import Collector

    device = torch.device("cuda", 0)
    collector = Collector(cfg, device, args.stage, args.seed)
    collector.transitions = 10_000_000  # burn_in回避(actor経路を測る)
    collector.warmup()
    print(f"[bench] stage={args.stage} seed={args.seed} envs={collector.N} "
          f"backend={collector.env.rig.backend} steps={args.steps}")

    for _ in range(args.warmup):
        collector.step()
    collector.prof.acc.clear()
    collector.prof.steps = 0
    collector.prof.transitions = 0
    collector.prof._t0 = time.perf_counter()

    t0 = time.perf_counter()
    for i in range(args.steps):
        collector.step()
    dt = time.perf_counter() - t0
    n = args.steps * collector.N
    print(f"[bench] {args.steps} steps ({n} transitions) in {dt:.1f}s "
          f"→ {n / dt:.0f} tps, {dt / args.steps * 1000:.1f} ms/step")


if __name__ == "__main__":
    main()
