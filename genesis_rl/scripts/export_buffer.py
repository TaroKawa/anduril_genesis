"""成功バッファをPhase 2(本番シムfine-tune)向けにエクスポートする。

学習チェックポイントとは別に、learnerプロセスの成功バッファは実行中のみGPUに存在する
ため、このスクリプトは学習済みポリシーでGenesis環境をデプロイ実行して成功エピソードの
特徴空間遷移を収集し、genesis_success.pt として保存する。

  uv run python -m genesis_rl.scripts.export_buffer \
      --ckpt checkpoints/best_gates.pt --episodes 200 --out checkpoints/genesis_success.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best_gates.pt")
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", type=str, default="checkpoints/genesis_success.pt")
    ap.add_argument("--min-gates", type=int, default=3)
    args = ap.parse_args()

    from ..config import load_config
    from ..training.checkpoint import load_checkpoint
    from ..training.collector import Collector
    from ..training.orchestrator import _load_curriculum_state, _pick_gpu_index

    cfg = load_config(args.config)
    device = torch.device("cuda", _pick_gpu_index(cfg.hw.collector_gpu, 0))
    stage, seed, _ = _load_curriculum_state(cfg)
    cfg.sac.burn_in_steps = 0
    cfg.sac.success_min_gates = args.min_gates

    collector = Collector(cfg, device, stage, seed)
    payload = load_checkpoint(args.ckpt)
    collector.actor.load_state_dict(payload["agent"]["actor"])
    collector.transitions = 10**9  # burn-in無効
    collector.warmup()

    episodes = 0
    chunks = []
    while episodes < args.episodes:
        _, succ, ep_infos = collector.step(deterministic=True)
        if succ is not None:
            chunks.append({k: v.cpu() for k, v in succ.items()})
        episodes += len(ep_infos)

    if not chunks:
        print("成功エピソードが収集できませんでした")
        return
    merged = {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"contract": payload["contract"], "transitions": merged}, out)
    n = merged["feat"].shape[0]
    print(f"saved {n} transitions ({episodes} episodes) → {out}")


if __name__ == "__main__":
    main()
