"""Phase 1 学習エントリポイント。

  uv run python -m genesis_rl.scripts.train --config configs/train.yaml --resume auto
  uv run python -m genesis_rl.scripts.train --smoke   # 数分の疎通テスト

mode=async: collector(3070Ti) + learner(4060) の2プロセス2GPU。
mode=sync : 1プロセス1GPU(デバッグ)。
カリキュラムのシーン再構築時はexit code 3で終了する(docker restart / 外側ループが再起動)。
"""

from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--resume", type=str, default=None, help="'auto' か ckptパス")
    ap.add_argument("--mode", choices=["async", "sync"], default=None)
    ap.add_argument("--smoke", action="store_true", help="小規模の疎通テスト(数分)")
    ap.add_argument("--set", action="append", default=[], help="設定上書き 例: --set sac.batch_size=256")
    args = ap.parse_args()

    from ..config import load_config

    cfg = load_config(args.config, args.set)
    if args.mode:
        cfg.run.mode = args.mode

    if args.smoke:
        cfg.env.num_envs = 8
        cfg.env.render.backend = "sequential"
        cfg.sac.batch_size = 256
        cfg.sac.learn_start = 512
        cfg.sac.burn_in_steps = 512
        cfg.sac.compile = False
        # syncスモークは全部が1GPUに載る。リプレイ1M(≈2.5GB)はVRAMを溢れさせるので縮小
        cfg.sac.replay_capacity = 50_000
        cfg.sac.success_capacity = 10_000
        cfg.run.mode = "sync"
        cfg.run.ckpt_interval_s = 60.0
        cfg.curriculum.enabled = False

    from ..training.orchestrator import run_async, run_sync

    if cfg.run.mode == "sync":
        code = run_sync(cfg, resume=args.resume, smoke=args.smoke)
    else:
        code = run_async(cfg, resume=args.resume)
    sys.exit(code)


if __name__ == "__main__":
    main()
