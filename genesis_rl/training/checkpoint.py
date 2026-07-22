"""checkpoint保存/復元。contract hash照合つき。"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ..contracts import contract_hash


def save_checkpoint(path: str | Path, agent, *, learner_step: int, env_transitions: int,
                    stage: int, cfg_snapshot: dict | None = None, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": contract_hash(),
        "agent": agent.state_dict(),
        "learner_step": learner_step,
        "env_transitions": env_transitions,
        "stage": stage,
        "cfg": cfg_snapshot or {},
        "time": time.time(),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, agent=None, strict_contract: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if strict_contract and payload.get("contract") != contract_hash():
        raise RuntimeError(
            f"contract hash mismatch: ckpt={payload.get('contract')} != current={contract_hash()}. "
            "観測/アクション契約が変わっています。--no-strict-contractで無視できます。")
    if agent is not None:
        agent.load_state_dict(payload["agent"])
    return payload


def find_latest(ckpt_dir: str | Path) -> Path | None:
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    cands = sorted(d.glob("latest.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None
