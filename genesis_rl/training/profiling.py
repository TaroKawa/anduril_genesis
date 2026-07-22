"""collector用の軽量セクションプロファイラ。

run.profile=true のときだけ有効。セクション前後で torch.cuda.synchronize() を
入れるため計測自体が実行を数%遅くする(計測モード専用、常用しない)。

セクションの入れ子に注意: env_step は render/physics を内包するので、
レポートの合計は step 時間を超えて見える。env_rest = env_step - render - physics。
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class StepProfiler:
    def __init__(self, enabled: bool, device: torch.device, report_every_s: float = 30.0):
        self.enabled = enabled
        self.device = device
        self.report_every_s = report_every_s
        self.acc: dict[str, float] = {}
        self.steps = 0
        self.transitions = 0
        self._t0 = time.perf_counter()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.acc[name] = self.acc.get(name, 0.0) + (time.perf_counter() - t)

    def wrap(self, name: str, fn):
        """束縛メソッドをセクション計測に包む(rig.render / scene.step 用)。"""
        if not self.enabled:
            return fn

        def wrapped(*a, **kw):
            with self.section(name):
                return fn(*a, **kw)

        return wrapped

    def tick(self, n_transitions: int):
        """ベクトルステップごとに呼ぶ。report_every_s ごとに集計をprint。"""
        if not self.enabled:
            return
        self.steps += 1
        self.transitions += n_transitions
        wall = time.perf_counter() - self._t0
        if wall < self.report_every_s or self.steps == 0:
            return
        per = {k: v / self.steps * 1000.0 for k, v in self.acc.items()}
        env = per.get("env_step", 0.0)
        inner = per.get("render", 0.0) + per.get("physics", 0.0)
        per["env_rest"] = env - inner
        per.pop("env_step", None)
        total_ms = wall / self.steps * 1000.0
        body = " ".join(f"{k}={v:.1f}" for k, v in sorted(per.items(), key=lambda kv: -kv[1]))
        print(f"[profile] step={total_ms:.1f}ms tps={self.transitions / wall:.0f} | {body}",
              flush=True)
        self.acc.clear()
        self.steps = 0
        self.transitions = 0
        self._t0 = time.perf_counter()
