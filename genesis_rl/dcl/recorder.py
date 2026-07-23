# -*- coding: utf-8 -*-
"""本番シム(DCL)飛行の同期ログ取り。sim-to-simギャップ分析 / Phase 2 fine-tune 用。

client.py の推論ループから1決定(30Hz)ごとに呼ばれ、方策が実際に見た観測・出した
アクション・そのときのテレメトリを時刻同期で1レコードにまとめて保存する。

出力レイアウト(--record-dir DIR):
  DIR/frames/000123.jpg   … その決定で方策が見た 224x224 RGB(= obs_rgb、encoder入力そのもの)
  DIR/steps.jsonl         … 1行=1決定。下記フィールド(numpyはlist化)
  DIR/meta.json           … ckpt / 契約hash / 検出器種別 / 起動時刻など

steps.jsonl の1レコード:
  step, t_wall, t_rel        決定インデックス / 壁時計 / 開始からの相対秒
  vec[55]                    方策へ渡した観測ベクトル(契約 contracts.VEC_* 準拠)
  raw_action[4]              方策出力 a∈[-1,1]^4
  cmd[4]                     物理コマンド (roll_rate, pitch_rate, yaw_rate[rad/s], thrust)
  gate{visible,center,rel_dist,age_s}  そのフレームの生ゲート検出(vec化前)
  race{active_gate_index,pin_released,start_pending,race_finished}
  imu{gyro[3],accel[3],t}    直近 HIGHRES_IMU 生値(符号は本番規約=左手系のまま)
  collision                  この決定の直後に衝突判定が立ったか(bool)
  frame                      対応するjpgの相対パス("frames/000123.jpg")

なぜこれで足りるか:
  - 方策の再現/オフライン評価: (frame, vec) → policy.act で決定を丸ごと再計算できる。
  - レート追従sysid: cmd[:3](指令レート)と後続 imu.gyro の応答から k_rate/遅れを推定。
  - rel_dist較正: 生 gate.rel_dist の分布を Genesis 側 SimGateDetector と突き合わせる。
  - one-hot規約検証: race.active_gate_index の実値系列を見て off-by-one を確定できる。
  - Phase 2: (frame, vec, action, 次frame) を遷移として貯める土台になる。
"""

from __future__ import annotations

import json
import os
import time

import numpy as np


def _to_list(x):
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.reshape(-1)]
    if isinstance(x, (tuple, list)):
        return [float(v) for v in x]
    return x


class FlightRecorder:
    """飛行1本ぶんを DIR に逐次書き出す。Ctrl+C されても途中まで残る(逐次flush)。"""

    def __init__(self, out_dir: str, meta: dict | None = None, save_frames: bool = True):
        self.dir = out_dir
        self.frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.save_frames = save_frames
        self._f = open(os.path.join(out_dir, "steps.jsonl"), "w", buffering=1)  # 行バッファ
        # 高レートIMUログ(HIGHRES_IMU受信ごと。steps.jsonlの30Hzでは時定数/遅延が測れない)
        self._fimu = open(os.path.join(out_dir, "imu.jsonl"), "w", buffering=1)
        self.imu_samples = 0
        self.step = 0
        self.t0 = time.time()
        m = {"t_start_wall": self.t0, "save_frames": save_frames}
        m.update(meta or {})
        with open(os.path.join(out_dir, "meta.json"), "w") as mf:
            json.dump(m, mf, indent=2, default=str)
        print(f"[recorder] logging to {out_dir}/ (frames={save_frames})", flush=True)

    def record(self, *, rgb224, vec, raw_action, cmd, shared: dict, collision: bool = False):
        """1決定を記録。rgb224 は RGB uint8 (224,224,3)、vec/raw_action/cmd は np/list。"""
        now = time.time()
        frame_rel = ""
        if self.save_frames and rgb224 is not None:
            import cv2
            frame_rel = f"frames/{self.step:06d}.jpg"
            # obs_rgb は RGB。imwrite は BGR を期待するので変換して保存(見た目が正しくなる)
            cv2.imwrite(os.path.join(self.dir, frame_rel),
                        cv2.cvtColor(np.ascontiguousarray(rgb224), cv2.COLOR_RGB2BGR))

        og = shared.get("obs_gate") or {}
        imu = shared.get("imu") or {}
        race = shared.get("race") or {}
        rec = {
            "step": self.step,
            "t_wall": now,
            "t_rel": now - self.t0,
            "vec": _to_list(vec),
            "raw_action": _to_list(raw_action),
            "cmd": _to_list(cmd),
            "gate": {
                "visible": int(og.get("visible", 0)),
                "center": _to_list(og.get("center", (0.5, 0.5))),
                "rel_dist": float(og.get("rel_dist", 1.0)),
                "age_s": float(now - float(og.get("t_wall", now))),
            },
            "race": {
                "active_gate_index": int(race.get("active_gate_index", -1)),
                "pin_released": bool(race.get("pin_released", False)),
                "start_pending": bool(race.get("start_pending", False)),
                "race_finished": bool(race.get("race_finished", False)),
            },
            "imu": {
                "gyro": _to_list(imu.get("gyro", (0.0, 0.0, 0.0))),
                "accel": _to_list(imu.get("accel", (0.0, 0.0, 0.0))),
                "t": float(imu.get("t", 0.0)),
            },
            "collision": bool(collision),
            "frame": frame_rel,
        }
        self._f.write(json.dumps(rec) + "\n")
        self.step += 1

    def record_event(self, ev: dict):
        """非同期イベント(衝突等)を events.jsonl へ。steps.jsonl の30Hzでは取り漏れるもの。"""
        if not hasattr(self, "_fev"):
            self._fev = open(os.path.join(self.dir, "events.jsonl"), "w", buffering=1)
        self._fev.write(json.dumps(ev) + "\n")

    def record_imu(self, sample: dict):
        """HIGHRES_IMU 1サンプルを imu.jsonl へ。
        {t_rx_wall, t_sim, gyro[3], accel[3]}(生値・符号は本番規約のまま)。"""
        self._fimu.write(json.dumps({
            "t_rx_wall": float(sample["t_rx_wall"]),
            "t_sim": float(sample["t_sim"]),
            "gyro": _to_list(sample["gyro"]),
            "accel": _to_list(sample["accel"]),
        }) + "\n")
        self.imu_samples += 1

    def close(self):
        for f in (self._f, self._fimu, getattr(self, "_fev", None)):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        print(f"[recorder] wrote {self.step} steps / {self.imu_samples} imu samples "
              f"to {self.dir}/", flush=True)
