# 記録済みDCL飛行を「修正後のデプロイ観測パイプライン」で再構成し、vec分布を学習側と比較する
import json
import sys

import cv2
import numpy as np

from genesis_rl import contracts as C
from genesis_rl.dcl.client import ACCEL_OBS_SIGN, GYRO_OBS_SIGN, make_gate_detector

det = make_gate_detector("yolox", "YOLOX_outputs_x/yolox_x_custom/best_ckpt.pth", None)


def rebuild(d, stride=6):
    rows = [json.loads(line) for line in open(f"{d}/steps.jsonl")][::stride]
    V, prev_a, prev_prev_a = [], np.zeros(4), np.zeros(4)
    for r in rows:
        vec = np.zeros(C.VEC_DIM, np.float32)
        imu = r["imu"]
        vec[C.VEC_GYRO] = np.clip(np.asarray(imu["gyro"]) * GYRO_OBS_SIGN / C.RATE_SCALE, -1, 1)
        vec[C.VEC_ACCEL] = np.clip(np.asarray(imu["accel"]) * ACCEL_OBS_SIGN / C.ACCEL_SCALE, -2.5, 2.5)
        img = cv2.imread(f"{d}/{r['frame']}")
        g = det(cv2.resize(img, (640, 360))) if img is not None else {"visible": 0}
        age_n = float(np.clip(r["gate"].get("age_s", 0.05) / C.GATE_OBS_MAX_AGE_S, 0, 1))
        if g["visible"]:
            vec[C.VEC_GATE] = (np.clip(g["center"][0] * 2 - 1, -1.5, 1.5),
                               np.clip(g["center"][1] * 2 - 1, -1.5, 1.5),
                               1.0, g["rel_dist"], age_n)
        else:
            vec[C.VEC_GATE] = (0.0, 0.0, 0.0, 1.0, age_n)
        vec[C.VEC_ONEHOT.start + min(max(r["race"]["active_gate_index"], 0), C.MAX_GATES - 1)] = 1.0
        vec[C.VEC_LAST_ACTION] = prev_prev_a
        prev_prev_a, prev_a = prev_a, np.asarray(r["raw_action"], np.float32)
        V.append(vec)
    return np.stack(V)


def show(V, name):
    g = V[:, 6:11]
    vis = g[:, 2] > 0.5
    print(f"{name:22s} gyro absmax {np.abs(V[:, 0:3]).max():.2f}  accel_z {V[:, 5].mean():+.3f}  "
          f"可視{vis.mean() * 100:3.0f}%  rel {g[vis, 3].mean():.3f}(p10 {np.percentile(g[vis, 3], 10):.3f})  "
          f"age_n 可視{g[vis, 4].mean():.3f}/不可視{g[~vis, 4].mean() if (~vis).any() else float('nan'):.3f}", flush=True)


for d in sys.argv[1:]:
    show(rebuild(d), d.split("/")[-1])
z = np.load("runs/obs_audit_genesis.npz")
show(z["V"], "Genesis(学習側)")
