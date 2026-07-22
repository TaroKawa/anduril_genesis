"""FPVカメラのバッチレンダリング管理。

バックエンド:
  batch      — Madronaバッチレンダラ(gs_madrona必須)。全envを一括レンダ。
  sequential — ラスタライザ + env_separate_rigid。1つのバッチカメラが
               rendered_envs_idx分のスタック画像を返す(内部でenvごとに描画)。
  none       — 画像なし(ゼロ埋め。Stage0のカメラオフブートストラップ用)。

どのバックエンドでも「カメラは1つ・出力は(スタックされた)全レンダ対象env」で統一。
Genesisカメラは OpenGL規約(ローカル -z 視線、+y 上、+x 右)。
FPVはボディ(FLU)原点・20°上チルト。
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..contracts import CAM_TILT_DEG, VFOV_DEG


def fpv_offset_T() -> np.ndarray:
    """FLUリンク座標系でのFPVカメラoffset(原点一致・20°上チルト)。"""
    t = math.radians(CAM_TILT_DEG)
    T = np.eye(4)
    # 列 = カメラ軸(x右, y上, z後方)をFLUで表現
    T[:3, 0] = (0.0, -1.0, 0.0)
    T[:3, 1] = (-math.sin(t), 0.0, math.cos(t))
    T[:3, 2] = (-math.cos(t), 0.0, -math.sin(t))
    return T


def chase_offset_T(back: float = 3.0, up: float = 1.2, pitch_down_deg: float = 15.0) -> np.ndarray:
    a = math.radians(pitch_down_deg)
    T = np.eye(4)
    T[:3, 0] = (0.0, -1.0, 0.0)
    T[:3, 1] = (math.sin(a), 0.0, math.cos(a))
    T[:3, 2] = (-math.cos(a), 0.0, math.sin(a))
    T[:3, 3] = (-back, 0.0, up)
    return T


def resolve_backend(requested: str) -> str:
    if requested in ("sequential", "none"):
        return requested
    try:
        import gs_madrona  # noqa: F401
        return "batch"
    except ImportError:
        if requested == "batch":
            print("[camera_rig] gs_madrona が見つからないため sequential にフォールバックします")
        return "sequential"


class CameraRig:
    """env用FPVカメラ。add_cameras()はscene.build()前、attach()はbuild後に呼ぶ。"""

    def __init__(self, backend: str, num_envs: int, width: int, height: int, device: torch.device,
                 max_seq_envs: int = 16):
        self.backend = backend
        self.num_envs = num_envs
        self.w, self.h = width, height
        self.device = device
        self.cam = None
        # sequentialでは全envのレンダは高価なので上限を設ける(それ以外のenvはrgbゼロ)
        self.n_rendered = num_envs if backend == "batch" else min(num_envs, max_seq_envs)
        if backend == "none":
            self.n_rendered = 0

    def rendered_envs_idx(self) -> list[int]:
        return list(range(max(self.n_rendered, 1)))

    def add_cameras(self, scene):
        if self.backend == "none":
            return
        self.cam = scene.add_camera(res=(self.w, self.h), fov=VFOV_DEG, GUI=False, near=0.05, far=250.0)

    def attach(self, drone_entity):
        if self.cam is not None:
            self.cam.attach(drone_entity.base_link, fpv_offset_T())

    def render(self) -> torch.Tensor:
        """(num_envs, H, W, 3) uint8。レンダ対象外のenvはゼロ。"""
        out = torch.zeros(self.num_envs, self.h, self.w, 3, dtype=torch.uint8, device=self.device)
        if self.backend == "none" or self.cam is None:
            return out
        rgb, _, _, _ = self.cam.render(rgb=True)
        rgb = torch.as_tensor(np.ascontiguousarray(np.asarray(rgb))) if not isinstance(rgb, torch.Tensor) else rgb
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        n = min(rgb.shape[0], self.num_envs)
        out[:n] = rgb[:n, :, :, :3].to(self.device, dtype=torch.uint8)
        return out
