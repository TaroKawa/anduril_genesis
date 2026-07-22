"""YAML設定 → ネストしたdataclass。CLIの --set a.b.c=value 上書きに対応。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class HwConfig:
    collector_gpu: str = "3070"   # デバイス名の部分一致(WSL2は列挙順不定のため名前でマッチ)
    learner_gpu: str = "4060"


@dataclass
class RenderConfig:
    backend: str = "auto"         # "batch" | "sequential" | "none" | "auto"
    width: int = 320              # 学習時レンダ解像度(本番intrinsicsの1/2、FoV同一)
    height: int = 180
    jpeg_dr: bool = False         # JPEG劣化DR(逐次モードのみ推奨)


@dataclass
class DroneConfig:
    mass: float = 0.9                     # 仮定値(比力モデルなので並進には影響しない)
    inertia: tuple = (0.0065, 0.0065, 0.011)  # 仮定値(280mmレーサー相当)
    k_rate: float = 20.0                  # レート追従P [1/s](τ≈50ms)
    hover_thrust: float = 0.2742          # sysid
    drag_c: float = 0.72                  # sysid(線形)
    rate_max: float = 4.0                 # 内部レート目標の絶対クランプ [rad/s]
    # DR倍率レンジ(エピソードごと)
    dr_mass: tuple = (0.9, 1.1)
    dr_k_rate: tuple = (0.6, 1.6)
    dr_drag: tuple = (0.85, 1.15)
    dr_hover: tuple = (0.98, 1.02)
    dr_inertia: tuple = (0.7, 1.3)


@dataclass
class SensorConfig:
    accel_sigma: float = 0.4              # [m/s^2] 白色
    accel_bias_init: float = 0.2
    accel_bias_walk: float = 0.01
    accel_spike_p: float = 0.001
    gyro_sigma: float = 0.008             # [rad/s]
    gyro_bias_init: float = 0.02
    gyro_bias_walk: float = 0.001
    det_px_base: float = 2.0              # σ_px(d) = base + gain/max(d,1)(近いほどノイズ大)
    det_px_gain: float = 40.0
    det_dropout_base: float = 0.02
    det_dropout_close: float = 0.10       # 至近でのドロップアウト増分
    det_outlier_p: float = 0.01           # 偽検出(白ロゴ等の誤検出の模擬)
    det_delay_frames: int = 1             # 17.3ms検出遅延 → 1フレーム
    det_delay_jitter: int = 1
    img_delay_frames: int = 1             # JPEG+UDP相当
    img_delay_jitter: int = 1
    act_delay_steps: int = 0              # アクション遅延(物理ステップ)
    act_delay_jitter: int = 1
    noise_scale: float = 1.0              # カリキュラムが 0.3/0.6/1.0 を設定


@dataclass
class EnvConfig:
    num_envs: int = 64
    n_gates: int = 18
    course_seed: int = 0
    stage: int = 0
    max_episode_s: float = 60.0
    no_gate_timeout_s: float = 5.0
    collision_grace_s: float = 1.0
    spawn_below_center: float = 0.3       # スタートゲート中心からの下オフセット [m]
    spawn_pitch_deg: float = -17.8        # 前傾(実測)
    spawn_jitter_deg: float = 2.0
    render: RenderConfig = field(default_factory=RenderConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    signs_cmd: tuple = (-1.0, -1.0, -1.0)
    signs_gyro: tuple = (-1.0, -1.0, -1.0)
    signs_accel: tuple = (1.0, 1.0, 1.0)
    color_dr: bool = False                # 色DR(シーン再構築ごと。カリキュラムStage2+で有効)
    clutter: bool = False                 # 駐機機体クラッタ(Stage3+)


@dataclass
class SacConfig:
    batch_size: int = 1024
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    n_step: int = 3
    hidden: int = 512
    replay_capacity: int = 1_000_000
    success_capacity: int = 250_000
    success_ratio: float = 0.5            # → カリキュラム進行で0.25へ
    success_min_gates: int = 1
    critic_dropout: float = 0.01
    critic_layernorm: bool = True
    target_entropy: float = -4.0
    alpha_init: float = 0.1
    burn_in_steps: int = 50_000           # ホバーバイアス付きランダム方策
    learn_start: int = 20_000
    replay_ratio_cap: float = 8.0
    weight_sync_updates: int = 500
    weight_sync_sec: float = 2.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    compile: bool = True
    encoder_bf16: bool = True
    total_transitions: int = 50_000_000


@dataclass
class CurriculumConfig:
    enabled: bool = True
    window: int = 200                     # trailing成功率の窓
    thresholds: tuple = (0.7, 0.7, 0.6, 0.5)  # stage0→1,1→2,2→3,3→4
    rebuild_episodes: int = 300           # シーン再構築(コース/色DR)間隔
    seed_pool: int = 32                   # Stage3+のコースシード数
    resume_hi: float = 0.8                # 逆カリキュラム: 成功率0時の途中スポーン確率(閾値到達で各stageの下限へ線形減衰)


@dataclass
class RunConfig:
    mode: str = "async"                   # "async"(2GPU) | "sync"(1GPU、デバッグ)
    ckpt_dir: str = "checkpoints"
    ckpt_interval_s: float = 600.0
    eval_interval: int = 250_000          # 遷移数
    eval_video: bool = True
    seed: int = 0


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SacConfig = field(default_factory=SacConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    run: RunConfig = field(default_factory=RunConfig)
    hw: HwConfig = field(default_factory=HwConfig)


def _apply(dc, d: dict):
    for k, v in d.items():
        if not hasattr(dc, k):
            raise KeyError(f"unknown config key: {type(dc).__name__}.{k}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _apply(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, (list, tuple)):
            setattr(dc, k, tuple(v))
        else:
            setattr(dc, k, v)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    cfg = TrainConfig()
    if path is not None and Path(path).exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _apply(cfg, data)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        cur = getattr(node, parts[-1])
        parsed = yaml.safe_load(val)
        if isinstance(cur, tuple) and isinstance(parsed, (list, tuple)):
            parsed = tuple(parsed)
        setattr(node, parts[-1], parsed)
    return cfg
