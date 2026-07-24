"""YAML設定 → ネストしたdataclass。CLIの --set a.b.c=value 上書きに対応。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .user_config import uc   # config.yaml のユーザー調整値(物理/符号/センサ。無ければ既定値)


@dataclass
class HwConfig:
    # GPU指定は「整数index」または「デバイス名の部分一致文字列」を受ける
    # (同名多数GPU=A100 8枚のような環境では index を使う。名前一致はWSL2など列挙順不定向け)。
    collector_gpu: str = "3070"   # collector_gpus が空のときに使う単一collector GPU
    learner_gpu: str = "4060"     # learner GPU(1枚専有)
    num_collectors: int = 1       # collectorプロセス総数(collector_gpusへ順番=round-robinで配置)
    # collectorを分散配置するGPU(index/名前)のリスト。空なら collector_gpu 1枚のみ。
    # 例: A100×8で learner=0 / collector=1..7 → collector_gpus: [1,2,3,4,5,6,7]
    collector_gpus: tuple = ()    # env.num_envs / render は per-collector


@dataclass
class RenderConfig:
    backend: str = "auto"         # "batch" | "sequential" | "none" | "auto"
    width: int = 320              # 学習時レンダ解像度(本番intrinsicsの1/2、FoV同一)
    height: int = 180
    jpeg_dr: bool = False         # JPEG劣化DR(逐次モードのみ推奨)
    max_seq_envs: int = 16        # sequentialバックエンドの実レンダenv数上限(per collector)


@dataclass
class DroneConfig:
    """動力学パラメータ。既定値は config.yaml(dynamics/domain_rand)から。無ければ DCL実機
    open-loop sysid(2026-07-23, runs/sysid_bit16_* + analyze_sysid)較正値へフォールバック。"""

    mass: float = uc("dynamics", "mass", 0.9)
    inertia: tuple = uc("dynamics", "inertia", (0.0065, 0.0065, 0.011))
    k_rate: float = uc("dynamics", "k_rate", 35.0)          # レート追従P [1/s]
    cmd_gain: tuple = uc("dynamics", "cmd_gain", (1.0, 1.0, 0.89))  # 達成レート=cmd_gain×指令
    hover_thrust: float = uc("dynamics", "hover_thrust", 0.2694)   # A = g*(t/hover)^alpha
    thrust_alpha: float = uc("dynamics", "thrust_alpha", 1.84)
    drag_c: float = uc("dynamics", "drag_c", 0.64)          # 線形ドラッグ
    rate_max: float = uc("dynamics", "rate_max", 4.0)       # 内部レート目標クランプ [rad/s]
    # DR倍率レンジ(エピソードごと)。config.yaml: domain_rand
    dr_mass: tuple = uc("domain_rand", "mass", (0.9, 1.1))
    dr_k_rate: tuple = uc("domain_rand", "k_rate", (0.6, 1.6))
    dr_cmd_gain: tuple = uc("domain_rand", "cmd_gain", (0.95, 1.05))
    dr_drag: tuple = uc("domain_rand", "drag", (0.85, 1.15))
    dr_hover: tuple = uc("domain_rand", "hover", (0.98, 1.02))
    dr_thrust_alpha: tuple = uc("domain_rand", "thrust_alpha", (0.95, 1.05))
    dr_inertia: tuple = uc("domain_rand", "inertia", (0.7, 1.3))


@dataclass
class SensorConfig:
    """IMU/検出ノイズ・遅延。既定値は config.yaml: sensor から。"""

    accel_sigma: float = uc("sensor", "accel_sigma", 0.4)          # [m/s^2] 白色
    accel_bias_init: float = uc("sensor", "accel_bias_init", 0.2)
    accel_bias_walk: float = uc("sensor", "accel_bias_walk", 0.01)
    accel_spike_p: float = uc("sensor", "accel_spike_p", 0.001)
    gyro_sigma: float = uc("sensor", "gyro_sigma", 0.008)          # [rad/s]
    gyro_bias_init: float = uc("sensor", "gyro_bias_init", 0.02)
    gyro_bias_walk: float = uc("sensor", "gyro_bias_walk", 0.001)
    det_px_base: float = uc("sensor", "det_px_base", 2.0)          # σ_px(d)=base+gain/max(d,1)
    det_px_gain: float = uc("sensor", "det_px_gain", 40.0)
    det_dropout_base: float = uc("sensor", "det_dropout_base", 0.02)
    det_dropout_close: float = uc("sensor", "det_dropout_close", 0.10)  # 至近ドロップアウト増分
    det_outlier_p: float = uc("sensor", "det_outlier_p", 0.01)     # 偽検出
    det_delay_frames: int = uc("sensor", "det_delay_frames", 1)    # 17.3ms検出遅延→1フレーム
    det_delay_jitter: int = uc("sensor", "det_delay_jitter", 1)
    img_delay_frames: int = uc("sensor", "img_delay_frames", 1)    # JPEG+UDP相当
    img_delay_jitter: int = uc("sensor", "img_delay_jitter", 1)
    act_delay_steps: int = uc("sensor", "act_delay_steps", 0)      # アクション遅延(物理ステップ)
    act_delay_jitter: int = uc("sensor", "act_delay_jitter", 1)
    noise_scale: float = 1.0              # カリキュラムが 0.3/0.6/1.0 を設定(実行時可変・非静的)


@dataclass
class EnvConfig:
    num_envs: int = 64
    n_gates: int = 18
    course_seed: int = 0
    stage: int = 0
    max_episode_s: float = 60.0
    no_gate_timeout_s: float = 5.0
    # 物理系(config.yaml: env_physics)。衝突グレースは発進ピン解除の反力残り(~17g)を除外。
    collision_grace_s: float = uc("env_physics", "collision_grace_s", 1.5)
    spawn_below_center: float = uc("env_physics", "spawn_below_center", 0.3)  # 中心からの下[m]
    spawn_pitch_deg: float = uc("env_physics", "spawn_pitch_deg", -17.8)      # 前傾(実測)
    spawn_jitter_deg: float = uc("env_physics", "spawn_jitter_deg", 2.0)
    render: RenderConfig = field(default_factory=RenderConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    # 指令・観測の符号(左手系挙動の再現)。config.yaml: signs。
    # bit16採用で signs_cmd を(+1,+1,-1)→(-1,-1,+1)へ再導出(指令→物理回転が全軸反転、
    # signs_gyro/accel は据え置き。runs/sysid_bit16_0723)。詳細は config.yaml / sim2sim-gaps。
    signs_cmd: tuple = uc("signs", "cmd", (-1.0, -1.0, 1.0))
    signs_gyro: tuple = uc("signs", "gyro", (1.0, 1.0, -1.0))
    signs_accel: tuple = uc("signs", "accel", (1.0, 1.0, 1.0))
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
    replay_capacity: int = 250_000        # 履歴窓(K,384)fp16で~12KB/遷移 → 250kで約3GB
    success_capacity: int = 50_000
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
    encoder_chunk: int = 512              # DINOv2 forwardの1回あたり最大画像数。全env一括だと
                                          # (N,3,224,224)の巨大activationでcollector GPUが80GB上限に
                                          # 達しcuMemAllocAsync OOM→クラッシュ。凍結エンコーダは
                                          # サンプル独立なのでchunk分割しても出力は厳密一致。0で無効(一括)。
    total_transitions: int = 50_000_000


@dataclass
class CurriculumConfig:
    enabled: bool = True
    window: int = 200                     # trailing成功率の窓
    thresholds: tuple = (0.7, 0.7, 0.7, 0.6, 0.5)  # stage0→1,1→2,2→3,3→4,4→5
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
    profile: bool = False                 # collectorのステップ内訳計測(sync入りで数%遅くなる)


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
