"""カリキュラム管理: trailing成功率でステージ進級、env実行時パラメータを更新。

| Stage | コース            | 要求        | ノイズ | 途中スポーン下限 | 色DR | クラッタ | 速度ボーナス | 動力学DR |
|-------|-------------------|-------------|--------|------------------|------|----------|--------------|----------|
| 0     | 直線(8ゲート)     | ゲート1     | x0.3   | 0                | -    | -        | -            | x1.0     |
| 1     | 近接緩カーブ(5-8m)| 2ゲート     | x0.4   | 0                | -    | -        | -            | x1.0     |
| 2     | 緩カーブ(標準間隔)| 4ゲート     | x0.6   | 0.3              | -    | -        | -            | x1.0     |
| 3     | フル生成          | 全18        | x1.0   | 0                | o    | -        | -            | x1.0     |
| 4     | 32シードプール    | 全18        | x1.0   | 0.3              | o    | o        | -            | x1.0     |
| 5     | per-env(間隔~6m)  | 全18        | x1.0   | 0                 | o    | o        | -(廃止)     | x1.0     |
| 6     | per-env(間隔~4.5m)| 全18        | x1.0   | 0                 | o    | o        | -            | x1.0     |
| 7     | per-env(間隔~3m)  | 全18        | x1.0   | 0                 | o    | o        | -            | x1.0     |

Stage 5-7 は「実シミュレータ(DCL本番シム)への汎化」を狙う堅牢化レジーム。5→6→7でゲート間隔の
下限を 6m→4.5m→3m へ漸減し(course_stage 3→4→5)、各stageを success_rate 0.8 まで習熟してから
次の間隔へ進む。per-env/初期位置スタート/全視覚DRは共通。速度ボーナスは
廃止し、視覚DR(photo_dr/色DR/クラッタ)でsim2simギャップに耐える方策へ仕上げる。
非視覚ノイズ・動力学DRは較正値のまま(x1.0): デプロイ実測(2026-07)で動力学・レート追従
(0.97/0.97/0.89)・映像遅延は較正どおり一致し、残るギャップは映像の見た目のみと判明した。
固有受容を実際よりノイジーに見せると方策が視覚依存を強めて逆効果のため、旧x1.5は廃止。

Stage 1(近接緩カーブ)は直線→標準カーブの間の中間難度: ゲート間隔を5-8mに
詰め、通過直後に次ゲートが視界に入る=報酬までの距離が短い状態でカーブ操作を学ぶ。

途中スポーン確率は逆カリキュラム: 各ステージ開始時はresume_hi(既定0.8)で
コース全域のゲート手前からスポーンし、成功率が進級閾値に近づくほど上表の
下限へ線形減衰して正規スタート比率を上げる(resume_prob_now)。
成功判定はスポーン地点からの相対通過数(スキップ分のクレジットなし)。

コース形状・色DR・クラッタの変更はシーン再構築が必要(needs_rebuild)。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import CurriculumConfig


@dataclass
class StageSpec:
    course_stage: int      # CourseGeneratorに渡すstage(0=直線,1=緩,2=フル)
    required_gates: int
    noise_scale: float
    resume_prob: float
    color_dr: bool
    clutter: bool
    speed_finish_w: float
    dr_scale: float = 1.0  # 動力学DRレンジの拡大係数(現在は全ステージx1.0=較正レンジのまま)


STAGES = [
    StageSpec(0, 1, 0.3, 0.0, False, False, 0.0),
    StageSpec(1, 2, 0.4, 0.0, False, False, 0.0),   # 近接緩カーブ(中間難度)
    StageSpec(2, 4, 0.6, 0.3, False, False, 0.0),
    StageSpec(3, 18, 1.0, 0.0, True, False, 0.0),
    StageSpec(3, 18, 1.0, 0.3, True, True, 0.0),
    # Stage5: 速度ボーナス廃止 → 実シミュレータ汎化ステージ。ノイズ・動力学DRは較正値の
    # まま(x1.0)。デプロイ実測で動力学・レート追従・遅延は較正どおり一致しており、残る
    # sim2simギャップは映像の見た目のみと判明したため、非視覚ノイズの拡大(旧x1.5)は廃止。
    # ランダム化はphoto_dr(測光DR)・色DR・クラッタなど視覚側に集中させる。
    # 途中スポーン(resume)は使わず初期位置スタートのみ(resume_prob=0)。コース多様化は
    # per-env(各envに別コース)で行い、6000エピソード再構築には頼らない。
    StageSpec(3, 18, 1.0, 0.0, True, True, 0.0, dr_scale=1.0),
    # Stage6,7: sim2sim堅牢化を保ったままゲート間隔を漸減する追加レジーム。course_stage 4→5 は
    # course._params でゲート間隔下限(min_gap/seg/clearance)を 6m→4.5m→3m へ狭める。per-env・
    # 初期位置スタート(resume_prob=0)・全視覚DRは stage5 と同じ。各stageを success_rate 0.8 まで
    # 習熟してから次の間隔へ進む(curriculum.thresholds)。
    StageSpec(4, 18, 1.0, 0.0, True, True, 0.0, dr_scale=1.0),  # idx6 ゲート間隔~4.5m
    StageSpec(5, 18, 1.0, 0.0, True, True, 0.0, dr_scale=1.0),  # idx7 ゲート間隔~3m(下限)
]

# per-envコース/初期位置スタート(resume_prob=0)/定期再構築なし を使う最初のステージ。
# ここから最終ステージまでが「sim2sim堅牢化 + ゲート間隔漸減(6m→3m)」の per-env レジーム。
PER_ENV_START = 5
# 後方互換の別名(旧: 最終ステージindexの意味で使っていた箇所向け)。現在は per-env 開始と同義。
PER_ENV_STAGE = PER_ENV_START


class CurriculumManager:
    def __init__(self, cfg: CurriculumConfig, start_stage: int = 0):
        self.cfg = cfg
        self.stage = start_stage
        self.results = deque(maxlen=cfg.window)
        self.episodes_since_rebuild = 0
        self.seed_counter = 0
        # per-envコースが実際に有効か(collectorが設定)。stage5でも env.per_env_courses=false
        # なら従来のシーン再構築でコースを回す(リボン/柱/クラッタ等のフルビジュアル維持)。
        self.per_env_active = False

    @property
    def spec(self) -> StageSpec:
        return STAGES[min(self.stage, len(STAGES) - 1)]

    def record_episodes(self, successes) -> None:
        """doneしたエピソードの成功フラグ(iterable of bool)を記録。"""
        successes = list(successes)
        for s in successes:
            self.results.append(bool(s))
        self.episodes_since_rebuild += len(successes)

    def success_rate(self) -> float:
        if len(self.results) < self.cfg.window // 2:
            return 0.0
        return sum(self.results) / len(self.results)

    def resume_prob_now(self) -> float:
        """逆カリキュラムの途中スポーン確率。

        ステージ開始直後(成功率0)はresume_hi(既定0.8)で全ゲート付近から練習し、
        成功率が進級閾値へ近づくにつれ各ステージの下限(spec.resume_prob)へ
        線形に減衰させて正規スタートの比率を上げる。
        """
        spec = self.spec
        # 最終ステージは初期位置スタートのみ(逆カリキュラムのアニールを無効化)
        if self.stage >= PER_ENV_STAGE:
            return 0.0
        if not self.cfg.enabled:
            return spec.resume_prob
        th = self.cfg.thresholds[min(self.stage, len(self.cfg.thresholds) - 1)]
        annealed = self.cfg.resume_hi * max(0.0, 1.0 - self.success_rate() / max(th, 1e-6))
        return max(spec.resume_prob, annealed)

    def maybe_advance(self) -> bool:
        """進級したらTrue(進級はシーン再構築を要求する)。"""
        if not self.cfg.enabled or self.stage >= len(STAGES) - 1:
            return False
        th = self.cfg.thresholds[min(self.stage, len(self.cfg.thresholds) - 1)]
        if len(self.results) >= self.cfg.window and self.success_rate() >= th:
            self.stage += 1
            self.results.clear()
            return True
        return False

    def needs_rebuild(self) -> bool:
        # per-envコース有効時は多様なコースを常時適用するため定期再構築しない。
        # (per-env無効の最終ステージは従来どおり再構築でコースを回す)
        if self.per_env_active:
            return False
        return self.episodes_since_rebuild >= self.cfg.rebuild_episodes

    def next_course_seed(self, base_seed: int) -> int:
        """再構築ごとに新しいコースシード。Stage4+はプールから循環。"""
        self.episodes_since_rebuild = 0
        self.seed_counter += 1
        if self.stage >= 4:
            return base_seed + (self.seed_counter % self.cfg.seed_pool)
        return base_seed + self.seed_counter
