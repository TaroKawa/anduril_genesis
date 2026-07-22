"""カリキュラム管理: trailing成功率でステージ進級、env実行時パラメータを更新。

| Stage | コース          | 要求        | ノイズ | 途中スポーン下限 | 色DR | クラッタ | 速度ボーナス |
|-------|-----------------|-------------|--------|------------------|------|----------|--------------|
| 0     | 直線(8ゲート)   | ゲート1     | x0.3   | 0                | -    | -        | -            |
| 1     | 緩カーブ        | 4ゲート     | x0.6   | 0.3              | -    | -        | -            |
| 2     | フル生成        | 全18        | x1.0   | 0                | o    | -        | -            |
| 3     | 32シードプール  | 全18        | x1.0   | 0.3              | o    | o        | -            |
| 4     | 同上            | 全18        | x1.0   | 0                | o    | o        | +20          |

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


STAGES = [
    StageSpec(0, 1, 0.3, 0.0, False, False, 0.0),
    StageSpec(1, 4, 0.6, 0.3, False, False, 0.0),
    StageSpec(2, 18, 1.0, 0.0, True, False, 0.0),
    StageSpec(2, 18, 1.0, 0.3, True, True, 0.0),
    StageSpec(2, 18, 1.0, 0.0, True, True, 20.0),
]


class CurriculumManager:
    def __init__(self, cfg: CurriculumConfig, start_stage: int = 0):
        self.cfg = cfg
        self.stage = start_stage
        self.results = deque(maxlen=cfg.window)
        self.episodes_since_rebuild = 0
        self.seed_counter = 0

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
        return self.episodes_since_rebuild >= self.cfg.rebuild_episodes

    def next_course_seed(self, base_seed: int) -> int:
        """再構築ごとに新しいコースシード。Stage3+はプールから循環。"""
        self.episodes_since_rebuild = 0
        self.seed_counter += 1
        if self.stage >= 3:
            return base_seed + (self.seed_counter % self.cfg.seed_pool)
        return base_seed + self.seed_counter
