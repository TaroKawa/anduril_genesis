"""コース生成(Genesis非依存・numpy)。

実コースの特徴(ユーザー確認+実スクリーンショット):
  - カーブ・高低差が多い。急上昇/急降下区間(青パスが天井近くまで立ち上がる)
  - ゲートは柱の近傍を縫うように配置され、パスが柱列の間を蛇行する
  - スタートゲート + 18ゲート。ゲート外形2.7m/内側開口1.5m/奥行0.26m(仕様§3.7)

全てNED座標(d=下向き正)。ゲート0=スタートゲート。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GATE_OUTER = 2.7
GATE_INNER = 1.5
GATE_DEPTH = 0.26
BAR_W = (GATE_OUTER - GATE_INNER) / 2.0  # 0.6


@dataclass
class HallSpec:
    length: float = 120.0   # N方向 [m]
    width: float = 50.0     # E方向 [m]
    height: float = 10.0    # [m]
    margin: float = 5.0     # コースが壁から取る距離 [m]


@dataclass
class GateSpec:
    center_ned: np.ndarray  # (3,)
    yaw: float              # ゲート法線の方位(コース進行方向) [rad]


@dataclass
class CourseSpec:
    gates: list[GateSpec]
    ribbon_pts: np.ndarray          # (M,3) NED、ゲート中心を貫通するCatmull-Romサンプル
    pillars: np.ndarray             # (P,2) NED水平位置
    hall: HallSpec
    seed: int
    gate_cum_arc: np.ndarray = field(default=None)  # (n_gates,) 各ゲートまでの累積弧長
    total_arc: float = 0.0

    @property
    def n_gates(self) -> int:
        return len(self.gates)


def _catmull_rom(points: np.ndarray, samples_per_seg: int = 40) -> np.ndarray:
    """(K,3) 制御点 → 全点を通る補間曲線 (M,3)。端点は複製で処理。"""
    p = np.concatenate([points[:1], points, points[-1:]], axis=0)
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
        out.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    out.append(points[-1:])
    return np.concatenate(out, axis=0)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点p(3,)と線分ab(3,)の距離。"""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


class CourseGenerator:
    """棄却サンプリング付きランダムウォークで18ゲートコースを生成。

    stage 0: 直線(離陸+ゲート1用の易しいコース)
    stage 1: 緩カーブ(Δψ≤25°、高低差控えめ)
    stage 2+: フルレンジ(Δψ≤50°、高度1.5-4.5m + 急上昇/急降下区間~7m)
    """

    def __init__(self, seed: int, stage: int = 2, n_gates: int = 18, hall: HallSpec | None = None):
        self.seed = seed
        self.stage = stage
        self.n_gates = n_gates
        self.hall = hall or HallSpec()

    def generate(self) -> CourseSpec:
        for attempt in range(64):
            rng = np.random.default_rng(self.seed + attempt * 1000)
            spec = self._try_generate(rng)
            if spec is not None:
                spec.seed = self.seed
                return spec
        raise RuntimeError(f"course generation failed for seed={self.seed}")

    # --- 内部 ---

    def _params(self, rng):
        # stage0はほぼ直線なので格納庫に収まるようゲート数を絞る(要求はゲート1通過のみ)
        if self.stage <= 0:
            return dict(dpsi_max=np.radians(8), z_range=(1.8, 2.2), climbs=0,
                        n_gates=min(self.n_gates, 8), seg=(8.0, 12.0))
        if self.stage == 1:
            return dict(dpsi_max=np.radians(25), z_range=(1.5, 3.5), climbs=0,
                        n_gates=self.n_gates, seg=(8.0, 15.0))
        return dict(dpsi_max=np.radians(50), z_range=(1.5, 4.5), climbs=int(rng.integers(1, 4)),
                    n_gates=self.n_gates, seg=(8.0, 18.0))

    def _try_generate(self, rng) -> CourseSpec | None:
        hall = self.hall
        p = self._params(rng)

        # スタートゲート: ホール手前側、+N向き
        start = np.array([-hall.length / 2 + hall.margin + 3.0, 0.0, -1.8])
        centers = [start]
        headings = [0.0]
        psi = 0.0
        dpsi_prev = 0.0

        n_gates = p["n_gates"]
        # 急上昇/急降下を入れるゲートindex(実画像3の垂直クライム区間の再現)
        climb_idx = set(rng.choice(np.arange(3, n_gates - 1), size=p["climbs"], replace=False).tolist()) \
            if p["climbs"] > 0 else set()

        for i in range(1, n_gates + 1):
            ok = False
            for retry in range(50):
                L = rng.uniform(*p["seg"])
                dpsi = 0.6 * dpsi_prev + 0.4 * rng.uniform(-p["dpsi_max"], p["dpsi_max"])
                psi_new = _wrap(psi + dpsi)
                prev = centers[-1]
                # 壁回避ステアリング: 壁に近い/リトライが続くときは中心方向へ寄せる
                near_wall = (
                    abs(prev[0]) > hall.length / 2 - hall.margin - 12.0
                    or abs(prev[1]) > hall.width / 2 - hall.margin - 12.0
                )
                if near_wall or retry >= 10:
                    psi_center = np.arctan2(-prev[1], -prev[0])
                    blend = 0.5 if self.stage >= 1 else 0.25
                    psi_new = _wrap(psi + _wrap(psi_center - psi) * blend)
                cand = prev + np.array([L * np.cos(psi_new), L * np.sin(psi_new), 0.0])
                if i in climb_idx:
                    # 急上昇/急降下: 天井近く(~7m)まで、または低空へ
                    z = -rng.uniform(5.5, 7.0) if prev[2] > -4.0 else -rng.uniform(1.5, 2.5)
                else:
                    z = np.clip(prev[2] + rng.uniform(-1.5, 1.5), -p["z_range"][1], -p["z_range"][0])
                cand[2] = z
                if not self._in_hall(cand):
                    continue
                if any(np.linalg.norm(cand - c) < 6.0 for c in centers):
                    continue
                # 自己交差防止: 新ゲートが既存セグメントに近すぎない/新セグメントが
                # 既存ゲートに近すぎない(コースが他ゲートの枠を突き抜けるのを防ぐ)
                clearance = 3.5
                if any(_point_seg_dist(cand, centers[j], centers[j + 1]) < clearance
                       for j in range(len(centers) - 1)):
                    continue
                if any(_point_seg_dist(c, prev, cand) < clearance for c in centers[:-1]):
                    continue
                ok = True
                break
            if not ok:
                return None
            centers.append(cand)
            headings.append(psi_new)
            psi, dpsi_prev = psi_new, dpsi

        centers = np.stack(centers)

        # ゲートyaw = 入る向きと出る向きの円平均
        gates = []
        for i in range(len(centers)):
            h_in = headings[i]
            h_out = headings[i + 1] if i + 1 < len(headings) else headings[i]
            yaw = np.arctan2(np.sin(h_in) + np.sin(h_out), np.cos(h_in) + np.cos(h_out))
            gates.append(GateSpec(center_ned=centers[i], yaw=float(yaw)))

        ribbon = _catmull_rom(centers, samples_per_seg=40)
        if not all(self._in_hall(q, slack=2.0) for q in ribbon[::10]):
            return None

        pillars = self._make_pillars(rng, ribbon)
        spec = CourseSpec(gates=gates, ribbon_pts=ribbon, pillars=pillars, hall=hall, seed=self.seed)
        self._compute_arc(spec)
        return spec

    def _in_hall(self, pt: np.ndarray, slack: float = 0.0) -> bool:
        hall = self.hall
        m = max(hall.margin - slack, 1.0)
        return (
            abs(pt[0]) < hall.length / 2 - m
            and abs(pt[1]) < hall.width / 2 - m
            and -(hall.height - 2.0) < pt[2] < -0.8
        )

    def _make_pillars(self, rng, ribbon: np.ndarray) -> np.ndarray:
        """壁沿い2列 + 中央帯にも数本(実画像: 柱列の間をパスが縫う)。リボンに近すぎる柱は削除。"""
        hall = self.hall
        rows_y = [-(hall.width / 2 - 3.0), -(hall.width / 4), hall.width / 4, hall.width / 2 - 3.0]
        xs = np.arange(-hall.length / 2 + 6.0, hall.length / 2 - 6.0, 12.0)
        pillars = []
        for y in rows_y:
            for x in xs:
                jitter = rng.uniform(-1.5, 1.5, size=2)
                pillars.append([x + jitter[0], y + jitter[1]])
        pillars = np.array(pillars)
        d = np.linalg.norm(pillars[:, None, :] - ribbon[None, :, :2], axis=-1).min(axis=1)
        return pillars[d > 2.5]

    @staticmethod
    def _compute_arc(spec: CourseSpec) -> None:
        seg = np.linalg.norm(np.diff(spec.ribbon_pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        spec.total_arc = float(cum[-1])
        gate_arc = []
        for g in spec.gates:
            i = int(np.argmin(np.linalg.norm(spec.ribbon_pts - g.center_ned, axis=1)))
            gate_arc.append(cum[i])
        spec.gate_cum_arc = np.array(gate_arc)


def ribbon_mesh(ribbon_pts: np.ndarray, width: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """リボン(発光ガイドパス)の三角形ストリップを生成。NED座標のまま返す。

    帯の面の向きはパスの接線に追従し、水平区間では水平帯・急上昇区間では
    実映像同様の垂直な壁状バンドに自然になる(側方ベクトル=接線×鉛直、
    接線が鉛直に近づくと側方を水平面内に固定)。
    returns (vertices (2M,3), faces (2(M-1),3))
    """
    pts = ribbon_pts
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    up = np.array([0.0, 0.0, -1.0])  # NEDの上
    side = np.cross(t, up)
    n = np.linalg.norm(side, axis=1, keepdims=True)
    # 接線がほぼ鉛直の区間: 側方をE軸に固定
    side = np.where(n > 0.2, side / np.maximum(n, 1e-9), np.array([0.0, 1.0, 0.0]))
    side /= np.linalg.norm(side, axis=1, keepdims=True)

    v0 = pts + side * (width / 2)
    v1 = pts - side * (width / 2)
    verts = np.empty((2 * len(pts), 3))
    verts[0::2] = v0
    verts[1::2] = v1
    faces = []
    for i in range(len(pts) - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces += [[a, b, c], [b, d, c]]
    return verts, np.array(faces, dtype=np.int64)
