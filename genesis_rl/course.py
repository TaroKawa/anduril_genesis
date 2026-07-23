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
    pitch: float = 0.0      # ゲート面の傾き [rad](形は変えず少しだけ傾ける)
    roll: float = 0.0

    def rotation_ned(self) -> np.ndarray:
        """ゲートローカル→NEDの回転行列。列 = (法線, 面内側方, 面内下方)。"""
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx


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
    stage 1: 近接緩カーブ(ゲート間隔5-8m。次ゲートがすぐ視界に入る中間難度)
    stage 2: 緩カーブ(Δψ≤40°、高低差控えめ、標準間隔)
    stage 3+: フルレンジ(Δψ≤90°、高度1.5-4.5m + 急上昇/急降下区間~7m)
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
                        n_gates=min(self.n_gates, 8), seg=(8.0, 12.0),
                        tilt=0.0, sharp_p=0.0, min_gap=6.0)
        if self.stage == 1:
            # 近接緩カーブ: ゲートを詰めて置く(通過→次ゲートがすぐ視界に入り
            # 報酬までの距離が短い)。直線→緩カーブへの中間難度。
            # far_frac<1でホール奥行きの使用分を絞る(奥行き予算がゲート間隔を
            # 決めるため、segを狭めるだけでは間隔が縮まらない)
            return dict(dpsi_max=np.radians(25), z_range=(1.6, 2.6), climbs=0,
                        n_gates=self.n_gates, seg=(4.5, 7.0),
                        tilt=np.radians(3), sharp_p=0.0, min_gap=4.0, far_frac=0.8)
        if self.stage == 2:
            return dict(dpsi_max=np.radians(40), z_range=(1.5, 3.5), climbs=0,
                        n_gates=self.n_gates, seg=(7.0, 13.0),
                        tilt=np.radians(5), sharp_p=0.1, min_gap=6.0)
        return dict(dpsi_max=np.radians(90), z_range=(1.5, 4.5), climbs=int(rng.integers(1, 4)),
                    n_gates=self.n_gates, seg=(6.0, 13.0),
                    tilt=np.radians(12), sharp_p=0.25, min_gap=6.0)

    def _try_generate(self, rng) -> CourseSpec | None:
        """始点(ホール手前)→終点(奥)へ縦方向に進行するコース。

        方位ψは+N基準で±95°にクランプ(後戻りしない=縦方向に伸びる)。
        横方向は自由: ターンは最大90°、確率的に70-90°の急ターン(真横へ行くゲート)を入れる。
        奥行きの残り予算が乏しくなったら前進方向へ絞る。
        """
        hall = self.hall
        p = self._params(rng)
        psi_lim = np.radians(95.0)
        far_limit = hall.length / 2 - hall.margin - 3.0

        # スタートゲート: ホール手前側、+N向き
        start = np.array([-hall.length / 2 + hall.margin + 3.0, rng.uniform(-4.0, 4.0), -1.8])
        # 近接ステージ: 使う奥行きを絞ってゲート間隔を詰める
        far_limit = start[0] + (far_limit - start[0]) * p.get("far_frac", 1.0)
        centers = [start]
        headings = [0.0]
        psi = 0.0

        n_gates = p["n_gates"]
        climb_idx = set(rng.choice(np.arange(3, n_gates - 1), size=p["climbs"], replace=False).tolist()) \
            if p["climbs"] > 0 else set()

        for i in range(1, n_gates + 1):
            ok = False
            for retry in range(60):
                prev = centers[-1]
                # 奥行き予算の均等割り: 残りゲートで残りN距離をほぼ使い切る
                remaining_n = max(far_limit - prev[0], 1.0)
                target_dn = remaining_n / (n_gates - i + 1)

                # ターン角: 通常は予算から導出±ノイズ、確率的に~90°の急ターン(真横)
                base = np.arccos(np.clip(target_dn / p["seg"][1], 0.05, 0.98))
                if rng.random() < p["sharp_p"]:
                    mag = rng.uniform(0.8, 1.0) * p["dpsi_max"]
                else:
                    mag = np.clip(base + rng.uniform(-0.5, 0.5), 0.0, p["dpsi_max"])
                # 横方向の符号: 慣性 or ランダム、壁際では中央へ
                if abs(prev[1]) > hall.width / 2 - hall.margin - 6.0:
                    sign = -np.sign(prev[1])
                elif abs(psi) > 0.1 and rng.random() < 0.5:
                    sign = np.sign(psi)
                else:
                    sign = rng.choice([-1.0, 1.0])
                psi_target = float(np.clip(sign * mag, -psi_lim, psi_lim))
                # ゲートでのターン角(進入→退出)を±dpsi_maxに制限。これを超えると
                # ヘアピン(ゲート法線が進入/退出の両方から70°超)になり通過不能になる
                dpsi = float(np.clip(_wrap(psi_target - psi), -p["dpsi_max"], p["dpsi_max"]))
                psi_new = float(np.clip(psi + dpsi, -psi_lim, psi_lim))

                # セグメント長: 前進成分がtarget_dn近くになるように選ぶ
                cos_p = max(np.cos(psi_new), 0.05)
                L = float(np.clip(target_dn / cos_p * rng.uniform(0.8, 1.3), *p["seg"]))

                cand = prev + np.array([L * np.cos(psi_new), L * np.sin(psi_new), 0.0])
                if cand[0] < prev[0] - 1.0:  # 縦方向にほぼ単調(後戻り禁止)
                    continue
                if i in climb_idx:
                    # 急上昇/急降下: 天井近く(~7m)まで、または低空へ
                    z = -rng.uniform(5.5, 7.0) if prev[2] > -4.0 else -rng.uniform(1.5, 2.5)
                else:
                    z = np.clip(prev[2] + rng.uniform(-1.5, 1.5), -p["z_range"][1], -p["z_range"][0])
                cand[2] = z
                if not self._in_hall(cand):
                    continue
                if any(np.linalg.norm(cand - c) < p["min_gap"] for c in centers):
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
            headings.append(float(psi_new))
            psi = float(psi_new)

        centers = np.stack(centers)

        # ゲートyaw = 入る向きと出る向きの円平均 + 面の小傾き(形は不変)
        gates = []
        for i in range(len(centers)):
            h_in = headings[i]
            h_out = headings[i + 1] if i + 1 < len(headings) else headings[i]
            yaw = np.arctan2(np.sin(h_in) + np.sin(h_out), np.cos(h_in) + np.cos(h_out))
            tilt = 0.0 if i == 0 else p["tilt"]  # スタートゲートは水平
            gates.append(GateSpec(center_ned=centers[i], yaw=float(yaw),
                                  pitch=float(rng.uniform(-tilt, tilt)),
                                  roll=float(rng.uniform(-tilt, tilt))))

        ribbon = _catmull_rom(centers, samples_per_seg=RIBBON_SAMPLES_PER_SEG)
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


RIBBON_SAMPLES_PER_SEG = 40


def ribbon_segments(spec: CourseSpec, width: float = 1.8) -> list[tuple[np.ndarray, np.ndarray]]:
    """ゲートiへ向かうリボン区間(ゲートi-1→i)のメッシュを個別に返す(i=1..n_gates-1)。

    動的表示(アクティブゲートから5ゲート先まで表示・通過区間は消す)のために
    区間ごとに別エンティティにする。区間の端点は隣接区間と1点共有して繋がって見える。
    """
    per = RIBBON_SAMPLES_PER_SEG
    segs = []
    for i in range(1, spec.n_gates):
        lo = per * (i - 1)
        hi = min(per * i + 1, len(spec.ribbon_pts))
        segs.append(ribbon_mesh(spec.ribbon_pts[lo:hi], width))
    return segs


def floor_line_segments(spec: CourseSpec, lane_half: float = 0.9, line_w: float = 0.18,
                        thickness: float = 0.06, floor_d: float = -0.05
                        ) -> list[tuple[np.ndarray, np.ndarray]]:
    """本番シム同様の「床の上のシアン・レーンライン(2本)」を区間ごとに返す。

    実映像の青パスは空中リボンではなく、床面をゲートへ収束する2本の発光ラインなので、
    ribbon_segments(空中バンド)の置き換えとして使う。表示窓/明滅の制御は
    genesis_race_env の ribbon_entities(区間1エンティティ)機構をそのまま流用できるよう、
    左右2本を1メッシュに結合して返す(区間数・並びは ribbon_segments と一致)。

    区間の中心線を床へ投影(NED d=floor_d)し、水平接線の左右に ±lane_half だけずらした
    2本を各々薄い角柱(watertight)にして結合する。
    """
    per = RIBBON_SAMPLES_PER_SEG
    segs = []
    for i in range(1, spec.n_gates):
        lo = per * (i - 1)
        hi = min(per * i + 1, len(spec.ribbon_pts))
        pts = spec.ribbon_pts[lo:hi].astype(float).copy()
        pts[:, 2] = floor_d                                  # 床へ投影
        t = np.gradient(pts, axis=0)
        t[:, 2] = 0.0                                        # 水平接線
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        side = np.stack([-t[:, 1], t[:, 0], np.zeros(len(t))], axis=1)  # 水平90°
        vL, fL = ribbon_mesh(pts + side * lane_half, width=line_w, thickness=thickness)
        vR, fR = ribbon_mesh(pts - side * lane_half, width=line_w, thickness=thickness)
        verts = np.vstack([vL, vR])
        faces = np.vstack([np.asarray(fL), np.asarray(fR) + len(vL)])
        segs.append((verts, faces))
    return segs


def ribbon_mesh(ribbon_pts: np.ndarray, width: float = 0.8,
                thickness: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """リボン(発光ガイドパス)の薄い角柱メッシュを生成。NED座標のまま返す。

    帯の面の向きはパスの接線に追従し、水平区間では水平帯・急上昇区間では
    実映像同様の垂直な壁状バンドに自然になる(側方ベクトル=接線×鉛直、
    接線が鉛直に近づくと側方を水平面内に固定)。

    厚みゼロの帯だと直線区間で凸包体積=0となり、非固定エンティティの慣性が
    特異になって剛体ソルバの質量行列がNaN化する(バッチ全envの制約ソルバが
    死ぬ)ため、watertightな閉じた角柱として返す。
    returns (vertices, faces)
    """
    import trimesh

    pts = ribbon_pts
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    up = np.array([0.0, 0.0, -1.0])  # NEDの上
    side = np.cross(t, up)
    n = np.linalg.norm(side, axis=1, keepdims=True)
    # 接線がほぼ鉛直の区間: 側方をE軸に固定
    side = np.where(n > 0.2, side / np.maximum(n, 1e-9), np.array([0.0, 1.0, 0.0]))
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    nrm = np.cross(t, side)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9

    v0 = pts + side * (width / 2)
    v1 = pts - side * (width / 2)
    layer = np.empty((2 * len(pts), 3))
    layer[0::2] = v0
    layer[1::2] = v1
    off = np.repeat(nrm * (thickness / 2), 2, axis=0)
    M = len(pts)
    verts = np.vstack([layer - off, layer + off])  # [0:2M)=下面, [2M:4M)=上面

    faces = []
    B = 2 * M
    for i in range(M - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces += [[a, b, c], [b, d, c]]                          # 下面
        faces += [[B + a, B + c, B + b], [B + b, B + c, B + d]]  # 上面(逆巻き)
        faces += [[a, c, B + c], [a, B + c, B + a]]              # 側壁(v0側)
        faces += [[b, B + d, d], [b, B + b, B + d]]              # 側壁(v1側)
    e0, e1 = 2 * (M - 1), 2 * (M - 1) + 1
    faces += [[0, B + 1, 1], [0, B, B + 1]]                      # 端キャップ(始点)
    faces += [[e0, e1, B + e1], [e0, B + e1, B + e0]]            # 端キャップ(終点)

    # 頂点マージ+法線の向きを外向きに統一(慣性計算がwatertight判定に依存するため)
    tm = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64), process=True)
    trimesh.repair.fix_normals(tm)
    return np.asarray(tm.vertices), np.asarray(tm.faces, dtype=np.int64)
