"""Genesisシーン構築: 格納庫・ゲート・青リボン・柱・天井ライト・クラッタ。

実トラックのスクリーンショットを再現:
  - 暗い格納庫、床にロードマーキング
  - オレンジ発光ゲート(白ロゴ/市松風の白マーキング付き)+ゲート真下の金色グロー
  - 空中を蛇行するシアン発光リボン(急上昇区間では垂直の壁状バンド)
  - "Station"柱、天井トラス+白発光ストリップ枠、駐機機体シルエット

色DRはシーン再構築ごと(バッチ環境は同一ジオメトリを共有するため)。
座標は構築時にNED→Genesis world(n=x, e=-y, d=-z)へ変換する。
"""

from __future__ import annotations

import colorsys
import tempfile
from dataclasses import dataclass

import numpy as np

from .course import (
    BAR_W,
    CourseSpec,
    GATE_DEPTH,
    GATE_INNER,
    GATE_OUTER,
    path_segments,
)


def ned2w(p) -> tuple:
    return (float(p[0]), float(-p[1]), float(-p[2]))


_T_FLIP = np.diag([1.0, -1.0, -1.0])

# per-envモードの装飾パラメータ(固定サイズ=剛体を再スケールできないため)
PE_GLOW_COL_H = 6.0   # 床から立てる光柱の固定高[m](ゲート高度差はxy位置のみで吸収)


def rot_ned_to_world(R_ned: np.ndarray) -> np.ndarray:
    """NED系の回転行列 → Genesis world系(x軸180°の相似変換)。"""
    return _T_FLIP @ R_ned @ _T_FLIP


def np_R_to_quat(R: np.ndarray) -> tuple:
    """回転行列 → クォータニオン wxyz。"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12)) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
        w, x, y, z = q
    return (float(w), float(x), float(y), float(z))


@dataclass
class SceneColors:
    gate_rgb: tuple
    ribbon_rgb: tuple
    glow_rgb: tuple
    ambient: float
    ribbon_fill_gain: float = 0.65   # リボンフィルの発光ゲイン(実機の「青く光る路面」の帯域)
    ribbon_fill_op: float = 0.4      # 同・不透明度


def sample_colors(rng: np.random.Generator, color_dr: bool) -> SceneColors:
    """実飛行フレーム(Spakona output/20260720_212416)の実測に基づく色。

    実測(85生フレーム): ゲート赤=RGB(235,100,95)のピンク寄り赤(ブルーム込み)、
    リボン=シアン(彩度高)、画素の42%がV<25の暗部、全体mean RGB(32,39,42)。
    DR時は実測色を中心に±の帯からサンプルする(視覚過適合の防止)。"""
    if color_dr:
        gate_h = rng.uniform(-12.0, 18.0) % 360   # 赤〜ピンク帯中心(実測)。稀にオレンジ寄り
        if rng.random() < 0.2:
            gate_h = rng.uniform(15.0, 30.0)
        gate = colorsys.hsv_to_rgb(gate_h / 360.0, rng.uniform(0.7, 1.0), rng.uniform(0.9, 1.0))
        ribbon = colorsys.hsv_to_rgb(rng.uniform(185.0, 215.0) / 360.0, rng.uniform(0.8, 1.0), rng.uniform(0.7, 1.0))
        glow = colorsys.hsv_to_rgb(rng.uniform(36.0, 84.0) / 360.0, rng.uniform(0.7, 1.0), rng.uniform(0.8, 1.0))
        ambient = rng.uniform(0.005, 0.04)         # 実機は暗环境(黒地に発光)。明るい絵は出さない
        # 実DCLは明部(V>=120)が画素の10%(リボン路面・天井灯・ゲートのブルーム)。
        # フィルは明るく(実測リボン帯 mean RGB(32,112,143)、max飽和)
        fill_gain = float(rng.uniform(0.7, 1.05))
        fill_op = float(rng.uniform(0.28, 0.5))
    else:
        gate = (1.0, 0.24, 0.22)      # 実機のネオン赤(彩度高)。白ロゴ/ハロが白飛び側を担う
        ribbon = (0.1, 0.85, 1.0)
        glow = (1.0, 0.8, 0.15)
        ambient = 0.02
        fill_gain, fill_op = 0.85, 0.38
    return SceneColors(gate_rgb=gate, ribbon_rgb=ribbon, glow_rgb=glow, ambient=ambient,
                       ribbon_fill_gain=fill_gain, ribbon_fill_op=fill_op)


class SceneBuilder:
    """1つのgs.Sceneに静的コースジオメトリ+ドローンを構築する。"""

    def __init__(self, course: CourseSpec, rng: np.random.Generator,
                 color_dr: bool = False, clutter: bool = False, per_env: bool = False):
        self.course = course
        self.rng = rng
        self.colors = sample_colors(rng, color_dr)
        self.clutter = clutter
        # per_env: ゲートを非固定・衝突なしの表示専用ボックスにし、envごとにset_pos/set_quatで
        # 別コースへ配置し直す(衝突は数値計算)。コース依存メッシュのリボン/柱/クラッタは省く。
        self.per_env = per_env
        self.drone_entity = None
        self.static_entities = []
        self.gate_bar_entities = []   # per_env時: (entity, gi, off_side, off_up) のリスト

    def build_scene(self, scene, drone_cfg):
        import genesis as gs

        self._add_hall(scene, gs)
        if not self.per_env:
            self._add_pillars(scene, gs)
        self._add_ceiling_lights(scene, gs)
        self._add_gates(scene, gs)
        if not self.per_env:
            self._add_ribbon(scene, gs)
            if self.clutter:
                self._add_clutter(scene, gs)
        else:
            # per-env: コース依存の装飾(柱/クラッタ/ポール/オーブ/リボン)を固定個数プールで
            # 生成し、genesis_race_env が各envのコース配置へ set_pos/set_quat する。
            self._add_world_pools_per_env(scene, gs)
        self._add_drone(scene, gs, drone_cfg)
        return self.drone_entity

    # --- 各要素 ---

    def _static(self, scene, gs, morph, color, emissive=None):
        surf = gs.surfaces.Emission(color=tuple(emissive)) if emissive is not None \
            else gs.surfaces.Rough(color=tuple(color))
        ent = scene.add_entity(morph, surface=surf)
        self.static_entities.append(ent)
        return ent

    def _add_hall(self, scene, gs):
        hall = self.course.hall
        L, W, H = hall.length, hall.width, hall.height
        # 床(平面): 実測の床領域は RGB≈(27,49,58) と青被りの暗灰(リボン/天井灯の照り返し)。
        # ラスタライザにGIは無いので、床自体に微弱な青系の自発光を持たせて照り返しを近似する。
        # 値は露出ゲイン(render.exposure≈2.6)込みで実測に合うよう逆算した帯域。
        f = float(self.rng.uniform(0.005, 0.014))
        e = float(self.rng.uniform(0.007, 0.017))
        ent = scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(
            color=(f, f * 1.3, f * 1.6), emissive=(e * 0.5, e * 1.0, e * 1.25)))
        self.static_entities.append(ent)
        # 床ロードマーキング: 実映像は縦に走る細い明灰色ライン(発光弱)。連続ラインを2-3本
        n_lines = int(self.rng.integers(2, 4))
        for i in range(n_lines):
            y = float(self.rng.uniform(-W / 2 + 6.0, W / 2 - 6.0))
            v = float(self.rng.uniform(0.09, 0.18))
            self._static(
                scene, gs,
                gs.morphs.Box(pos=(0.0, y, 0.01), size=(L - 6.0, 0.18, 0.02),
                              fixed=True, collision=False),
                None, emissive=(v, v, v),
            )
        # 横切りライン(駐機区画風、まばら)
        for x in np.arange(-L / 2 + 10.0, L / 2 - 6.0, float(self.rng.uniform(16.0, 26.0))):
            v = float(self.rng.uniform(0.06, 0.13))
            self._static(
                scene, gs,
                gs.morphs.Box(pos=(float(x), 0.0, 0.01), size=(0.15, W - 8.0, 0.02),
                              fixed=True, collision=False),
                None, emissive=(v, v, v),
            )
        # 壁4面(衝突あり・可視): ほぼ黒
        t = 0.3
        for pos, size in [
            ((0.0, W / 2 + t / 2, H / 2), (L, t, H)),
            ((0.0, -W / 2 - t / 2, H / 2), (L, t, H)),
            ((L / 2 + t / 2, 0.0, H / 2), (t, W, H)),
            ((-L / 2 - t / 2, 0.0, H / 2), (t, W, H)),
        ]:
            self._static(scene, gs, gs.morphs.Box(pos=pos, size=size, fixed=True), (0.02, 0.02, 0.025))
        # 天井は衝突のみ・非表示(平行光を屋内へ通す。見た目の天井はライトストリップが担う)
        ent = scene.add_entity(gs.morphs.Box(pos=(0.0, 0.0, H + t / 2), size=(L, W, t),
                                             fixed=True, visualization=False))
        self.static_entities.append(ent)

    def _add_pillars(self, scene, gs):
        hall = self.course.hall
        for (n, e) in self.course.pillars:
            x, y = float(n), float(-e)
            self._static(
                scene, gs,
                gs.morphs.Box(pos=(x, y, hall.height / 2), size=(1.5, 1.5, hall.height), fixed=True),
                (0.05, 0.05, 0.055),   # 実映像の柱はほぼ黒
            )
            # "Station XX"サイン: 柱の面に縦長の白い淡発光ストリップ(縦書きテキストの輝度分布を近似)
            if self.rng.random() < 0.75:
                z = float(self.rng.uniform(3.5, hall.height - 2.5))
                h = float(self.rng.uniform(1.8, 3.2))
                v = float(self.rng.uniform(0.3, 0.55))
                sides = [(0.78, 0.0, 0.06, 0.5), (0.0, 0.78, 0.5, 0.06)]
                dx, dy, sx, sy = sides[int(self.rng.integers(0, 2))]
                self._static(
                    scene, gs,
                    gs.morphs.Box(pos=(x + dx, y + dy, z), size=(max(sx, 0.06), max(sy, 0.06), h),
                                  fixed=True, collision=False),
                    None, emissive=(v, v, v * 1.02),
                )

    def _add_ceiling_lights(self, scene, gs):
        hall = self.course.hall
        z = hall.height - 0.25
        xs = np.arange(-hall.length / 2 + 8.0, hall.length / 2 - 4.0, 14.0)
        ys = np.arange(-hall.width / 2 + 6.0, hall.width / 2 - 3.0, 11.0)
        v = self.rng.uniform(0.85, 1.0)   # 実機の天井灯は白飛びに近い明るさ
        for x in xs:
            for y in ys:
                # 白発光の矩形ストリップ枠(実映像の天井ライトグリッド)
                for dx, dy, sx, sy in [(0, 2.0, 4.0, 0.25), (0, -2.0, 4.0, 0.25),
                                       (2.0, 0, 0.25, 4.0), (-2.0, 0, 0.25, 4.0)]:
                    self._static(
                        scene, gs,
                        gs.morphs.Box(pos=(x + dx, y + dy, z), size=(sx, sy, 0.1),
                                      fixed=True, collision=False),
                        None, emissive=(v, v, v),
                    )
        # 天井トラス: 実映像は黒天井に細い白ラインが縦横に走る(骨組みの照り返し)。
        # 長手方向+横方向の細い淡発光ストリップで近似(本数・輝度はDR)
        tz = hall.height - 0.6
        tv = float(self.rng.uniform(0.08, 0.18))
        for y in np.arange(-hall.width / 2 + 4.0, hall.width / 2 - 2.0, float(self.rng.uniform(7.0, 12.0))):
            self._static(
                scene, gs,
                gs.morphs.Box(pos=(0.0, float(y), tz), size=(hall.length - 4.0, 0.08, 0.08),
                              fixed=True, collision=False),
                None, emissive=(tv, tv, tv),
            )
        for x in np.arange(-hall.length / 2 + 6.0, hall.length / 2 - 4.0, float(self.rng.uniform(12.0, 18.0))):
            self._static(
                scene, gs,
                gs.morphs.Box(pos=(float(x), 0.0, tz), size=(0.08, hall.width - 4.0, 0.08),
                              fixed=True, collision=False),
                None, emissive=(tv, tv, tv),
            )

    def _add_gates(self, scene, gs):
        if self.per_env:
            self._add_gates_per_env(scene, gs)
            return
        c = self.colors
        self.glow_entities = []
        self.glow_col_entities = []
        for gi, gate in enumerate(self.course.gates):
            cw = np.array(ned2w(gate.center_ned))
            R_w = rot_ned_to_world(gate.rotation_ned())  # 列: x=法線, y=±側方, z=面内上方
            quat = np_R_to_quat(R_w)

            def place(off_side, off_up, size_side, size_up, collision=True):
                """ゲート面内(side=横, up=縦)のオフセット → world配置(傾き込み)。"""
                pos = cw + R_w @ np.array([0.0, off_side, off_up])
                return gs.morphs.Box(pos=tuple(pos), quat=quat,
                                     size=(GATE_DEPTH, size_side, size_up),
                                     fixed=True, collision=collision)

            half = (GATE_INNER + BAR_W) / 2  # バー中心オフセット 1.05m
            emis = c.gate_rgb
            # 左右バー(縦 2.7m)+ 上下バー(横 1.5m)
            self._static(scene, gs, place(+half, 0.0, BAR_W, GATE_OUTER), None, emissive=emis)
            self._static(scene, gs, place(-half, 0.0, BAR_W, GATE_OUTER), None, emissive=emis)
            self._static(scene, gs, place(0.0, +half, GATE_INNER, BAR_W), None, emissive=emis)
            self._static(scene, gs, place(0.0, -half, GATE_INNER, BAR_W), None, emissive=emis)

            # 発光ハロー(疑似ブルーム): 実映像のゲートは強いブルームで枠の周囲が滲む。
            # ラスタライザにブルームは無いので、枠より一回り大きい半透明発光ボックスで近似
            if self.rng.random() < 0.9:
                op = float(self.rng.uniform(0.08, 0.18))
                halo_col = tuple(min(1.0, v * 0.95) for v in emis)
                for off_s, off_u, ss, su in [(+half, 0.0, BAR_W * 2.0, GATE_OUTER + 0.15),
                                             (-half, 0.0, BAR_W * 2.0, GATE_OUTER + 0.15),
                                             (0.0, +half, GATE_INNER + 0.15, BAR_W * 2.0),
                                             (0.0, -half, GATE_INNER + 0.15, BAR_W * 2.0)]:
                    pos = cw + R_w @ np.array([0.0, off_s, off_u])
                    m = gs.morphs.Box(pos=tuple(pos), quat=quat,
                                      size=(GATE_DEPTH * 0.4, ss, su),
                                      fixed=True, collision=False)
                    ent = scene.add_entity(m, surface=gs.surfaces.Rough(
                        color=tuple(v * 0.2 for v in halo_col), emissive=halo_col, opacity=op))
                    self.static_entities.append(ent)

            # 白ロゴバンド("AI-GP"風): 上バーの中央に白発光の横帯(実ゲートのロゴ輝度を近似)
            if self.rng.random() < 0.9:
                lw = float(self.rng.uniform(0.7, 1.2))
                pos = cw + R_w @ np.array([0.0, 0.0, half])
                m = gs.morphs.Box(pos=tuple(pos), quat=quat,
                                  size=(GATE_DEPTH + 0.02, lw, 0.16),
                                  fixed=True, collision=False)
                self._static(scene, gs, m, None, emissive=(0.95, 0.95, 0.95))

            # 白ロゴ/市松風マーキング(バー面上の小さな白発光パッチ、YOLOX偽検出源の再現)
            n_marks = int(self.rng.integers(2, 5))
            for _ in range(n_marks):
                side = float(self.rng.uniform(-1.2, 1.2))
                up = float(self.rng.choice([-half, half])) if abs(side) < GATE_INNER / 2 \
                    else float(self.rng.uniform(-1.2, 1.2))
                w = float(self.rng.uniform(0.15, 0.5))
                pos = cw + R_w @ np.array([0.0, side, up])
                m = gs.morphs.Box(pos=tuple(pos), quat=quat,
                                  size=(GATE_DEPTH + 0.02, w, 0.18),
                                  fixed=True, collision=False)
                self._static(scene, gs, m, None, emissive=(0.95, 0.95, 0.95))

            # ゲート脇のポール+発光球(実映像の信号灯: 緑ランプ / ピンクのグロー球)
            for sgn in (+1.0, -1.0):
                if self.rng.random() < 0.75:
                    dist = float(self.rng.uniform(2.6, 4.5))
                    base = cw + R_w @ np.array([0.0, sgn * dist, 0.0])
                    ph = float(self.rng.uniform(1.6, 2.6))
                    self._static(scene, gs,
                                 gs.morphs.Box(pos=(float(base[0]), float(base[1]), ph / 2),
                                               size=(0.07, 0.07, ph), fixed=True, collision=False),
                                 (0.04, 0.04, 0.045))
                    orb = [(1.0, 0.55, 0.62), (0.45, 1.0, 0.5)][int(self.rng.integers(0, 2))]
                    r_orb = float(self.rng.uniform(0.10, 0.18))
                    ent = scene.add_entity(
                        gs.morphs.Sphere(pos=(float(base[0]), float(base[1]), ph + r_orb), radius=r_orb,
                                         fixed=True, collision=False),
                        surface=gs.surfaces.Emission(color=orb))
                    self.static_entities.append(ent)
                    # 球のハロー(半透明の大きい球)
                    ent = scene.add_entity(
                        gs.morphs.Sphere(pos=(float(base[0]), float(base[1]), ph + r_orb),
                                         radius=r_orb * 2.4, fixed=True, collision=False),
                        surface=gs.surfaces.Rough(color=tuple(v * 0.2 for v in orb),
                                                  emissive=orb, opacity=0.25))
                    self.static_entities.append(ent)

            # ゲート直前の床の黄色グロー(実映像のゲート手前の黄色い帯)。ゲートの向き
            # (法線の水平成分)に沿ってレーン状に伸ばし、収束する青ラインの終端＝ゲート
            # 位置を強調する。「次に行くべきゲート」だけ点灯(リボン同様 非固定+重力補償)。
            nrm_w = R_w[:, 0]                                  # ゲート法線(world)
            ang = float(np.degrees(np.arctan2(nrm_w[1], nrm_w[0])))
            glow = scene.add_entity(
                gs.morphs.Box(pos=(cw[0], cw[1], 0.02), euler=(0.0, 0.0, ang),
                              size=(4.0, 0.7, 0.02), fixed=False, collision=False),
                material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
                surface=gs.surfaces.Emission(color=tuple(c.glow_rgb)),
            )
            self.glow_entities.append(glow)
            # 床→ゲート下端への光柱(実映像のアクティブゲート直下に立つ黄色い光の柱)。
            # glowと同じ per-env 表示制御(genesis_race_env._update_glow が同期移動)
            hcol = max(float(cw[2]) - 1.2, 0.5)
            col_ent = scene.add_entity(
                gs.morphs.Box(pos=(cw[0], cw[1], hcol / 2), euler=(0.0, 0.0, ang),
                              size=(0.5, 0.5, hcol), fixed=False, collision=False),
                material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
                surface=gs.surfaces.Rough(color=tuple(v * 0.2 for v in c.glow_rgb),
                                          emissive=tuple(c.glow_rgb), opacity=0.4),
            )
            self.glow_col_entities.append(col_ent)

    # ゲート4バーの面内オフセット(side,up)とサイズ(side,up)。全ゲート共通。
    GATE_BARS = None  # 遅延初期化(モジュール定数から)

    # ---------- per-envメッシュ統合ヘルパー ----------
    # 装飾を非固定boxで作ると1個=1剛体(6DOF)になり、n_dofsが膨張してGenesisの
    # int32 Jacobian上限(len_constraints_×n_dofs×n_envs<2^31)でnum_envsが激減する。
    # 複数box/sphereを1メッシュ(=1剛体)へ統合してDOFを~3倍削り、num_envsを保つ。

    @staticmethod
    def _combine_mesh(boxes=(), spheres=()):
        """box/sphere群を1つのtrimeshへ統合。boxes=[(center3,size3,R|None)], spheres=[(center3,r)]。"""
        import trimesh

        parts = []
        for (center, size, R) in boxes:
            T = np.eye(4)
            if R is not None:
                T[:3, :3] = R
            T[:3, 3] = center
            parts.append(trimesh.creation.box(extents=size, transform=T))
        for (center, r) in spheres:
            s = trimesh.creation.icosphere(subdivisions=1, radius=r)
            s.apply_translation(center)
            parts.append(s)
        return trimesh.util.concatenate(parts)

    def _mesh_file(self, tri) -> str:
        fh = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        tri.export(fh.name)
        return fh.name

    def _mesh_ent(self, scene, gs, obj_file, surface):
        """統合メッシュ(ローカル座標)から非固定・衝突なしの表示エンティティを作る。"""
        return scene.add_entity(
            gs.morphs.Mesh(file=obj_file, fixed=False, collision=False,
                           decimate=False, convexify=False),
            material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
            surface=surface)

    def _add_gates_per_env(self, scene, gs):
        """per-envモード: ゲートを非固定・衝突なしの表示専用ボックスで作る。

        ビルド時は course(=pool[0])の配置に置くが、genesis_race_env が build 後に
        envごとに set_pos/set_quat で各envのコースへ移動する。衝突は数値計算に切替える
        ため collision=False。gravity_compensation=1.0 で自由落下しない(リボン/グロー同様)。
        """
        c = self.colors
        self.glow_entities = []
        self.glow_col_entities = []
        half = (GATE_INNER + BAR_W) / 2  # 1.05m
        bars = [
            (+half, 0.0, BAR_W, GATE_OUTER),
            (-half, 0.0, BAR_W, GATE_OUTER),
            (0.0, +half, GATE_INNER, BAR_W),
            (0.0, -half, GATE_INNER, BAR_W),
        ]
        emis = c.gate_rgb
        halo_col = tuple(min(1.0, v * 0.95) for v in emis)
        # マークは実映像のYOLOX偽検出源。per-envはメッシュ統合で個数固定=全ゲート共通配置。
        mark_offsets = [(0.0, 0.6, half), (0.0, -0.5, half), (0.0, 0.0, -half)]
        # ゲートローカル座標(x=法線/奥行, y=側方, z=上方)で box を並べ、3メッシュへ統合:
        #   枠(4バー・赤発光) / ハロー(4・半透明赤) / 白(ロゴ+マーク)。各ゲート1エンティティ×3。
        frame_boxes = [((0.0, os, ou), (GATE_DEPTH, ss, su), None) for (os, ou, ss, su) in bars]
        halo_boxes = [((0.0, os, ou), (GATE_DEPTH * 0.4, ss * 2.0, su + 0.15), None)
                      for (os, ou, ss, su) in bars]
        white_boxes = [((0.0, 0.0, half), (GATE_DEPTH + 0.02, 1.0, 0.16), None)]
        white_boxes += [((mo[0], mo[1], mo[2]), (GATE_DEPTH + 0.02, 0.35, 0.18), None)
                        for mo in mark_offsets]
        frame_f = self._mesh_file(self._combine_mesh(boxes=frame_boxes))
        halo_f = self._mesh_file(self._combine_mesh(boxes=halo_boxes))
        white_f = self._mesh_file(self._combine_mesh(boxes=white_boxes))
        frame_surf = gs.surfaces.Emission(color=tuple(emis))
        halo_surf = gs.surfaces.Rough(color=tuple(v * 0.2 for v in halo_col),
                                      emissive=halo_col, opacity=0.14)
        white_surf = gs.surfaces.Emission(color=(0.95, 0.95, 0.95))
        for gi, gate in enumerate(self.course.gates):
            cw = np.array(ned2w(gate.center_ned))
            # 3メッシュはゲートローカル座標。off=(0,0,0)で登録し、_place_gates_per_env が
            # ゲート中心へ set_pos・ゲート向き(R_w)へ set_quat する(バーと同じ配置経路)。
            for fpath, surf in ((frame_f, frame_surf), (halo_f, halo_surf), (white_f, white_surf)):
                ent = self._mesh_ent(scene, gs, fpath, surf)
                self.gate_bar_entities.append((ent, gi, (0.0, 0.0, 0.0)))
            # 床グロー+光柱(非固定box)。build時はcourse[0]配置、_place_glow_per_envがenv毎に移動。
            self.glow_entities.append(scene.add_entity(
                gs.morphs.Box(pos=(cw[0], cw[1], 0.02), size=(4.0, 0.7, 0.02),
                              fixed=False, collision=False),
                material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
                surface=gs.surfaces.Emission(color=tuple(c.glow_rgb)),
            ))
            self.glow_col_entities.append(scene.add_entity(
                gs.morphs.Box(pos=(cw[0], cw[1], PE_GLOW_COL_H / 2),
                              size=(0.5, 0.5, PE_GLOW_COL_H), fixed=False, collision=False),
                material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
                surface=gs.surfaces.Rough(color=tuple(v * 0.2 for v in c.glow_rgb),
                                          emissive=tuple(c.glow_rgb), opacity=0.4),
            ))

    def _add_ribbon(self, scene, gs):
        """青パス=ゲート内側(中心より下)を貫く帯(半透明フィル＋細い縁レール2本)。

        実映像の青パスは床でも空中リボンでもなく、ゲート開口の内側下寄りを通り次ゲートへ
        続く3Dパス。course.path_segments が区間毎に (fill, rails) を返すので、
          - fill  : rail間を埋める薄い帯 → 半透明・淡発光(ribbon_entities)
          - rails : ±rail_half の細い2本 → 明るい発光(ribbon_rail_entities)
        の2エンティティを積む。どちらも非固定+gravity_compensation=1.0 で、表示/非表示は
        genesis_race_env._update_ribbon が ribbon_entities と ribbon_rail_entities を同期
        移動して切り替える(区間並びは従来と一致)。
        self.ribbon_entities[i] = ゲートi+1へ向かう区間のフィル。
        """
        import trimesh

        def _mesh_entity(verts_ned, faces, surface):
            verts_w = verts_ned.copy()
            verts_w[:, 1] *= -1.0
            verts_w[:, 2] *= -1.0
            mesh = trimesh.Trimesh(vertices=verts_w, faces=faces, process=False)
            fh = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
            mesh.export(fh.name)
            return scene.add_entity(
                gs.morphs.Mesh(file=fh.name, fixed=False, collision=False,
                               decimate=False, convexify=False),
                material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
                surface=surface,
            )

        self.ribbon_entities = []
        self.ribbon_rail_entities = []
        r, g, b = self.colors.ribbon_rgb
        # 実映像ではリボンが最も目立つ要素(画素の8%がシアン帯)。フィルの発光・不透明度は
        # 実機の「青く光る路面」帯域をDRで包含する(sample_colors: gain 0.45-0.9 / op 0.3-0.55)
        fg, fo = self.colors.ribbon_fill_gain, self.colors.ribbon_fill_op
        # フィルは白を混ぜた薄い水色(実機の路面はガラス様に透け、白っぽく光る)
        wr, wg, wb = (r * 0.6 + 0.4), (g * 0.6 + 0.4), (b * 0.6 + 0.4)
        for (fv, ff), (rv, rf) in path_segments(self.course, rail_half=0.7):
            fill = _mesh_entity(fv, ff, gs.surfaces.Rough(
                color=(wr * 0.15, wg * 0.15, wb * 0.15),
                emissive=(wr * fg, wg * fg, wb * fg), opacity=fo))
            # レール: 細く明るい縁
            rail = _mesh_entity(rv, rf, gs.surfaces.Rough(
                color=(r * 0.2, g * 0.2, b * 0.2), emissive=(r, g, b), opacity=0.95))
            self.ribbon_entities.append(fill)
            self.ribbon_rail_entities.append(rail)

    def _add_clutter(self, scene, gs):
        """駐機機体シルエット(箱の組合せ)。実映像ではコース脇のすぐ近くに中灰色の
        戦闘機が並ぶため、配置をコース寄り(d>4m)・数を多め・色を中灰にする。"""
        hall = self.course.hall
        for _ in range(int(self.rng.integers(4, 9))):
            for _try in range(20):
                n = self.rng.uniform(-hall.length / 2 + 10, hall.length / 2 - 10)
                e = self.rng.uniform(-hall.width / 2 + 6, hall.width / 2 - 6)
                d = np.linalg.norm(self.course.ribbon_pts[:, :2] - np.array([n, e]), axis=1).min()
                if d > 4.0:
                    break
            else:
                continue
            x, y = float(n), float(-e)
            deg = float(self.rng.uniform(0, 360))
            g = float(self.rng.uniform(0.1, 0.22))   # 実映像の機体は黒背景に浮く中灰
            body = (g, g, g * 1.05)
            self._static(scene, gs, gs.morphs.Box(pos=(x, y, 0.8), euler=(0, 0, deg),
                                                  size=(9.0, 1.6, 1.6), fixed=True), body)
            self._static(scene, gs, gs.morphs.Box(pos=(x, y, 0.6), euler=(0, 0, deg),
                                                  size=(2.5, 7.0, 0.35), fixed=True), body)
            # 垂直尾翼(実機シルエットの特徴)
            self._static(scene, gs, gs.morphs.Box(pos=(x - 3.4 * np.cos(np.radians(deg)),
                                                       y - 3.4 * np.sin(np.radians(deg)), 1.6),
                                                  euler=(0, 0, deg), size=(1.6, 0.25, 1.8), fixed=True), body)
            self._static(scene, gs, gs.morphs.Box(pos=(x - 3.0 * np.cos(np.radians(deg)),
                                                       y - 3.0 * np.sin(np.radians(deg)), 1.1),
                                                  euler=(0, 0, deg), size=(1.2, 3.0, 0.9), fixed=True), body)

    # ---------- per-env ワールド配置プール(柱/クラッタ/ポール/リボン) ----------
    # コース依存の装飾を「固定個数の非固定box/sphereプール」として作り、genesis_race_env が
    # 各envのコース(_pool_specs)から計算した world pose へ set_pos/set_quat する。未使用スロットは
    # 床下(-80m)へ沈める。剛体は再スケール不可のためサイズは固定=全コース共通。

    @staticmethod
    def _quat_z(rad: float) -> tuple:
        return (float(np.cos(rad / 2)), 0.0, 0.0, float(np.sin(rad / 2)))

    def _pool(self, ents, pos_c, quat_c, valid_c):
        self.pe_pools.append({
            "ents": ents,
            "pos_c": np.asarray(pos_c, np.float32),
            "quat_c": np.asarray(quat_c, np.float32),
            "valid_c": np.asarray(valid_c, bool),
        })

    def _add_world_pools_per_env(self, scene, gs):
        specs = getattr(self, "_pool_specs", None) or [self.course]
        K = len(specs)
        self.pe_pools = []
        self._pool_pillars(scene, gs, specs, K)
        if self.clutter:
            self._pool_clutter(scene, gs, specs, K)
        self._pool_poles(scene, gs, specs, K)
        self._pool_ribbon(scene, gs, specs, K)

    def _pool_pillars(self, scene, gs, specs, K):
        hall = self.course.hall
        PMAX = 20   # DOF上限: コース毎にリボンへ近い順で最大20本(残りは視界外の壁際で寄与小)
        # 各コースのリボンに近い順に上位PMAX本を選ぶ
        sel = []
        for sp in specs:
            pil = np.asarray(sp.pillars, float)
            if len(pil) > PMAX:
                d = np.linalg.norm(pil[:, None, :] - sp.ribbon_pts[None, :, :2], axis=-1).min(axis=1)
                pil = pil[np.argsort(d)[:PMAX]]
            sel.append(pil)
        n_max = max((len(p) for p in sel), default=0)
        if n_max == 0:
            return
        pos_c = np.zeros((K, n_max, 3), np.float32)
        quat_c = np.tile(np.array([1.0, 0, 0, 0], np.float32), (K, n_max, 1))
        valid_c = np.zeros((K, n_max), bool)
        for c, pil in enumerate(sel):
            for i, (n, e) in enumerate(pil):
                pos_c[c, i] = (float(n), float(-e), hall.height / 2)
                valid_c[c, i] = True
        ents = [scene.add_entity(
            gs.morphs.Box(pos=(0, 0, -100), size=(1.5, 1.5, hall.height),
                          fixed=False, collision=False),
            material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
            surface=gs.surfaces.Rough(color=(0.05, 0.05, 0.055))) for _ in range(n_max)]
        self._pool(ents, pos_c, quat_c, valid_c)

    def _pool_clutter(self, scene, gs, specs, K):
        hall = self.course.hall
        NA = 8  # 最大駐機機体数(非per-envのrng.integers(4,9)上限)
        # 機体シルエット=4ボックスを機体ローカル座標で1メッシュへ統合(1機=1剛体)。
        # 機体poseは (x,y,0) + z回転。z高はメッシュ内に保持(z回転で不変)。
        ac_boxes = [
            ((0.0, 0.0, 0.8), (9.0, 1.6, 1.6), None),      # 胴体
            ((0.0, 0.0, 0.6), (2.5, 7.0, 0.35), None),     # 主翼
            ((-3.4, 0.0, 1.6), (1.6, 0.25, 1.8), None),    # 垂直尾翼
            ((-3.0, 0.0, 1.1), (1.2, 3.0, 0.9), None),     # 水平尾翼
        ]
        ac_file = self._mesh_file(self._combine_mesh(boxes=ac_boxes))
        pos_c = np.zeros((K, NA, 3), np.float32)
        quat_c = np.tile(np.array([1.0, 0, 0, 0], np.float32), (K, NA, 1))
        valid_c = np.zeros((K, NA), bool)
        for c, sp in enumerate(specs):
            rng = np.random.default_rng(int(sp.seed) + 4242)
            for a in range(NA):
                for _try in range(20):
                    n = rng.uniform(-hall.length / 2 + 10, hall.length / 2 - 10)
                    e = rng.uniform(-hall.width / 2 + 6, hall.width / 2 - 6)
                    d = np.linalg.norm(sp.ribbon_pts[:, :2] - np.array([n, e]), axis=1).min()
                    if d > 4.0:
                        rad = np.radians(float(rng.uniform(0, 360)))
                        pos_c[c, a] = (float(n), float(-e), 0.0)
                        quat_c[c, a] = self._quat_z(rad)
                        valid_c[c, a] = True
                        break
        g = 0.16
        ents = [self._mesh_ent(scene, gs, ac_file,
                               gs.surfaces.Rough(color=(g, g, g * 1.05))) for _ in range(NA)]
        self._pool(ents, pos_c, quat_c, valid_c)

    def _pool_poles(self, scene, gs, specs, K):
        # ゲート脇の信号ポール+発光球。DOF節約のため1ゲート1本(左右交互)、ハロー球は省く。
        G = self.course.n_gates
        n = G
        dist, ph, r_orb = 3.2, 2.1, 0.16
        pole_pos = np.zeros((K, n, 3), np.float32)
        orb_pos = np.zeros((K, n, 3), np.float32)
        ident = np.tile(np.array([1.0, 0, 0, 0], np.float32), (K, n, 1))
        valid = np.zeros((K, n), bool)
        for c, sp in enumerate(specs):
            for gi, gate in enumerate(sp.gates):
                sgn = 1.0 if gi % 2 == 0 else -1.0
                base = gate.center_ned + gate.rotation_ned() @ np.array([0.0, sgn * dist, 0.0])
                w = ned2w(base)
                pole_pos[c, gi] = (w[0], w[1], ph / 2)
                orb_pos[c, gi] = (w[0], w[1], ph + r_orb)
                valid[c, gi] = True
        pole_ents = [scene.add_entity(
            gs.morphs.Box(pos=(0, 0, -100), size=(0.07, 0.07, ph), fixed=False, collision=False),
            material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
            surface=gs.surfaces.Rough(color=(0.04, 0.04, 0.045))) for _ in range(n)]
        self._pool(pole_ents, pole_pos, ident, valid)
        cols = [(1.0, 0.55, 0.62) if i % 2 == 0 else (0.45, 1.0, 0.5) for i in range(n)]
        orb_ents = [scene.add_entity(
            gs.morphs.Sphere(pos=(0, 0, -100), radius=r_orb, fixed=False, collision=False),
            material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
            surface=gs.surfaces.Emission(color=cols[i])) for i in range(n)]
        self._pool(orb_ents, orb_pos, ident, valid)

    def _pool_ribbon(self, scene, gs, specs, K):
        from .course import ribbon_dashes

        # DOF節約のため粗い間隔(5.0m)。ダッシュ長4.8mでほぼ連続に見える(緩曲線では逸脱小)。
        dashes = [ribbon_dashes(sp, spacing=5.0) for sp in specs]
        n_max = max((len(d[0]) for d in dashes), default=0)
        if n_max == 0:
            return
        pos_c = np.zeros((K, n_max, 3), np.float32)
        quat_c = np.tile(np.array([1.0, 0, 0, 0], np.float32), (K, n_max, 1))
        valid_c = np.zeros((K, n_max), bool)
        up = np.array([0.0, 0.0, -1.0])   # NEDの上
        for c, (pos, tan) in enumerate(dashes):
            for i in range(len(pos)):
                t = tan[i]
                side = np.cross(t, up)
                ns = np.linalg.norm(side)
                side = side / ns if ns > 0.2 else np.array([0.0, 1.0, 0.0])
                nrm = np.cross(t, side)
                nrm /= np.linalg.norm(nrm) + 1e-9
                R_ned = np.stack([t, side, nrm], axis=1)          # 列=(接線,側方,法線)
                quat_c[c, i] = np_R_to_quat(rot_ned_to_world(R_ned))
                pos_c[c, i] = ned2w(pos[i])
                valid_c[c, i] = True
        r, g, b = self.colors.ribbon_rgb
        ents = [scene.add_entity(
            gs.morphs.Box(pos=(0, 0, -100), size=(3.4, 1.3, 0.06), fixed=False, collision=False),
            material=gs.materials.Rigid(rho=1.0, gravity_compensation=1.0),
            surface=gs.surfaces.Rough(color=(r * 0.2, g * 0.2, b * 0.2),
                                      emissive=(r, g, b), opacity=0.9)) for _ in range(n_max)]
        self._pool(ents, pos_c, quat_c, valid_c)

    def _add_drone(self, scene, gs, drone_cfg):
        # Box剛体(280x280x160mm)。密度で質量を合わせる(力は比力×質量で印加するので
        # 並進はmassフリー、回転はDRで吸収)。
        vol = 0.28 * 0.28 * 0.16
        rho = drone_cfg.mass / vol
        # 寸法は仕様§3.6の280x280x160mm(ゲート開口1.5mに対する比もspec通り)。
        # 暗色だと映像で見失うため明るい色にする(FPVは箱の内側=背面カリングで映らない)
        self.drone_entity = scene.add_entity(
            gs.morphs.Box(pos=(0.0, 0.0, 1.8), size=(0.28, 0.28, 0.16), fixed=False),
            material=gs.materials.Rigid(rho=rho),
            surface=gs.surfaces.Rough(color=(0.95, 0.95, 1.0)),
        )
        return self.drone_entity
