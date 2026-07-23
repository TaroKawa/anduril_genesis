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


def sample_colors(rng: np.random.Generator, color_dr: bool) -> SceneColors:
    """実測Hue帯(ゲート: OpenCV H 0-25 ≒ 0-50°、パス: H 75-140 ≒ 150-280°)からサンプル。"""
    if color_dr:
        gate_h = rng.uniform(0.0, 25.0)
        if rng.random() < 0.3:
            gate_h = rng.uniform(340.0, 360.0)
        gate = colorsys.hsv_to_rgb((gate_h % 360) / 360.0, rng.uniform(0.85, 1.0), rng.uniform(0.85, 1.0))
        ribbon = colorsys.hsv_to_rgb(rng.uniform(185.0, 215.0) / 360.0, rng.uniform(0.8, 1.0), rng.uniform(0.7, 1.0))
        glow = colorsys.hsv_to_rgb(rng.uniform(36.0, 84.0) / 360.0, rng.uniform(0.7, 1.0), rng.uniform(0.8, 1.0))
        ambient = rng.uniform(0.08, 0.2)
    else:
        gate = (0.98, 0.24, 0.06)     # 実色 BGR #FA3C0F → RGB
        ribbon = (0.1, 0.85, 1.0)
        glow = (1.0, 0.8, 0.15)
        ambient = 0.12
    return SceneColors(gate_rgb=gate, ribbon_rgb=ribbon, glow_rgb=glow, ambient=ambient)


class SceneBuilder:
    """1つのgs.Sceneに静的コースジオメトリ+ドローンを構築する。"""

    def __init__(self, course: CourseSpec, rng: np.random.Generator,
                 color_dr: bool = False, clutter: bool = False):
        self.course = course
        self.rng = rng
        self.colors = sample_colors(rng, color_dr)
        self.clutter = clutter
        self.drone_entity = None
        self.static_entities = []

    def build_scene(self, scene, drone_cfg):
        import genesis as gs

        self._add_hall(scene, gs)
        self._add_pillars(scene, gs)
        self._add_ceiling_lights(scene, gs)
        self._add_gates(scene, gs)
        self._add_ribbon(scene, gs)
        if self.clutter:
            self._add_clutter(scene, gs)
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
        dark = (0.07, 0.07, 0.08)
        # 床(平面)
        self._static(scene, gs, gs.morphs.Plane(), (0.10, 0.10, 0.11))
        # 床ロードマーキング(薄い発光気味の帯、collision無し)
        n_marks = 8
        for i in range(n_marks):
            x = -L / 2 + (i + 0.5) * L / n_marks
            self._static(
                scene, gs,
                gs.morphs.Box(pos=(x, 0.0, 0.01), size=(L / n_marks - 4.0, 0.25, 0.02),
                              fixed=True, collision=False),
                None, emissive=(0.5, 0.5, 0.35),
            )
        # 壁4面(衝突あり・可視)
        t = 0.3
        for pos, size in [
            ((0.0, W / 2 + t / 2, H / 2), (L, t, H)),
            ((0.0, -W / 2 - t / 2, H / 2), (L, t, H)),
            ((L / 2 + t / 2, 0.0, H / 2), (t, W, H)),
            ((-L / 2 - t / 2, 0.0, H / 2), (t, W, H)),
        ]:
            self._static(scene, gs, gs.morphs.Box(pos=pos, size=size, fixed=True), (0.05, 0.05, 0.06))
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
                (0.13, 0.13, 0.14),
            )
            # "Station"サイン風の微発光バンド
            if self.rng.random() < 0.5:
                z = self.rng.uniform(4.0, hall.height - 2.0)
                self._static(
                    scene, gs,
                    gs.morphs.Box(pos=(x, y, z), size=(1.56, 1.56, 0.8), fixed=True, collision=False),
                    None, emissive=(0.35, 0.35, 0.38),
                )

    def _add_ceiling_lights(self, scene, gs):
        hall = self.course.hall
        z = hall.height - 0.25
        xs = np.arange(-hall.length / 2 + 8.0, hall.length / 2 - 4.0, 14.0)
        ys = np.arange(-hall.width / 2 + 6.0, hall.width / 2 - 3.0, 11.0)
        v = self.rng.uniform(0.7, 1.0)
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

    def _add_gates(self, scene, gs):
        c = self.colors
        self.glow_entities = []
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
        for (fv, ff), (rv, rf) in path_segments(self.course):
            # フィル: 半透明・淡発光(レール間を埋める)
            fill = _mesh_entity(fv, ff, gs.surfaces.Rough(
                color=(r * 0.15, g * 0.15, b * 0.15),
                emissive=(r * 0.5, g * 0.5, b * 0.5), opacity=0.3))
            # レール: 細く明るい縁
            rail = _mesh_entity(rv, rf, gs.surfaces.Rough(
                color=(r * 0.2, g * 0.2, b * 0.2), emissive=(r, g, b), opacity=0.95))
            self.ribbon_entities.append(fill)
            self.ribbon_rail_entities.append(rail)

    def _add_clutter(self, scene, gs):
        """駐機機体シルエット(箱の組合せ、暗灰色)。リボンから離れた床に配置。"""
        hall = self.course.hall
        for _ in range(int(self.rng.integers(3, 7))):
            for _try in range(20):
                n = self.rng.uniform(-hall.length / 2 + 10, hall.length / 2 - 10)
                e = self.rng.uniform(-hall.width / 2 + 8, hall.width / 2 - 8)
                d = np.linalg.norm(self.course.ribbon_pts[:, :2] - np.array([n, e]), axis=1).min()
                if d > 6.0:
                    break
            else:
                continue
            x, y = float(n), float(-e)
            deg = float(self.rng.uniform(0, 360))
            body = (0.16, 0.16, 0.17)
            self._static(scene, gs, gs.morphs.Box(pos=(x, y, 0.8), euler=(0, 0, deg),
                                                  size=(9.0, 1.6, 1.6), fixed=True), body)
            self._static(scene, gs, gs.morphs.Box(pos=(x, y, 0.6), euler=(0, 0, deg),
                                                  size=(2.5, 7.0, 0.35), fixed=True), body)
            self._static(scene, gs, gs.morphs.Box(pos=(x - 3.4 * np.cos(np.radians(deg)),
                                                       y - 3.4 * np.sin(np.radians(deg)), 1.4),
                                                  euler=(0, 0, deg), size=(1.2, 3.0, 1.4), fixed=True), body)

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
