import numpy as np
import pytest

from genesis_rl.course import CourseGenerator, ribbon_mesh, GATE_OUTER


@pytest.mark.parametrize("seed", range(0, 100, 7))
@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_course_invariants(seed, stage):
    spec = CourseGenerator(seed=seed, stage=stage).generate()
    assert spec.n_gates == (9 if stage == 0 else 19)  # スタート + 8(直線) / 18
    hall = spec.hall
    centers = np.stack([g.center_ned for g in spec.gates])
    # ホール内(壁マージン)
    assert (np.abs(centers[:, 0]) < hall.length / 2).all()
    assert (np.abs(centers[:, 1]) < hall.width / 2).all()
    # NEDのd(下向き正)は負=空中、天井より下
    assert (centers[:, 2] < -0.8).all()
    assert (centers[:, 2] > -(hall.height - 1.0)).all()
    # ゲート間距離 >= min_gap(stage1=近接コースは4.5m)
    min_gap = 4.5 if stage == 1 else 6.0
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= min_gap - 1e-6
    # 弧長が単調
    assert (np.diff(spec.gate_cum_arc) > 0).all()
    assert spec.total_arc > 50.0
    # 柱はリボンから2.5m以上
    if len(spec.pillars):
        dp = np.linalg.norm(spec.pillars[:, None, :] - spec.ribbon_pts[None, :, :2], axis=-1).min(axis=1)
        assert (dp > 2.5).all()


def test_stage1_gates_are_close():
    """近接ステージ(stage1)は隣接ゲート間隔が標準(stage2)より詰まっている。"""
    def mean_step(stage):
        vals = []
        for seed in range(6):
            c = np.stack([g.center_ned for g in CourseGenerator(seed=seed, stage=stage).generate().gates])
            vals.append(np.linalg.norm(np.diff(c, axis=0), axis=1).mean())
        return np.mean(vals)
    assert mean_step(1) < mean_step(2) - 1.5


def test_longitudinal_progression():
    """コースは始点(ホール手前)→終点(奥)へ縦方向に伸びる(Nがほぼ単調増加)。"""
    for seed in range(6):
        spec = CourseGenerator(seed=seed, stage=3).generate()
        n = np.stack([g.center_ned for g in spec.gates])[:, 0]
        assert (np.diff(n) > -1.5).all()          # 後戻りしない
        assert n[-1] - n[0] > 50.0                # 奥までしっかり進む


def test_gate_tilts_bounded():
    """ゲートは形を変えず少しだけ傾く(±12°以内)。スタートゲートは水平。"""
    spec = CourseGenerator(seed=1, stage=3).generate()
    assert spec.gates[0].pitch == 0.0 and spec.gates[0].roll == 0.0
    tilts = np.array([[g.pitch, g.roll] for g in spec.gates[1:]])
    assert (np.abs(tilts) <= np.radians(12) + 1e-9).all()
    assert np.abs(tilts).max() > 1e-4  # 実際に傾いている


def test_sharp_turns_exist():
    """最大90°級のターン(真横へ行くゲート)が生成される。"""
    max_turn = 0.0
    for seed in range(10):
        spec = CourseGenerator(seed=seed, stage=3).generate()
        yaws = np.array([g.yaw for g in spec.gates])
        d = np.abs((np.diff(yaws) + np.pi) % (2 * np.pi) - np.pi)
        max_turn = max(max_turn, float(d.max()))
    assert max_turn > np.radians(55)


def test_stage2_has_altitude_variation():
    diffs = []
    for seed in range(8):
        spec = CourseGenerator(seed=seed, stage=3).generate()
        z = np.stack([g.center_ned for g in spec.gates])[:, 2]
        diffs.append(z.max() - z.min())
    # フルレンジでは高低差(急上昇区間含む)があるコースが大半
    assert np.median(diffs) > 2.0


def test_ribbon_mesh_watertight():
    """リボンはwatertightな薄角柱(体積>0)。平面だと非固定エンティティの慣性が
    特異になり剛体ソルバがNaN化する(2026-07-22の修正)。"""
    import trimesh

    spec = CourseGenerator(seed=0, stage=3).generate()
    verts, faces = ribbon_mesh(spec.ribbon_pts, width=0.8)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    assert tm.is_watertight
    assert tm.volume > 0.01
    assert faces.max() < len(verts)


def test_deterministic():
    a = CourseGenerator(seed=5, stage=3).generate()
    b = CourseGenerator(seed=5, stage=3).generate()
    assert np.allclose(
        np.stack([g.center_ned for g in a.gates]), np.stack([g.center_ned for g in b.gates])
    )
