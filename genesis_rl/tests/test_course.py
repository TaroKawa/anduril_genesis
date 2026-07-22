import numpy as np
import pytest

from genesis_rl.course import CourseGenerator, ribbon_mesh, GATE_OUTER


@pytest.mark.parametrize("seed", range(0, 100, 7))
@pytest.mark.parametrize("stage", [0, 1, 2])
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
    # ゲート間距離 >= 6m
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 6.0 - 1e-6
    # 弧長が単調
    assert (np.diff(spec.gate_cum_arc) > 0).all()
    assert spec.total_arc > 50.0
    # 柱はリボンから2.5m以上
    if len(spec.pillars):
        dp = np.linalg.norm(spec.pillars[:, None, :] - spec.ribbon_pts[None, :, :2], axis=-1).min(axis=1)
        assert (dp > 2.5).all()


def test_stage2_has_altitude_variation():
    diffs = []
    for seed in range(8):
        spec = CourseGenerator(seed=seed, stage=2).generate()
        z = np.stack([g.center_ned for g in spec.gates])[:, 2]
        diffs.append(z.max() - z.min())
    # フルレンジでは高低差(急上昇区間含む)があるコースが大半
    assert np.median(diffs) > 2.0


def test_ribbon_mesh_shapes():
    spec = CourseGenerator(seed=0, stage=2).generate()
    verts, faces = ribbon_mesh(spec.ribbon_pts, width=0.8)
    assert verts.shape == (2 * len(spec.ribbon_pts), 3)
    assert faces.shape == (2 * (len(spec.ribbon_pts) - 1), 3)
    assert faces.max() < len(verts)


def test_deterministic():
    a = CourseGenerator(seed=5, stage=2).generate()
    b = CourseGenerator(seed=5, stage=2).generate()
    assert np.allclose(
        np.stack([g.center_ned for g in a.gates]), np.stack([g.center_ned for g in b.gates])
    )
