"""GenesisRaceEnv — バッチ並列ドローンレース環境。

物理120Hz、決定30Hz(4ステップ/決定、カメラフレームと同期)。
観測(actor): rgb (N,H,W,3) u8 + vec (N,55)、特権(critic): priv (N,39)。
アクション: a∈[-1,1]^4 → (roll_rate, pitch_rate, yaw_rate, thrust)(contracts.ActionMap)。

コース/色DRはシーン再構築(インスタンス作り直し)ごと。スポーン・ノイズ・動力学DRは
エピソードごと。Genesisは1プロセス1シーンが前提なので、再構築はプロセス内で
close() → 新インスタンス生成で行う。
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .. import contracts as C
from ..config import EnvConfig
from ..course import GATE_DEPTH, GATE_INNER, GATE_OUTER, CourseGenerator
from ..drone import DroneModel
from ..frames import (
    ProductionSigns,
    quat_from_euler_frd_ned,
    quat_rotate,
    quat_rotate_inv,
    quat_to_rot6d,
)
from ..latency import DelayQueue
from ..rewards import RewardComputer, RewardWeights
from ..scene_builder import SceneBuilder, ned2w, np_R_to_quat, rot_ned_to_world
from ..sensors.camera_rig import CameraRig, resolve_backend
from ..sensors.gate_detector import SimGateDetector
from ..sensors.imu import ImuSim

_GS_INITIALIZED = False


def _ensure_gs_init(seed: int):
    global _GS_INITIALIZED
    import genesis as gs

    if not _GS_INITIALIZED:
        gs.init(backend=gs.gpu, precision="32", seed=seed, logging_level="warning")
        # gs.init()はtorch.set_default_device()を呼ぶが、これは全torch API呼び出しに
        # Pythonフック(DeviceContext.__torch_function__)を挟み、collectorのCPU時間の
        # ~45%を占める(py-spy実測)。genesis_rl側のテンソル生成は全てdevice明示、
        # Genesis内部もgs.device明示(n_envs==0分岐を除く)なので、モードを外す。
        # default dtype(float32)の設定はそのまま生きる。
        # A/B検証・切り戻し用: GENESIS_RL_KEEP_DEFAULT_DEVICE=1 で従来挙動。
        import os

        if os.environ.get("GENESIS_RL_KEEP_DEFAULT_DEVICE") != "1":
            torch.set_default_device(None)
        _GS_INITIALIZED = True
    return gs


class GenesisRaceEnv:
    def __init__(self, cfg: EnvConfig, num_envs: int | None = None, course_seed: int | None = None,
                 stage: int | None = None, show_viewer: bool = False, extra_cameras: bool = False,
                 rng_seed: int | None = None):
        # rng_seed: torch/numpyグローバルRNGのシード(マルチcollectorでrankごとにずらす。
        # コース形状はcourse_seed固定なので、rankが違っても同一コース・別ノイズ系列になる)
        gs = _ensure_gs_init(cfg.course_seed if rng_seed is None else rng_seed)
        self.gs = gs
        self.cfg = cfg
        self.num_envs = num_envs or cfg.num_envs
        self.stage = cfg.stage if stage is None else stage
        seed = cfg.course_seed if course_seed is None else course_seed
        self.device = torch.device(str(gs.device))
        self.action_map = C.ActionMap()
        self.signs = ProductionSigns(cfg.signs_cmd, cfg.signs_gyro, cfg.signs_accel)

        # --- コース(per-env対応) ---
        # per_env: 各envに別コースを割り当てる(ゲートは表示専用+数値衝突)。有効化は
        # collectorがカリキュラムstage>=5のときだけ cfg.per_env_courses を立てて制御する
        # (envに渡るstageはcourse_stage=3で、カリキュラムstageとは別のため)。
        self.per_env = bool(getattr(cfg, "per_env_courses", False))
        self._env_ar = torch.arange(self.num_envs, device=self.device)
        self._init_courses(seed)

        # --- シーン ---
        rng = np.random.default_rng(seed + 777)
        backend = resolve_backend(cfg.render.backend)
        self.rig = CameraRig(backend, self.num_envs, cfg.render.width, cfg.render.height, self.device,
                             max_seq_envs=cfg.render.max_seq_envs,
                             exposure=float(getattr(cfg.render, "exposure", 1.0)))

        renderer = None
        vis_kwargs = {}
        if backend == "batch":
            renderer = gs.renderers.BatchRenderer()
        if backend == "sequential":
            vis_kwargs["env_separate_rigid"] = True
        rendered_idx = self.rig.rendered_envs_idx() or [0]

        amb = SceneBuilder(self.course, rng, cfg.color_dr, cfg.clutter, per_env=self.per_env)
        amb._pool_specs = self.pool_specs   # per-env装飾(柱/クラッタ/ポール/リボン)の全コース配置用
        self.builder = amb
        # 屋内シーン。ポイントライトは8192^2キューブシャドウマップを確保しVRAMを食い潰す
        # ため使わない。天井スラブは非表示(衝突のみ)にして平行光を屋内に届かせる
        # (実映像の天井も「黒地に発光ストリップ」なので見た目は一致する)。
        # 実飛行フレームの実測(黒地に発光要素のみ、暗部42%・明部10%)に合わせ平行光は弱く。
        # 実測: 実DCLは中間調(V 25-120)が画素の48%(柱/トラス/機体が薄灰で見える)。
        # 露出2.6込みで柱・機体の中間灰が出る強度に較正(強すぎると床が灰色に浮く)。
        lights = [
            {"type": "directional", "dir": (-0.3, -0.4, -1.0), "color": (1.0, 1.0, 1.0),
             "intensity": float(rng.uniform(0.35, 0.7))},
            {"type": "directional", "dir": (0.5, 0.3, -1.0), "color": (0.9, 0.9, 1.0),
             "intensity": float(rng.uniform(0.15, 0.4))},
        ]
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=C.DT_PHYS, substeps=1),
            rigid_options=gs.options.RigidOptions(dt=C.DT_PHYS, enable_collision=True,
                                                  constraint_solver=gs.constraint_solver.Newton),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=rendered_idx,
                ambient_light=(amb.colors.ambient,) * 3,
                background_color=(0.0, 0.0, 0.0),   # 実機は黒背景(既定の紺色だと露出増感で青モヤになる)
                lights=lights,
                **vis_kwargs,
            ),
            renderer=renderer if renderer is not None else gs.renderers.Rasterizer(),
            show_viewer=show_viewer,
        )
        self.drone_entity = amb.build_scene(self.scene, cfg.drone)
        self.rig.add_cameras(self.scene)
        self.extra_cams = {}
        if extra_cameras:
            self._add_extra_cameras()
        self.scene.build(n_envs=self.num_envs)
        self.rig.attach(self.drone_entity)
        if extra_cameras:
            self._attach_extra_cameras()
        if self.per_env:
            self._place_gates_per_env()
            self._place_world_pools_per_env()

        # --- モデル・センサー ---
        self.drone = DroneModel(self.drone_entity, self.scene.rigid_solver, cfg.drone, self.signs,
                                self.num_envs, self.device)
        self.imu = ImuSim(self.num_envs, cfg.sensors, self.signs, self.device)
        self.detector = SimGateDetector(self.num_envs, cfg.sensors, self.device)
        self.rewards = RewardComputer(self.num_envs, self.device, RewardWeights())

        s = cfg.sensors
        self.det_queue = DelayQueue(self.num_envs, (4,), max_delay=s.det_delay_frames + s.det_delay_jitter + 1,
                                    device=self.device)
        # 観測画像はレンダ解像度(=本番と同じ640x360)から obs_res へ全面リサイズしてから
        # キューへ積む。deploy(client.py)の cv2.resize(frame,(224,224)) と同じ形にするためで、
        # 副次的に遅延キューのVRAMも 640x360 の 1/4.6 で済む。
        self.obs_res = int(getattr(cfg.render, "obs_res", 224))
        self.img_queue = DelayQueue(self.num_envs, (self.obs_res, self.obs_res, 3),
                                    max_delay=s.img_delay_frames + s.img_delay_jitter,
                                    device=self.device, dtype=torch.uint8)

        # --- バッファ ---
        N = self.num_envs
        z = lambda *sh, dtype=torch.float32: torch.zeros(*sh, device=self.device, dtype=dtype)
        self.active_gate = z(N, dtype=torch.long) + 1     # 次に通過すべきゲート(0=スタートゲートの中でスポーン)
        self.spawn_gate = z(N, dtype=torch.long) + 1      # スポーン時のactive_gate(成功判定はここからの相対通過数)
        self.spawn_dist_g1 = z(N)                         # スポーン位置→ゲート1中心の距離 [m](カリキュラム可視化用)
        self.prev_x_rel = z(N)
        self.episode_steps = z(N, dtype=torch.long)       # 物理ステップ
        self.steps_since_gate = z(N, dtype=torch.long)
        self.last_action = z(N, C.ACTION_DIM)      # a_{t-1}
        self.last_action2 = z(N, C.ACTION_DIM)     # a_{t-2}(振動=符号反転の検出用)
        self.prev_omega = z(N, 3)                  # 前決定の角速度 [rad/s](角加速度ペナルティ用)
        self.prev_cmd = z(N, C.ACTION_DIM)
        self.act_delay = z(N, dtype=torch.long)
        self.d_prev = z(N)
        self.collision = z(N, dtype=torch.bool)
        self.gate_pass_flag = z(N, dtype=torch.bool)
        self.wrong_way_flag = z(N, dtype=torch.bool)
        self.resume_prob = 0.0                            # カリキュラムStage1+で>0
        self.required_gates = 1
        self._all_idx = torch.arange(N, device=self.device)
        self._static_geom_start = None                    # 衝突フィルタ用(ドローン以外は全てstatic)
        # 測光DR(cfg.render.photo_dr>0時のみ使用): per-envのゲイン/ガンマ/コントラスト/ノイズ
        self._photo_gain = torch.ones(N, 1, 1, 3, device=self.device)
        self._photo_gamma = torch.ones(N, 1, 1, 1, device=self.device)
        self._photo_contrast = torch.ones(N, 1, 1, 1, device=self.device)
        self._photo_offset = torch.zeros(N, 1, 1, 1, device=self.device)
        self._photo_noise = torch.zeros(N, 1, 1, 1, device=self.device)

    # --- コース初期化(単一 or per-envプール) ---

    def _init_courses(self, base_seed: int):
        """コースを生成し、ゲート幾何を (N,G,·) テンソルへ展開する。

        per_env時は course_pool 種(0ならenvごとにユニーク)のコースを生成し env に割当てる。
        非per_env時は全envが同一コース(pool=1)。どちらでもゲート参照は (N,G,·) で統一され、
        indexは env-arange gather になる(挙動は従来と一致)。
        """
        cfg = self.cfg
        N = self.num_envs
        dev = self.device
        if self.per_env:
            K = cfg.course_pool if cfg.course_pool and cfg.course_pool > 0 else N
            K = max(1, min(K, N))
        else:
            K = 1
        specs = [CourseGenerator(seed=base_seed + j, stage=self.stage,
                                 n_gates=cfg.n_gates).generate() for j in range(K)]
        # 全コースは同一ゲート数(stage固定)である前提
        self.course = specs[0]
        self.pool_specs = specs
        self.n_gates = self.course.n_gates
        G = self.n_gates
        env2course = np.array([e % K for e in range(N)], dtype=np.int64)
        self.env2course = env2course

        def _arrays(sp):
            centers = np.stack([g.center_ned for g in sp.gates])           # (G,3)
            yaws = np.array([g.yaw for g in sp.gates], dtype=np.float64)    # (G,)
            rots = np.stack([g.rotation_ned() for g in sp.gates])          # (G,3,3)
            arc = np.asarray(sp.gate_cum_arc, dtype=np.float64)            # (G,)
            return centers, yaws, rots, arc

        arrs = [_arrays(sp) for sp in specs]
        centers = np.stack([arrs[c][0] for c in env2course]).astype(np.float32)   # (N,G,3)
        yaws = np.stack([arrs[c][1] for c in env2course]).astype(np.float32)      # (N,G)
        rots = np.stack([arrs[c][2] for c in env2course]).astype(np.float32)      # (N,G,3,3)
        arc = np.stack([arrs[c][3] for c in env2course]).astype(np.float32)       # (N,G)
        self.gate_pos = torch.tensor(centers, device=dev)                          # (N,G,3) NED
        self.gate_yaw = torch.tensor(yaws, device=dev)                             # (N,G)
        self.gate_normal = torch.tensor(rots[:, :, :, 0], device=dev)             # (N,G,3)
        self.gate_side = torch.tensor(rots[:, :, :, 1], device=dev)
        self.gate_up = torch.tensor(-rots[:, :, :, 2], device=dev)
        self.gate_cum_arc = torch.tensor(arc, device=dev)                          # (N,G)
        self.total_arc = torch.tensor(
            np.array([max(specs[c].total_arc, 1e-6) for c in env2course], dtype=np.float32),
            device=dev)                                                            # (N,)

        # 柱の水平位置(N,P,2)。実機のYOLOXは柱の裏のゲートを検出できないので、検出器の
        # 遮蔽判定に使う。コース毎に本数が違うのでPmaxへゼロ埋め+validマスクで持つ。
        pil = [np.asarray(sp.pillars, np.float32).reshape(-1, 2) for sp in specs]
        pmax = max((len(p) for p in pil), default=0)
        pxy = np.zeros((len(specs), max(pmax, 1), 2), np.float32)
        pvalid = np.zeros((len(specs), max(pmax, 1)), bool)
        for j, p in enumerate(pil):
            pxy[j, :len(p)] = p
            pvalid[j, :len(p)] = True
        self.pillar_xy = torch.tensor(pxy[env2course], device=dev)                 # (N,P,2) NED
        self.pillar_valid = torch.tensor(pvalid[env2course], device=dev)           # (N,P)

        # 青パス(ribbon)を粗くダウンサンプルして per-env 保持(進路追従シェイピング用)。
        # 全コースは同一ゲート数=同一サンプル数なので固定長でstackできる。
        RIB_STRIDE = 8
        rds = [np.asarray(sp.ribbon_pts, np.float32)[::RIB_STRIDE] for sp in specs]
        Md = min(len(r) for r in rds)
        rds = np.stack([r[:Md] for r in rds])                                      # (K,Md,3)
        self.ribbon_ds = torch.tensor(rds[env2course], device=dev)                 # (N,Md,3) NED
        self._path_lead = 2   # 先読み点数(ds間隔≈stride×0.25m=2m → ~4m先)
        tilt = math.radians(C.CAM_TILT_DEG)
        self._cam_axis = torch.tensor([math.cos(tilt), 0.0, -math.sin(tilt)],
                                      device=dev)                                  # FRD: 前方+上tilt
        self._cos_fov = math.cos(math.radians(50.0))   # 視野±50°をゆるく視界内とみなす閾

    def _place_gates_per_env(self):
        """build後、per-envのゲート表示ボックスを各envのコース配置へ move する。

        バーは非固定+gravity_compensation。set_pos/set_quat は envs_idx でenv一括指定。
        コスト: (4×G) バー × 2 API ≈ 数百ms(起動時1回のみ)。
        """
        bars = getattr(self.builder, "gate_bar_entities", [])
        if not bars:
            return
        dev = self.device
        specs = self.pool_specs
        # コースごと・ゲートごとの world中心/回転/quat を前計算
        spec_gate = []
        for sp in specs:
            gl = []
            for gate in sp.gates:
                cw = np.array(ned2w(gate.center_ned), dtype=np.float64)
                R_w = rot_ned_to_world(gate.rotation_ned())
                q = np.array(np_R_to_quat(R_w), dtype=np.float64)
                gl.append((cw, R_w, q))
            spec_gate.append(gl)
        env2c = self.env2course                                  # (N,)
        for (ent, gi, off3) in bars:
            off = np.asarray(off3, dtype=np.float64)             # (normal, side, up) ゲート面内
            pos_c = np.stack([spec_gate[c][gi][0] + spec_gate[c][gi][1] @ off
                              for c in range(len(specs))]).astype(np.float32)   # (K,3)
            quat_c = np.stack([spec_gate[c][gi][2]
                               for c in range(len(specs))]).astype(np.float32)  # (K,4)
            pos = torch.tensor(pos_c[env2c], device=dev)                        # (N,3)
            quat = torch.tensor(quat_c[env2c], device=dev)                      # (N,4)
            # relative=False: pos/quatを直接world系として設定(morphのbuild時pose offsetを無視)。
            # ゲートバーは非identity quatで生成されるため、set_quatも必ずrelative=Falseにする。
            # (_all_idxはbuffer節で後から定義されるため、ここでは早期定義の_env_arを使う)
            ent.set_pos(pos, envs_idx=self._env_ar, zero_velocity=True, relative=False)
            ent.set_quat(quat, envs_idx=self._env_ar, zero_velocity=True, relative=False)

        self._place_glow_per_env(spec_gate, env2c)

    def _place_glow_per_env(self, spec_gate, env2c):
        """床グロー+光柱を各envのコース配置へ置き、per-env home を保存する。

        glow/col はアクティブゲートのみ点灯する動的表示なので、_update_glow が使う
        per-env home テンソル (_glow_home_penv / _glow_col_home_penv) をここで確定する
        (非per-envのように env0 の pose を共有すると各envのゲート位置がずれる)。
        """
        from ..scene_builder import PE_GLOW_COL_H

        glows = getattr(self.builder, "glow_entities", [])
        cols = getattr(self.builder, "glow_col_entities", [])
        if not glows:
            return
        dev = self.device
        K = len(spec_gate)
        G = len(spec_gate[0]) if K else 0
        # コース毎・ゲート毎の glow/col world中心 と z回転quat
        glow_pos_c = np.zeros((K, G, 3), np.float32)
        col_pos_c = np.zeros((K, G, 3), np.float32)
        quat_c = np.zeros((K, G, 4), np.float32)
        for c in range(K):
            for gi in range(G):
                cw, R_w, _ = spec_gate[c][gi]
                nrm = R_w[:, 0]
                ang = float(np.arctan2(nrm[1], nrm[0]))
                glow_pos_c[c, gi] = (cw[0], cw[1], 0.02)
                col_pos_c[c, gi] = (cw[0], cw[1], PE_GLOW_COL_H / 2.0)
                quat_c[c, gi] = (np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2))
        glow_home = torch.tensor(glow_pos_c[env2c], device=dev)   # (N,G,3)
        col_home = torch.tensor(col_pos_c[env2c], device=dev)     # (N,G,3)
        quat = torch.tensor(quat_c[env2c], device=dev)            # (N,G,4)
        self._glow_home_penv = glow_home
        self._glow_col_home_penv = col_home
        for k in range(G):
            glows[k].set_pos(glow_home[:, k], envs_idx=self._env_ar,
                             zero_velocity=True, relative=False)
            glows[k].set_quat(quat[:, k], envs_idx=self._env_ar,
                              zero_velocity=True, relative=False)
            if k < len(cols):
                cols[k].set_pos(col_home[:, k], envs_idx=self._env_ar,
                                zero_velocity=True, relative=False)
                cols[k].set_quat(quat[:, k], envs_idx=self._env_ar,
                                 zero_velocity=True, relative=False)

        # スタート固定台(gate 0 差し替え)を各envの gate 0(x,y,床)へ移動。
        # meshは床基準ローカル座標なので原点を(x,y,0)に置くだけ。向きはyaw不問=identity。
        pads = getattr(self.builder, "start_pad_entities", [])
        if pads and G > 0:
            pad_pos_c = np.stack([np.array([spec_gate[c][0][0][0], spec_gate[c][0][0][1], 0.0],
                                           dtype=np.float32) for c in range(K)])   # (K,3)
            pad_pos = torch.tensor(pad_pos_c[env2c], device=dev)                    # (N,3)
            pads[0].set_pos(pad_pos, envs_idx=self._env_ar,
                            zero_velocity=True, relative=False)

    def _place_world_pools_per_env(self):
        """柱/クラッタ/ポール/オーブ/リボンダッシュのプールを各envのコース配置へ置く。

        builder.pe_pools 各要素は {ents, pos_c(K,n,3), quat_c(K,n,4), valid_c(K,n)}。
        env2course で各envのコース値を gather し、未使用スロット(valid=False)は床下へ沈める。
        全て build 時1回のみ(装飾はエピソード中に動かない=静的)。
        """
        pools = getattr(self.builder, "pe_pools", [])
        if not pools:
            return
        dev = self.device
        env2c = self.env2course
        for pool in pools:
            ents = pool["ents"]
            pos_n = pool["pos_c"][env2c].copy()       # (N,n,3)
            quat_n = pool["quat_c"][env2c]            # (N,n,4)
            valid_n = pool["valid_c"][env2c]          # (N,n)
            pos_n[~valid_n, 2] = -80.0                # 未使用スロットは床下へ
            pos_t = torch.tensor(pos_n, device=dev)
            quat_t = torch.tensor(quat_n, device=dev)
            for i, ent in enumerate(ents):
                ent.set_pos(pos_t[:, i], envs_idx=self._env_ar, zero_velocity=True, relative=False)
                ent.set_quat(quat_t[:, i], envs_idx=self._env_ar, zero_velocity=True, relative=False)

    # --- 追加カメラ(preview用) ---

    def _add_extra_cameras(self):
        from ..sensors.camera_rig import chase_offset_T  # noqa: F401
        gsm = self.gs
        hall = self.course.hall
        self.extra_cams["fpv_hd"] = self.scene.add_camera(res=(C.IMG_W, C.IMG_H), fov=C.VFOV_DEG,
                                                          GUI=False, near=0.05, far=300.0)
        self.extra_cams["chase"] = self.scene.add_camera(res=(640, 360), fov=50, GUI=False,
                                                         near=0.05, far=300.0)
        # 俯瞰はホール内側のコーナー上部から(壁・天井があるため外からは見えない)
        self.extra_cams["overview"] = self.scene.add_camera(
            res=(640, 360), fov=75, GUI=False, near=0.1, far=500.0,
            pos=(-hall.length / 2 + 4.0, -hall.width / 2 + 4.0, hall.height - 1.0),
            lookat=(hall.length / 4, 0.0, 1.5))

    def _attach_extra_cameras(self):
        from ..sensors.camera_rig import chase_offset_T, fpv_offset_T
        self.extra_cams["fpv_hd"].attach(self.drone_entity.base_link, fpv_offset_T())
        # チェイスは近め(機体280mmが画面で見えるように)
        self.extra_cams["chase"].attach(self.drone_entity.base_link,
                                        chase_offset_T(back=2.0, up=0.9, pitch_down_deg=12.0))

    # --- リセット ---

    def reset(self):
        self.reset_idx(self._all_idx)
        # 初期観測を作るためのウォームアップ1決定(ホバー相当)
        obs, priv, *_ = self.step(torch.zeros(self.num_envs, C.ACTION_DIM, device=self.device))
        return obs, priv

    def reset_idx(self, envs_idx: torch.Tensor):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        cfg = self.cfg
        dev = self.device

        # スポーン: スタートゲート内側・中心よりやや下・前傾-17.8°(実測)
        start_pos = self.gate_pos[envs_idx, 0].clone()      # (n,3) 各envのスタートゲート
        start_pos[:, 2] += cfg.spawn_below_center  # NED: d正=下
        jitter = (torch.rand(n, 3, device=dev) * 2 - 1) * 0.1
        pos = start_pos + jitter
        jd = math.radians(cfg.spawn_jitter_deg)
        roll = (torch.rand(n, device=dev) * 2 - 1) * jd
        pitch = math.radians(cfg.spawn_pitch_deg) + (torch.rand(n, device=dev) * 2 - 1) * jd
        yaw = self.gate_yaw[envs_idx, 0].clone() + (torch.rand(n, device=dev) * 2 - 1) * jd
        active = torch.ones(n, device=dev, dtype=torch.long)

        # 途中スポーン(逆カリキュラム): 全ゲートのいずれかの2m手前・ゲート正対。
        # resume_probはcollectorが成功率に応じてアニールする(下手なうちはコース中の
        # ゲート直前から「1個通す」練習を全域で積み、上達したら正規スタートへ寄せる)
        if self.resume_prob > 0.0 and self.n_gates > 2:
            resume = torch.rand(n, device=dev) < self.resume_prob
            k = torch.randint(1, self.n_gates, (n,), device=dev)
            gp = self.gate_pos[envs_idx, k]
            gn = self.gate_normal[envs_idx, k]
            rp = gp - gn * 2.0
            pos = torch.where(resume.unsqueeze(1), rp, pos)
            yaw = torch.where(resume, self.gate_yaw[envs_idx, k], yaw)
            pitch = torch.where(resume, torch.zeros_like(pitch), pitch)
            active = torch.where(resume, k, active)

        quat = quat_from_euler_frd_ned(roll, pitch, yaw)
        self.drone.set_state(pos, quat, envs_idx)
        self.drone.reset_idx(envs_idx, dr=True)
        self.imu.reset_idx(envs_idx)
        self.rewards.reset_idx(envs_idx)

        s = cfg.sensors
        self.det_queue.reset_idx(envs_idx)
        self.det_queue.set_delay(
            s.det_delay_frames + torch.randint(0, s.det_delay_jitter + 1, (n,), device=dev), envs_idx)
        self.img_queue.reset_idx(envs_idx)
        self.img_queue.set_delay(
            s.img_delay_frames + torch.randint(0, s.img_delay_jitter + 1, (n,), device=dev), envs_idx)
        self.act_delay[envs_idx] = s.act_delay_steps + torch.randint(0, s.act_delay_jitter + 1, (n,), device=dev)

        # 測光DRパラメータの再サンプル(エピソード粒度)
        p = float(getattr(cfg.render, "photo_dr", 0.0))
        if p > 0.0:
            u = lambda lo, hi, *sh: torch.rand(n, *sh, device=dev) * (hi - lo) + lo
            self._photo_gain[envs_idx] = (1.0 + u(-0.35 * p, 0.35 * p, 1, 1, 3))
            self._photo_gamma[envs_idx] = torch.exp(u(-0.45 * p, 0.45 * p, 1, 1, 1))
            self._photo_contrast[envs_idx] = 1.0 + u(-0.30 * p, 0.30 * p, 1, 1, 1)
            self._photo_offset[envs_idx] = u(-0.10 * p, 0.10 * p, 1, 1, 1)
            self._photo_noise[envs_idx] = u(0.0, 0.03 * p, 1, 1, 1)

        self.active_gate[envs_idx] = active
        self.spawn_gate[envs_idx] = active
        g1 = min(1, self.n_gates - 1)
        self.spawn_dist_g1[envs_idx] = (pos - self.gate_pos[envs_idx, g1]).norm(dim=1)
        self._update_ribbon(envs_idx)
        self._update_glow(envs_idx)
        self.episode_steps[envs_idx] = 0
        self.steps_since_gate[envs_idx] = 0
        self.last_action[envs_idx] = 0.0
        self.last_action2[envs_idx] = 0.0
        self.prev_omega[envs_idx] = 0.0
        self.prev_cmd[envs_idx] = 0.0
        self.collision[envs_idx] = False
        gp = self.gate_pos[envs_idx, active]
        self.d_prev[envs_idx] = (pos - gp).norm(dim=1)
        rel = pos - gp
        self.prev_x_rel[envs_idx] = (rel * self.gate_normal[envs_idx, active]).sum(dim=1)

    # --- ステップ ---

    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        N = self.num_envs
        cmd_new = self.action_map.to_command(actions.to(self.device))
        self.gate_pass_flag[:] = False
        self.wrong_way_flag[:] = False
        self.collision[:] = False
        finish = torch.zeros(N, device=self.device, dtype=torch.bool)

        state = self.drone.state()
        for k in range(C.DECIMATION):
            use_new = (self.act_delay <= k).unsqueeze(1)
            cmd = torch.where(use_new, cmd_new, self.prev_cmd)
            self.drone.apply(cmd, state)
            self.scene.step()
            state = self.drone.state()
            # IMU(比力はDroneModelが印加力から解析計算)
            self.imu.tick_analytic(state["quat_ned"], self.drone.last_specific_force_frd,
                                   state["omega_frd"], C.DT_PHYS)
            self._check_gate_pass(state, finish)
            self._check_collision(state)
            self.episode_steps += 1
            self.steps_since_gate += 1

        self.prev_cmd = cmd_new
        # ゲート通過したenvはリボン表示窓と床グロー(次ゲートのみ点灯)を進める
        if self.gate_pass_flag.any():
            passed_idx = self.gate_pass_flag.nonzero(as_tuple=False).squeeze(1)
            self._update_ribbon(passed_idx)
            self._update_glow(passed_idx)
        # 青パスの点滅(全env共通の明滅、区間ごとに位相ずれ)
        self._tick_blink()
        # --- フレーム境界(30Hz): レンダ + 検出 ---
        rgb = self._to_obs_res(self.rig.render())
        if float(getattr(cfg.render, "photo_dr", 0.0)) > 0.0:
            rgb = self._apply_photo_dr(rgb)
        self.img_queue.push(rgb)
        act_idx = self.active_gate.clamp(max=self.n_gates - 1)
        gp = self.gate_pos[self._env_ar, act_idx]     # 報酬(接近距離)用のアクティブゲート
        # 観測用の検出は本番YOLOXと同じく「全ゲートの中で最も大きく映る1個」。どのゲートが
        # アクティブかは検出器に教えない(deployのVQ2ではトラック情報が来ないため)。
        det = self.detector.detect_scene(state["pos_ned"], state["quat_ned"],
                                         self.gate_pos, self.gate_normal, noise=True,
                                         pillar_xy=self.pillar_xy,
                                         pillar_valid=self.pillar_valid)
        self.det_queue.push(det)

        obs, priv, closeness = self._build_obs(state, actions)

        # --- 報酬 ---
        d_now = (state["pos_ned"] - gp).norm(dim=1)
        episode_t = self.episode_steps.float() * C.DT_PHYS
        path_view = self._path_view(state["pos_ned"], state["quat_ned"])
        reward = self.rewards.compute(
            gate_pass=self.gate_pass_flag, finish=finish, collision=self.collision,
            d_prev=self.d_prev, d_now=d_now, closeness=closeness, path_view=path_view,
            action=actions, last_action=self.last_action, last_action2=self.last_action2,
            omega=state["omega_frd"], prev_omega=self.prev_omega,
            wrong_way=self.wrong_way_flag,
            episode_t=episode_t, max_episode_s=cfg.max_episode_s,
        )
        self.d_prev = d_now.clone()
        self.last_action2 = self.last_action
        self.last_action = actions.clone()
        self.prev_omega = state["omega_frd"].clone()

        # --- 終端 ---
        timeout_gate = self.steps_since_gate > int(cfg.no_gate_timeout_s * C.PHYS_HZ)
        timeout_ep = self.episode_steps > int(cfg.max_episode_s * C.PHYS_HZ)
        time_outs = (timeout_gate | timeout_ep) & ~self.collision & ~finish
        done = self.collision | finish | time_outs

        # 成功はスポーン地点からの相対通過数で判定(途中スポーンにスキップ分の
        # クレジットを与えない)。終盤スポーンでrequired本残っていない場合はfinishで成功。
        gates_passed = (self.active_gate - self.spawn_gate).clamp(min=0)
        success = (gates_passed >= self.required_gates) | finish
        info = {
            "time_outs": time_outs,
            "gates_passed": gates_passed.clone(),
            "success": success,
            "finish": finish,
            "collision": self.collision.clone(),
        }
        if done.any():
            idx = done.nonzero(as_tuple=False).squeeze(1)
            info["episode"] = {k: v[idx].mean().item() for k, v in self.rewards.episode_sums.items()}
            info["done_idx"] = idx
            info["done_gates"] = gates_passed[idx].clone()
            info["done_success"] = success[idx].clone()
            info["done_spawn_gate"] = self.spawn_gate[idx].clone()
            info["done_spawn_dist_g1"] = self.spawn_dist_g1[idx].clone()
            # 終端時の最終観測(n-stepのnext_obs用)。返り値のobsはリセット後に差し替える。
            info["final_obs"] = {"rgb": obs["rgb"][idx].clone(), "vec": obs["vec"][idx].clone()}
            info["final_priv"] = priv[idx].clone()
            self.reset_idx(idx)
            state_new = self.drone.state()
            obs_new, priv_new, _ = self._build_obs(state_new, self.last_action)
            obs["rgb"][idx] = obs_new["rgb"][idx]
            obs["vec"][idx] = obs_new["vec"][idx]
            priv[idx] = priv_new[idx]

        return obs, priv, reward, done, info

    # --- 内部 ---

    def _path_view(self, pos, quat):
        """機体近傍の青パス(ribbon)の少し先の点が、カメラ視野中心に近いほど1に近い値(N,)∈[0,1]。

        ゲートが柱裏/軸外で見えない旋回中でも密な航法シェイピングを与える。特権情報(ribbon幾何)を
        使うが観測ではないためDCL転移に影響しない。学習される行動「進路方向を視野に入れ続ける」は
        リボンの無いDCLでも有効(=次ゲートへ向く)。
        """
        rb = self.ribbon_ds                                  # (N,Md,3) NED
        d = (rb - pos.unsqueeze(1)).pow(2).sum(dim=2)        # (N,Md) 各ds点までの距離^2
        j = d.argmin(dim=1)                                  # 最近点
        jl = (j + self._path_lead).clamp(max=rb.shape[1] - 1)   # 少し先の点
        tgt = rb[self._env_ar, jl]                           # (N,3) 先読み点
        rel_body = quat_rotate_inv(quat, tgt - pos)          # 機体FRDでの方向
        rel_body = rel_body / (rel_body.norm(dim=1, keepdim=True) + 1e-6)
        cos_ang = (rel_body * self._cam_axis).sum(dim=1)     # カメラ光軸との内積(=中心度)
        return ((cos_ang - self._cos_fov) / (1.0 - self._cos_fov)).clamp(0.0, 1.0)

    def _check_gate_pass(self, state, finish: torch.Tensor):
        act = self.active_gate.clamp(max=self.n_gates - 1)
        gp = self.gate_pos[self._env_ar, act]
        gn = self.gate_normal[self._env_ar, act]
        gside = self.gate_side[self._env_ar, act]
        gup = self.gate_up[self._env_ar, act]
        rel = state["pos_ned"] - gp
        x_rel = (rel * gn).sum(dim=1)
        crossed = (self.prev_x_rel < 0) & (x_rel >= 0) & (self.active_gate < self.n_gates)
        passed = torch.zeros_like(crossed)
        if crossed.any():
            # 交点の面内オフセット(側方・面内上方、ゲート傾き込み)。1物理ステップの
            # 移動は小さいので現在位置の面内成分で近似(120Hzチェック、誤差は数cm)。
            y_off = (rel * gside).sum(dim=1)
            z_off = (rel * gup).sum(dim=1)
            inside = (y_off.abs() < 0.75) & (z_off.abs() < 0.75)
            passed = crossed & inside
            if passed.any():  # noqa: SIM102
                self.gate_pass_flag |= passed
                self.steps_since_gate[passed] = 0
                self.active_gate[passed] += 1
                fin = passed & (self.active_gate >= self.n_gates)
                finish |= fin
                # 新しいアクティブゲートへの基準を更新
                na = self.active_gate.clamp(max=self.n_gates - 1)
                np_ = self.gate_pos[self._env_ar, na]
                nn = self.gate_normal[self._env_ar, na]
                nrel = state["pos_ned"] - np_
                nx = (nrel * nn).sum(dim=1)
                self.prev_x_rel = torch.where(passed, nx, self.prev_x_rel)
                self.d_prev = torch.where(passed, nrel.norm(dim=1), self.d_prev)
        # 逆走: アクティブゲート面を逆向きに横切る
        back = (self.prev_x_rel >= 0) & (x_rel < 0)
        self.wrong_way_flag |= back & ~crossed
        # 通過したenvはprev_x_relが新ゲート基準に更新済み。それ以外は現在値へ。
        self.prev_x_rel = torch.where(passed, self.prev_x_rel, x_rel)

    RIBBON_AHEAD = 5      # 青パスはアクティブゲートから5ゲート先まで表示
    BLINK_PERIOD = 18     # 点滅周期 [決定ステップ] = 0.6s @30Hz
    BLINK_ON = 13         # うち点灯ステップ数(デューティ~72%)
    BLINK_PHASE = 4       # 区間ごとの位相ずれ(流れるような明滅)

    def _update_ribbon(self, envs_idx: torch.Tensor, segments: list[int] | None = None):
        """リボン区間の表示を更新(表示窓 ∧ 点滅状態)。非表示は床下(-80m)へ沈める。

        区間エンティティは非固定+gravity_compensation=1.0なのでper-envにset_posできる。
        set_posはGenesis API 1回≈1.4msと高価なので、前回書いた表示状態(_ribbon_vis)
        との差分がある(区間×env)だけ書く。ビルド直後は全区間homeにある(=表示)。
        """
        ents = getattr(self.builder, "ribbon_entities", [])
        rails = getattr(self.builder, "ribbon_rail_entities", [])
        if not ents or len(envs_idx) == 0:
            return
        if not hasattr(self, "_ribbon_home"):
            # build直後の基準位置をキャプチャ(メッシュ再センタリングに依存しないため)
            self._ribbon_home = [ent.get_pos()[0].clone() for ent in ents]
            self._ribbon_rail_home = [e.get_pos()[0].clone() for e in rails]
            self._blink_on = torch.ones(len(ents), device=self.device, dtype=torch.bool)
            self._ribbon_vis = torch.ones(self.num_envs, len(ents),
                                          device=self.device, dtype=torch.bool)
        active = self.active_gate[envs_idx]
        sink = torch.tensor([0.0, 0.0, -80.0], device=self.device)
        gate_i = torch.arange(1, len(ents) + 1, device=self.device)  # 区間kが導くゲート番号
        vis_all = ((active.unsqueeze(1) <= gate_i)
                   & (gate_i < (active + self.RIBBON_AHEAD).unsqueeze(1))
                   & self._blink_on)                                  # (n, K)
        diff_all = vis_all != self._ribbon_vis[envs_idx]
        if segments is not None:
            seg_mask = torch.zeros(len(ents), device=self.device, dtype=torch.bool)
            seg_mask[segments] = True
            diff_all &= seg_mask
        for k in diff_all.any(dim=0).nonzero(as_tuple=False).squeeze(1).tolist():
            d = diff_all[:, k]
            idx = envs_idx[d]
            vis = vis_all[d, k]
            home = self._ribbon_home[k].expand(len(idx), 3)
            pos = torch.where(vis.unsqueeze(1), home, home + sink)
            ents[k].set_pos(pos, envs_idx=idx, zero_velocity=True, relative=False)
            if k < len(rails):   # 縁レールもフィルと同期して表示/非表示
                rail_home = self._ribbon_rail_home[k].expand(len(idx), 3)
                rails[k].set_pos(torch.where(vis.unsqueeze(1), rail_home, rail_home + sink),
                                 envs_idx=idx, zero_velocity=True, relative=False)
            self._ribbon_vis[idx, k] = vis

    def _tick_blink(self):
        """点滅状態を1決定ステップ進め、変化した区間だけ全envで表示を更新する。"""
        if not hasattr(self, "_blink_tick"):
            self._blink_tick = 0
        self._blink_tick += 1
        ents = getattr(self.builder, "ribbon_entities", [])
        if not ents or not hasattr(self, "_blink_on"):
            return
        k_ar = torch.arange(len(ents), device=self.device)
        on = ((self._blink_tick + k_ar * self.BLINK_PHASE) % self.BLINK_PERIOD) < self.BLINK_ON
        changed = (on != self._blink_on).nonzero(as_tuple=False).squeeze(1)
        self._blink_on = on
        if len(changed) > 0:
            self._update_ribbon(self._all_idx, segments=changed.tolist())

    def _update_glow(self, envs_idx: torch.Tensor):
        """床の金色グロー+光柱は「次に行くべきゲート」1つだけ点灯する。

        _update_ribbonと同様、前回書いた状態(_glow_vis)との差分だけset_posする。
        glow_col_entities(光柱)は glow_entities(床帯)と同じ可視状態で同期移動する。
        """
        glows = getattr(self.builder, "glow_entities", [])
        cols = getattr(self.builder, "glow_col_entities", [])
        if not glows or len(envs_idx) == 0:
            return
        # per-env(コースがenv毎に違う)は _place_glow_per_env が確定した per-env home を使う。
        # 非per-envは全env同一コースなので env0 の pose を共有ホームにする(従来挙動)。
        penv = hasattr(self, "_glow_home_penv")
        if not penv and not hasattr(self, "_glow_home"):
            self._glow_home = [g.get_pos()[0].clone() for g in glows]
            self._glow_col_home = [g.get_pos()[0].clone() for g in cols]
        if not hasattr(self, "_glow_vis"):
            self._glow_vis = torch.ones(self.num_envs, len(glows),
                                        device=self.device, dtype=torch.bool)
        active = self.active_gate[envs_idx]
        sink = torch.tensor([0.0, 0.0, -80.0], device=self.device)
        k_ar = torch.arange(len(glows), device=self.device)
        vis_all = active.unsqueeze(1) == k_ar                        # (n, G)
        diff_all = vis_all != self._glow_vis[envs_idx]
        for k in diff_all.any(dim=0).nonzero(as_tuple=False).squeeze(1).tolist():
            d = diff_all[:, k]
            idx = envs_idx[d]
            vis = vis_all[d, k]
            home = self._glow_home_penv[idx, k] if penv else self._glow_home[k].expand(len(idx), 3)
            glows[k].set_pos(torch.where(vis.unsqueeze(1), home, home + sink),
                             envs_idx=idx, zero_velocity=True, relative=False)
            if k < len(cols):
                col_home = self._glow_col_home_penv[idx, k] if penv \
                    else self._glow_col_home[k].expand(len(idx), 3)
                cols[k].set_pos(torch.where(vis.unsqueeze(1), col_home, col_home + sink),
                                envs_idx=idx, zero_velocity=True, relative=False)
            self._glow_vis[idx, k] = vis

    def _to_obs_res(self, rgb_u8: torch.Tensor) -> torch.Tensor:
        """(N,H,W,3) uint8 → (N,obs_res,obs_res,3) uint8。deployと同じ全面リサイズ。

        本番は 640x360 のJPEGを cv2.resize(...,(224,224)) でアスペクトを潰して224角にする。
        学習も同じ 640x360 からの縮小にしないと、エンコーダから見た鮮鋭度/エイリアスが
        別ドメインになる(旧: 320x180 を224へ拡大していた)。
        """
        if rgb_u8.shape[1] == self.obs_res and rgb_u8.shape[2] == self.obs_res:
            return rgb_u8
        x = rgb_u8.permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(x, size=(self.obs_res, self.obs_res),
                                            mode="bilinear", align_corners=False)
        return x.clamp(0.0, 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    def _apply_photo_dr(self, rgb_u8: torch.Tensor) -> torch.Tensor:
        """per-env測光変換: ガンマ→コントラスト→per-channelゲイン→輝度オフセット→ノイズ。

        レンダラ(Madrona/rasterizer/実シムDCL)の色応答・露出・粒状感の差を学習時に
        経験させ、視覚特徴のドメイン過適合を防ぐ。(N,H,W,3) uint8 → 同形uint8。
        """
        x = rgb_u8.float() / 255.0
        x = x.clamp(min=1e-4).pow(self._photo_gamma)
        x = (x - 0.45) * self._photo_contrast + 0.45
        x = x * self._photo_gain + self._photo_offset
        noise = self._photo_noise
        if bool((noise > 0).any()):
            x = x + torch.randn_like(x) * noise
        return (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    def _check_collision(self, state=None):
        if self.per_env:
            self._check_collision_numeric(state)
            return
        contacts = self.drone_entity.get_contacts()
        if contacts is None or "valid_mask" not in contacts:
            return
        valid = contacts["valid_mask"]
        if valid is None or valid.numel() == 0:
            return
        hit = torch.as_tensor(valid, device=self.device).any(dim=-1)
        grace = self.episode_steps < int(self.cfg.collision_grace_s * C.PHYS_HZ)
        self.collision |= hit & ~grace

    # 数値衝突用のマージン(機体半サイズ ~0.14m)
    _COLL_MARGIN = 0.14

    def _check_collision_numeric(self, state):
        """per-envモードの衝突判定(物理接触の代わり)。ホール境界 + アクティブゲート枠。

        ホール(120×50×10, 固定)は全env共通。ゲート枠は各envのコース配置に対して、
        面近傍(奥行±)かつ開口(1.5m)の外・外形(2.7m)の内=フレーム帯にいれば衝突。
        """
        if state is None:
            return
        m = self._COLL_MARGIN
        pos = state["pos_ned"]                         # (N,3) NED (n,e,d)
        hall = self.course.hall
        n_c, e_c, d_c = pos[:, 0], pos[:, 1], pos[:, 2]
        alt = -d_c                                     # world z (上正)
        hall_hit = ((n_c.abs() > hall.length / 2 - m)
                    | (e_c.abs() > hall.width / 2 - m)
                    | (alt < m)
                    | (alt > hall.height - m))
        # アクティブゲート枠
        act = self.active_gate.clamp(max=self.n_gates - 1)
        gp = self.gate_pos[self._env_ar, act]
        gn = self.gate_normal[self._env_ar, act]
        gside = self.gate_side[self._env_ar, act]
        gup = self.gate_up[self._env_ar, act]
        rel = pos - gp
        x_rel = (rel * gn).sum(dim=1).abs()
        y_off = (rel * gside).sum(dim=1).abs()
        z_off = (rel * gup).sum(dim=1).abs()
        opening = GATE_INNER / 2                        # 0.75
        outer = GATE_OUTER / 2                          # 1.35
        near = x_rel < GATE_DEPTH / 2 + m
        within = (y_off <= outer + m) & (z_off <= outer + m)
        in_frame = within & ((y_off >= opening - m) | (z_off >= opening - m))
        gate_hit = near & in_frame
        grace = self.episode_steps < int(self.cfg.collision_grace_s * C.PHYS_HZ)
        self.collision |= (hall_hit | gate_hit) & ~grace

    def _build_obs(self, state, actions):
        N = self.num_envs
        vec = torch.zeros(N, C.VEC_DIM, device=self.device)
        imu = self.imu.read()
        vec[:, C.VEC_GYRO] = imu[:, :3] / C.RATE_SCALE
        vec[:, C.VEC_ACCEL] = imu[:, 3:] / C.ACCEL_SCALE

        det = self.det_queue.read()  # [u_n, v_n, vis, rel_dist]
        age_s = self.det_queue.age().float() * C.DT_POLICY
        stale = age_s > C.GATE_OBS_MAX_AGE_S
        vis = det[:, 2] * (~stale).float()
        age_n = (age_s / C.GATE_OBS_MAX_AGE_S).clamp(0, 1)
        vec[:, C.VEC_GATE] = torch.stack([det[:, 0], det[:, 1], vis, det[:, 3], age_n], dim=1)

        passed = (self.active_gate - 1).clamp(min=0, max=C.MAX_GATES - 1)
        onehot = torch.nn.functional.one_hot(passed, C.MAX_GATES).float()
        vec[:, C.VEC_ONEHOT] = onehot
        # last_action は「直前に出したアクション」= 引数の actions。self.last_action は
        # このメソッドの後で更新されるため、そちらを使うと1決定ぶん古い a_{t-2} が載り、
        # deploy(a_{t-1}を載せる)と33msズレる(2026-07-26に実測で確認した旧バグ)。
        vec[:, C.VEC_LAST_ACTION] = actions

        obs = {"rgb": self.img_queue.read(), "vec": vec}

        # --- 特権(critic) ---
        act = self.active_gate.clamp(max=self.n_gates - 1)
        idx3 = torch.stack([act, (act + 1).clamp(max=self.n_gates - 1),
                            (act + 2).clamp(max=self.n_gates - 1)], dim=1)  # (N,3)
        er = self._env_ar.unsqueeze(1)                # (N,1) env-arange for per-env gather
        gp3 = self.gate_pos[er, idx3]                 # (N,3,3)
        rel3 = gp3 - state["pos_ned"].unsqueeze(1)
        rel3_body = quat_rotate_inv(state["quat_ned"].unsqueeze(1).expand(-1, 3, -1), rel3)
        gy3 = self.gate_yaw[er, idx3]
        # ゲート法線方位 − 機体ヨー
        x_body = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(N, 3)
        heading = quat_rotate(state["quat_ned"], x_body)
        body_yaw = torch.atan2(heading[:, 1], heading[:, 0])
        dyaw = gy3 - body_yaw.unsqueeze(1)

        t_since_gate = self.steps_since_gate.float() * C.DT_PHYS
        ep_t = self.episode_steps.float() * C.DT_PHYS
        # 弧長進捗(per-envコース対応: gate_cum_arc/total_arc は (N,G)/(N,))
        gidx = (self.active_gate - 1).clamp(min=0, max=self.n_gates - 1)
        prog = self.gate_cum_arc[self._env_ar, gidx] / self.total_arc

        priv = torch.cat(
            [
                state["pos_ned"] / C.POS_SCALE,
                state["vel_ned"] / C.V_SCALE,
                quat_to_rot6d(state["quat_ned"]),
                state["omega_frd"] / C.RATE_SCALE,
                (rel3_body / C.GATE_REL_SCALE).reshape(N, 9),
                torch.sin(dyaw), torch.cos(dyaw),
                actions,
                (t_since_gate / self.cfg.no_gate_timeout_s).clamp(0, 1).unsqueeze(1),
                (ep_t / self.cfg.max_episode_s).clamp(0, 1).unsqueeze(1),
                (self.active_gate.float() / self.n_gates).unsqueeze(1),
                prog.unsqueeze(1),
                self.prev_cmd[:, 3:4],
            ],
            dim=1,
        )

        # 真値closeness(報酬用・ノイズなし)
        gp = self.gate_pos[self._env_ar, act]
        gn = self.gate_normal[self._env_ar, act]
        # 報酬用のclosenessも遮蔽込みで評価する(柱越しに「見えている」ことにすると、
        # 実際には映らない位置取りを強化してしまうため)
        det_true = self.detector.detect(state["pos_ned"], state["quat_ned"], gp, gn, noise=False,
                                        pillar_xy=self.pillar_xy,
                                        pillar_valid=self.pillar_valid)
        closeness = (1.0 - det_true[:, 3]) * det_true[:, 2]

        return obs, priv, closeness

    def set_stage_runtime(self, *, noise_scale: float | None = None, resume_prob: float | None = None,
                          required_gates: int | None = None, speed_finish_w: float | None = None,
                          dr_scale: float | None = None):
        """再構築不要なステージ依存パラメータの更新。"""
        if noise_scale is not None:
            self.cfg.sensors.noise_scale = noise_scale
        if resume_prob is not None:
            self.resume_prob = resume_prob
        if required_gates is not None:
            self.required_gates = required_gates
        if speed_finish_w is not None:
            self.rewards.w.speed_finish = speed_finish_w
        if dr_scale is not None:
            self.drone.dr_scale = dr_scale

    def close(self):
        # Genesisはシーン単位の破棄APIが限定的。プロセス内再構築はscene参照を捨てるだけで
        # メモリが再利用されないことがあるため、再構築はワーカープロセス再起動で行うのが安全。
        self.scene = None
