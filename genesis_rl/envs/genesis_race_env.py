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
from ..course import CourseGenerator
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
from ..scene_builder import SceneBuilder
from ..sensors.camera_rig import CameraRig, resolve_backend
from ..sensors.gate_detector import SimGateDetector
from ..sensors.imu import ImuSim

_GS_INITIALIZED = False


def _ensure_gs_init(seed: int):
    global _GS_INITIALIZED
    import genesis as gs

    if not _GS_INITIALIZED:
        gs.init(backend=gs.gpu, precision="32", seed=seed, logging_level="warning")
        _GS_INITIALIZED = True
    return gs


class GenesisRaceEnv:
    def __init__(self, cfg: EnvConfig, num_envs: int | None = None, course_seed: int | None = None,
                 stage: int | None = None, show_viewer: bool = False, extra_cameras: bool = False):
        gs = _ensure_gs_init(cfg.course_seed)
        self.gs = gs
        self.cfg = cfg
        self.num_envs = num_envs or cfg.num_envs
        self.stage = cfg.stage if stage is None else stage
        seed = cfg.course_seed if course_seed is None else course_seed
        self.device = torch.device(str(gs.device))
        self.action_map = C.ActionMap()
        self.signs = ProductionSigns(cfg.signs_cmd, cfg.signs_gyro, cfg.signs_accel)

        # --- コース ---
        self.course = CourseGenerator(seed=seed, stage=self.stage, n_gates=cfg.n_gates).generate()
        self.n_gates = self.course.n_gates
        gate_centers = np.stack([g.center_ned for g in self.course.gates])
        gate_yaws = np.array([g.yaw for g in self.course.gates])
        # ゲートは傾き(pitch/roll)を持つ → 面の3軸(法線・側方・面内上方)を回転行列から取る
        rots = np.stack([g.rotation_ned() for g in self.course.gates])  # (G,3,3)
        self.gate_pos = torch.tensor(gate_centers, device=self.device, dtype=torch.float32)   # (G,3) NED
        self.gate_yaw = torch.tensor(gate_yaws, device=self.device, dtype=torch.float32)      # (G,)
        self.gate_normal = torch.tensor(rots[:, :, 0], device=self.device, dtype=torch.float32)
        self.gate_side = torch.tensor(rots[:, :, 1], device=self.device, dtype=torch.float32)
        self.gate_up = torch.tensor(-rots[:, :, 2], device=self.device, dtype=torch.float32)

        # --- シーン ---
        rng = np.random.default_rng(seed + 777)
        backend = resolve_backend(cfg.render.backend)
        self.rig = CameraRig(backend, self.num_envs, cfg.render.width, cfg.render.height, self.device)

        renderer = None
        vis_kwargs = {}
        if backend == "batch":
            renderer = gs.renderers.BatchRenderer()
        if backend == "sequential":
            vis_kwargs["env_separate_rigid"] = True
        rendered_idx = self.rig.rendered_envs_idx() or [0]

        amb = SceneBuilder(self.course, rng, cfg.color_dr, cfg.clutter)
        self.builder = amb
        # 屋内シーン。ポイントライトは8192^2キューブシャドウマップを確保しVRAMを食い潰す
        # ため使わない。天井スラブは非表示(衝突のみ)にして平行光を屋内に届かせる
        # (実映像の天井も「黒地に発光ストリップ」なので見た目は一致する)。
        lights = [
            {"type": "directional", "dir": (-0.3, -0.4, -1.0), "color": (1.0, 1.0, 1.0),
             "intensity": float(rng.uniform(2.5, 4.5))},
            {"type": "directional", "dir": (0.5, 0.3, -1.0), "color": (0.9, 0.9, 1.0),
             "intensity": float(rng.uniform(1.0, 2.5))},
        ]
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=C.DT_PHYS, substeps=1),
            rigid_options=gs.options.RigidOptions(dt=C.DT_PHYS, enable_collision=True,
                                                  constraint_solver=gs.constraint_solver.Newton),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=rendered_idx,
                ambient_light=(amb.colors.ambient,) * 3,
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

        # --- モデル・センサー ---
        self.drone = DroneModel(self.drone_entity, self.scene.rigid_solver, cfg.drone, self.signs,
                                self.num_envs, self.device)
        self.imu = ImuSim(self.num_envs, cfg.sensors, self.signs, self.device)
        self.detector = SimGateDetector(self.num_envs, cfg.sensors, self.device)
        self.rewards = RewardComputer(self.num_envs, self.device, RewardWeights())

        s = cfg.sensors
        self.det_queue = DelayQueue(self.num_envs, (4,), max_delay=s.det_delay_frames + s.det_delay_jitter + 1,
                                    device=self.device)
        self.img_queue = DelayQueue(self.num_envs, (cfg.render.height, cfg.render.width, 3),
                                    max_delay=s.img_delay_frames + s.img_delay_jitter,
                                    device=self.device, dtype=torch.uint8)

        # --- バッファ ---
        N = self.num_envs
        z = lambda *sh, dtype=torch.float32: torch.zeros(*sh, device=self.device, dtype=dtype)
        self.active_gate = z(N, dtype=torch.long) + 1     # 次に通過すべきゲート(0=スタートゲートの中でスポーン)
        self.prev_x_rel = z(N)
        self.episode_steps = z(N, dtype=torch.long)       # 物理ステップ
        self.steps_since_gate = z(N, dtype=torch.long)
        self.last_action = z(N, C.ACTION_DIM)
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
        start_pos = self.gate_pos[0].expand(n, 3).clone()
        start_pos[:, 2] += cfg.spawn_below_center  # NED: d正=下
        jitter = (torch.rand(n, 3, device=dev) * 2 - 1) * 0.1
        pos = start_pos + jitter
        jd = math.radians(cfg.spawn_jitter_deg)
        roll = (torch.rand(n, device=dev) * 2 - 1) * jd
        pitch = math.radians(cfg.spawn_pitch_deg) + (torch.rand(n, device=dev) * 2 - 1) * jd
        yaw = self.gate_yaw[0].expand(n).clone() + (torch.rand(n, device=dev) * 2 - 1) * jd
        active = torch.ones(n, device=dev, dtype=torch.long)

        # 途中スポーン(カリキュラム): ゲートkの2m手前・ゲート正対
        if self.resume_prob > 0.0 and self.n_gates > 2:
            resume = torch.rand(n, device=dev) < self.resume_prob
            k = torch.randint(1, min(5, self.n_gates - 1), (n,), device=dev)
            gp = self.gate_pos[k]
            gn = self.gate_normal[k]
            rp = gp - gn * 2.0
            pos = torch.where(resume.unsqueeze(1), rp, pos)
            yaw = torch.where(resume, self.gate_yaw[k], yaw)
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

        self.active_gate[envs_idx] = active
        self._update_ribbon(envs_idx)
        self._update_glow(envs_idx)
        self.episode_steps[envs_idx] = 0
        self.steps_since_gate[envs_idx] = 0
        self.last_action[envs_idx] = 0.0
        self.prev_cmd[envs_idx] = 0.0
        self.collision[envs_idx] = False
        gp = self.gate_pos[active]
        self.d_prev[envs_idx] = (pos - gp).norm(dim=1)
        rel = pos - gp
        self.prev_x_rel[envs_idx] = (rel * self.gate_normal[active]).sum(dim=1)

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
            self._check_collision()
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
        rgb = self.rig.render()
        self.img_queue.push(rgb)
        act_idx = self.active_gate.clamp(max=self.n_gates - 1)
        gp = self.gate_pos[act_idx]
        gn = self.gate_normal[act_idx]
        det = self.detector.detect(state["pos_ned"], state["quat_ned"], gp, gn, noise=True)
        self.det_queue.push(det)

        obs, priv, closeness = self._build_obs(state, actions)

        # --- 報酬 ---
        d_now = (state["pos_ned"] - gp).norm(dim=1)
        episode_t = self.episode_steps.float() * C.DT_PHYS
        reward = self.rewards.compute(
            gate_pass=self.gate_pass_flag, finish=finish, collision=self.collision,
            d_prev=self.d_prev, d_now=d_now, closeness=closeness,
            action=actions, last_action=self.last_action,
            omega_norm=state["omega_frd"].norm(dim=1), wrong_way=self.wrong_way_flag,
            episode_t=episode_t, max_episode_s=cfg.max_episode_s,
        )
        self.d_prev = d_now.clone()
        self.last_action = actions.clone()

        # --- 終端 ---
        timeout_gate = self.steps_since_gate > int(cfg.no_gate_timeout_s * C.PHYS_HZ)
        timeout_ep = self.episode_steps > int(cfg.max_episode_s * C.PHYS_HZ)
        time_outs = (timeout_gate | timeout_ep) & ~self.collision & ~finish
        done = self.collision | finish | time_outs

        gates_passed = (self.active_gate - 1).clamp(min=0)
        success = gates_passed >= self.required_gates
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

    def _check_gate_pass(self, state, finish: torch.Tensor):
        act = self.active_gate.clamp(max=self.n_gates - 1)
        gp = self.gate_pos[act]
        gn = self.gate_normal[act]
        gside = self.gate_side[act]
        gup = self.gate_up[act]
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
                np_ = self.gate_pos[na]
                nn = self.gate_normal[na]
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
        """
        ents = getattr(self.builder, "ribbon_entities", [])
        if not ents or len(envs_idx) == 0:
            return
        if not hasattr(self, "_ribbon_home"):
            # build直後の基準位置をキャプチャ(メッシュ再センタリングに依存しないため)
            self._ribbon_home = [ent.get_pos()[0].clone() for ent in ents]
            self._blink_on = torch.ones(len(ents), device=self.device, dtype=torch.bool)
        active = self.active_gate[envs_idx]
        sink = torch.tensor([0.0, 0.0, -80.0], device=self.device)
        for k in (range(len(ents)) if segments is None else segments):
            gate_i = k + 1  # この区間が導くゲート番号
            vis = (active <= gate_i) & (gate_i < active + self.RIBBON_AHEAD) & self._blink_on[k]
            home = self._ribbon_home[k].expand(len(envs_idx), 3)
            ents[k].set_pos(torch.where(vis.unsqueeze(1), home, home + sink),
                            envs_idx=envs_idx, zero_velocity=True, relative=False)

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
        """床の金色グローは「次に行くべきゲート」1つだけ点灯する。"""
        glows = getattr(self.builder, "glow_entities", [])
        if not glows or len(envs_idx) == 0:
            return
        if not hasattr(self, "_glow_home"):
            self._glow_home = [g.get_pos()[0].clone() for g in glows]
        active = self.active_gate[envs_idx]
        sink = torch.tensor([0.0, 0.0, -80.0], device=self.device)
        for k, g in enumerate(glows):
            vis = active == k
            home = self._glow_home[k].expand(len(envs_idx), 3)
            g.set_pos(torch.where(vis.unsqueeze(1), home, home + sink),
                      envs_idx=envs_idx, zero_velocity=True, relative=False)

    def _check_collision(self):
        contacts = self.drone_entity.get_contacts()
        if contacts is None or "valid_mask" not in contacts:
            return
        valid = contacts["valid_mask"]
        if valid is None or valid.numel() == 0:
            return
        hit = torch.as_tensor(valid, device=self.device).any(dim=-1)
        grace = self.episode_steps < int(self.cfg.collision_grace_s * C.PHYS_HZ)
        self.collision |= hit & ~grace

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
        vec[:, C.VEC_LAST_ACTION] = self.last_action

        obs = {"rgb": self.img_queue.read(), "vec": vec}

        # --- 特権(critic) ---
        act = self.active_gate.clamp(max=self.n_gates - 1)
        idx3 = torch.stack([act, (act + 1).clamp(max=self.n_gates - 1),
                            (act + 2).clamp(max=self.n_gates - 1)], dim=1)  # (N,3)
        gp3 = self.gate_pos[idx3]                     # (N,3,3)
        rel3 = gp3 - state["pos_ned"].unsqueeze(1)
        rel3_body = quat_rotate_inv(state["quat_ned"].unsqueeze(1).expand(-1, 3, -1), rel3)
        gy3 = self.gate_yaw[idx3]
        # ゲート法線方位 − 機体ヨー
        x_body = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(N, 3)
        heading = quat_rotate(state["quat_ned"], x_body)
        body_yaw = torch.atan2(heading[:, 1], heading[:, 0])
        dyaw = gy3 - body_yaw.unsqueeze(1)

        t_since_gate = self.steps_since_gate.float() * C.DT_PHYS
        ep_t = self.episode_steps.float() * C.DT_PHYS
        # 弧長進捗
        arc = torch.tensor(self.course.gate_cum_arc, device=self.device, dtype=torch.float32)
        prog = arc[(self.active_gate - 1).clamp(min=0, max=self.n_gates - 1)] / max(self.course.total_arc, 1e-6)

        priv = torch.cat(
            [
                state["pos_ned"] / C.POS_SCALE,
                state["vel_ned"] / C.V_SCALE,
                quat_to_rot6d(state["quat_ned"]),
                state["omega_frd"] / C.RATE_SCALE,
                (rel3_body / C.GATE_REL_SCALE).reshape(N, 9),
                torch.sin(dyaw), torch.cos(dyaw),
                self.last_action,
                (t_since_gate / self.cfg.no_gate_timeout_s).clamp(0, 1).unsqueeze(1),
                (ep_t / self.cfg.max_episode_s).clamp(0, 1).unsqueeze(1),
                (self.active_gate.float() / self.n_gates).unsqueeze(1),
                prog.unsqueeze(1),
                self.prev_cmd[:, 3:4],
            ],
            dim=1,
        )

        # 真値closeness(報酬用・ノイズなし)
        gp = self.gate_pos[act]
        gn = self.gate_normal[act]
        det_true = self.detector.detect(state["pos_ned"], state["quat_ned"], gp, gn, noise=False)
        closeness = (1.0 - det_true[:, 3]) * det_true[:, 2]

        return obs, priv, closeness

    def set_stage_runtime(self, *, noise_scale: float | None = None, resume_prob: float | None = None,
                          required_gates: int | None = None, speed_finish_w: float | None = None):
        """再構築不要なステージ依存パラメータの更新。"""
        if noise_scale is not None:
            self.cfg.sensors.noise_scale = noise_scale
        if resume_prob is not None:
            self.resume_prob = resume_prob
        if required_gates is not None:
            self.required_gates = required_gates
        if speed_finish_w is not None:
            self.rewards.w.speed_finish = speed_finish_w

    def close(self):
        # Genesisはシーン単位の破棄APIが限定的。プロセス内再構築はscene参照を捨てるだけで
        # メモリが再利用されないことがあるため、再構築はワーカープロセス再起動で行うのが安全。
        self.scene = None
