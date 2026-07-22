"""コレクター: Genesis環境を回して遷移を生成する側。

- 凍結ResNetで画像→512次元特徴(replayには特徴のみ)
- バーンイン(ホバーバイアス付きランダム方策) → actorコピー(推論のみ)
- ベクトル化n-stepアセンブラで熟成遷移を出力
- 成功エピソード(ゲート≥success_min_gates)はエピソード単位で再送(RLPD成功バッファ用)
- カリキュラム: 進級/再構築が必要になったら rebuild_requested を立てる
  (シーン再構築はGenesisの制約上プロセス再起動で行う — 呼び出し側が面倒を見る)
- 定期評価: 決定的方策で一定ステップ回し、env0のフレームをmp4に書き出す

sync(同一プロセス)/async(子プロセス)の両方から使う。
"""

from __future__ import annotations

import torch

from .. import contracts as C
from ..config import TrainConfig
from ..curriculum import CurriculumManager, STAGES
from ..envs.genesis_race_env import GenesisRaceEnv
from ..models.actor import SACActor
from ..models.encoder import FrozenDINOv2
from .nstep import NStepAssembler
from .profiling import StepProfiler


def _hist_windows(seq: torch.Tensor, k: int) -> torch.Tensor:
    """(T,D...) の時系列 → (T,k,D...) の履歴窓。エピソード先頭はゼロパディング。

    ランタイムのローリング履歴(境界でゼロ埋め)と同じ窓を、成功エピソード
    キャッシュの列から復元するために使う。
    """
    T = seq.shape[0]
    out = seq.new_zeros(T, k, *seq.shape[1:])
    for j in range(k):
        out[j:, k - 1 - j] = seq[: T - j] if j else seq
    return out


class Collector:
    def __init__(self, cfg: TrainConfig, device: torch.device, stage: int, course_seed: int):
        self.cfg = cfg
        self.device = device
        spec = STAGES[min(stage, len(STAGES) - 1)]
        env_cfg = cfg.env
        env_cfg.stage = spec.course_stage
        env_cfg.color_dr = spec.color_dr
        env_cfg.clutter = spec.clutter
        self.env = GenesisRaceEnv(env_cfg, num_envs=env_cfg.num_envs, course_seed=course_seed,
                                  stage=spec.course_stage)
        self.curriculum = CurriculumManager(cfg.curriculum, start_stage=stage)
        # collectorはプロセス再起動で作り直されるため、シード連番を現seedから復元する
        # (これがないとnext_course_seedが毎回base+1を返し、同一コースを再構築し続ける)
        self.curriculum.seed_counter = max(0, course_seed - cfg.env.course_seed)
        self.env.set_stage_runtime(noise_scale=spec.noise_scale,
                                   resume_prob=self.curriculum.resume_prob_now(),
                                   required_gates=min(spec.required_gates, self.env.n_gates - 1),
                                   speed_finish_w=spec.speed_finish_w)
        self.N = self.env.num_envs
        self.encoder = FrozenDINOv2(bf16=cfg.sac.encoder_bf16).to(self.env.device).eval()
        self._blank_feat = None  # ゼロ画像のエンコーダ特徴(定数)のキャッシュ
        self.actor = SACActor(hidden=cfg.sac.hidden).to(self.env.device).eval()
        self.nstep = NStepAssembler(self.N, cfg.sac.n_step, cfg.sac.gamma,
                                    (C.HIST_K, C.VEC_DIM), C.PRIV_DIM, C.ACTION_DIM,
                                    (C.HIST_K, C.FEAT_DIM), self.env.device)
        self.transitions = 0
        self._obs = None
        self._priv = None
        self._feat = None
        self.rebuild_requested = False
        self.prof = StepProfiler(cfg.run.profile, self.env.device)
        if self.prof.enabled:
            self.env.rig.render = self.prof.wrap("render", self.env.rig.render)
            self.env.scene.step = self.prof.wrap("physics", self.env.scene.step)

    # --- 内部 ---

    def _encode(self, rgb_u8: torch.Tensor) -> torch.Tensor:
        """レンダ対象外env(画像が全ゼロ)はResNetを通さず定数特徴で埋める。

        sequentialバックエンドではnum_envsのうち16envしか実画像を持たないため、
        全envをResNetに通すのは大半が同一のゼロ画像の再計算になる。出力は
        全件エンコードと厳密に一致する(ゼロ画像→定数ベクトル)。
        """
        nz = rgb_u8.flatten(1).any(dim=1)
        if nz.all():
            return self.encoder(C.to_resnet(rgb_u8))
        if self._blank_feat is None:
            blank = torch.zeros(1, *rgb_u8.shape[1:], dtype=rgb_u8.dtype, device=rgb_u8.device)
            self._blank_feat = self.encoder(C.to_resnet(blank))[0]
        out = self._blank_feat.expand(rgb_u8.shape[0], -1).clone()
        if nz.any():
            out[nz] = self.encoder(C.to_resnet(rgb_u8[nz]))
        return out

    def warmup(self):
        self._obs, self._priv = self.env.reset()
        self._feat = self._encode(self._obs["rgb"])
        dev = self.env.device
        self._feat_hist = torch.zeros(self.N, C.HIST_K, C.FEAT_DIM, device=dev)
        self._vec_hist = torch.zeros(self.N, C.HIST_K, C.VEC_DIM, device=dev)
        self._feat_hist[:, -1] = self._feat
        self._vec_hist[:, -1] = self._obs["vec"]

    def policy_action(self, deterministic: bool = False) -> torch.Tensor:
        if self.transitions < self.cfg.sac.burn_in_steps and not deterministic:
            # ホバーバイアス付きランダム(a3=0 → thrust 0.3325 = 緩上昇)
            return (torch.randn(self.N, C.ACTION_DIM, device=self.env.device) * 0.3).clamp(-1, 1)
        with torch.no_grad():
            return self.actor.act(self._feat_hist, self._vec_hist, deterministic=deterministic)

    def step(self, deterministic: bool = False):
        """1ベクトルステップ。returns (matured_batch|None, success_batch|None, ep_infos)"""
        with self.prof.section("actor"):
            a = self.policy_action(deterministic)
        with self.prof.section("env_step"):
            obs2, priv2, rew, done, info = self.env.step(a)
        with self.prof.section("encode"):
            feat2 = self._encode(obs2["rgb"])

        nfeat, nvec, npriv = feat2, obs2["vec"], priv2
        if done.any():
            idx = info["done_idx"]
            with self.prof.section("encode"):
                f_final = self._encode(info["final_obs"]["rgb"])
            nfeat = feat2.clone()
            nvec = obs2["vec"].clone()
            npriv = priv2.clone()
            nfeat[idx] = f_final
            nvec[idx] = info["final_obs"]["vec"]
            npriv[idx] = info["final_priv"]

        terminal = done & ~info["time_outs"]
        # 遷移用のnext履歴窓。doneのenvはリセット前の履歴+終端観測(final swap済みのnfeat/nvec)
        with self.prof.section("nstep"):
            nfeat_hist = torch.cat([self._feat_hist[:, 1:], nfeat.unsqueeze(1)], dim=1)
            nvec_hist = torch.cat([self._vec_hist[:, 1:], nvec.unsqueeze(1)], dim=1)
            matured = self.nstep.push(self._feat_hist, self._vec_hist, self._priv, a,
                                      rew, done, terminal, nfeat_hist, nvec_hist, npriv)

        # 成功エピソード抽出用に直近の全遷移をバッチ列でキャッシュ(単ステップで保持し、
        # 抽出時に_hist_windowsで窓を復元してメモリを節約する)
        with self.prof.section("cache"):
            self._cache_step(a, rew, done, terminal, nfeat, nvec, npriv)
            success_batch, ep_infos = self._flush_episodes(done, info)

        # ストリーム履歴を前進(次ステップの「現在」観測。doneのenvはリセット後観測)
        self._feat_hist = torch.cat([self._feat_hist[:, 1:], feat2.unsqueeze(1)], dim=1)
        self._vec_hist = torch.cat([self._vec_hist[:, 1:], obs2["vec"].unsqueeze(1)], dim=1)
        if done.any():
            idx = info["done_idx"]
            self._feat_hist[idx, :-1] = 0.0  # エピソード境界: 過去をゼロ埋め(=パディング)
            self._vec_hist[idx, :-1] = 0.0

        self._obs, self._priv, self._feat = obs2, priv2, feat2
        self.transitions += self.N
        self.prof.tick(self.N)
        return matured, success_batch, ep_infos

    # --- 成功エピソードキャッシュ(バッチ列として保持、doneでenv別に切り出し) ---

    def _cache_step(self, a, rew, done, terminal, nfeat, nvec, npriv):
        if not hasattr(self, "_cache"):
            self._cache = []
        self._cache.append({
            "feat": self._feat, "vec": self._obs["vec"], "priv": self._priv, "act": a,
            "rew": rew, "done": done.float(), "terminal": terminal.float(),
            "nfeat": nfeat, "nvec": nvec, "npriv": npriv,
        })
        # エピソード上限(60s@30Hz=1800) + n分あれば十分
        max_len = int(self.cfg.env.max_episode_s * C.POLICY_HZ) + self.cfg.sac.n_step + 2
        if len(self._cache) > max_len:
            self._cache.pop(0)
            self._cache_offset = getattr(self, "_cache_offset", 0) + 1

    def _flush_episodes(self, done: torch.Tensor, info: dict):
        """doneしたenvの成功エピソードを1-step遷移列として返し、統計をep_infosへ。"""
        ep_infos = []
        if not done.any():
            return None, ep_infos
        idx = info["done_idx"].tolist()
        gates = info["done_gates"].tolist()
        succ = info["done_success"].tolist()
        spawn_g = info["done_spawn_gate"].tolist()
        spawn_d = info["done_spawn_dist_g1"].tolist()
        for i, g, s, sg, sd in zip(idx, gates, succ, spawn_g, spawn_d):
            ep_infos.append({"gates": int(g), "success": bool(s),
                             "collision": bool(info["collision"][i]),
                             "finish": bool(info["finish"][i]),
                             "spawn_gate": int(sg), "spawn_dist_g1": float(sd),
                             "resume_prob": float(self.env.resume_prob),
                             "stage": int(self.curriculum.stage),
                             "episode_sums": info.get("episode", {})})
        self.curriculum.record_episodes([bool(s) for s in succ])
        # 逆カリキュラム: 成功率の変化に追従して途中スポーン確率を更新
        self.env.set_stage_runtime(resume_prob=self.curriculum.resume_prob_now())

        # 成功エピソード(ゲート数はenv.required_gates基準のsuccessフラグで判定済み)の
        # 遷移をキャッシュから抽出(1-step、gpow=γ)。feat/vecは_hist_windowsで
        # ランタイムと同一の履歴窓(エピソード先頭ゼロパディング)を復元する。
        out = None
        succ_envs = [i for i, s, g in zip(idx, succ, gates)
                     if s or int(g) >= self.cfg.sac.success_min_gates]
        if succ_envs and hasattr(self, "_cache"):
            gamma = self.cfg.sac.gamma
            episodes = []
            for env_i in succ_envs:
                # このenvの現エピソード開始位置: 直近のdone(自分)より前の区間を遡る
                rows = []
                for t in range(len(self._cache) - 1, -1, -1):
                    step_t = self._cache[t]
                    if t < len(self._cache) - 1 and step_t["done"][env_i] > 0.5:
                        break  # 前のエピソードに到達
                    rows.append(t)
                rows.reverse()
                seq = lambda k: torch.stack([self._cache[t][k][env_i] for t in rows])
                fh = _hist_windows(seq("feat"), C.HIST_K)
                vh = _hist_windows(seq("vec"), C.HIST_K)
                terminal = seq("terminal")
                episodes.append({
                    "feat": fh, "vec": vh,
                    "nfeat": torch.cat([fh[:, 1:], seq("nfeat").unsqueeze(1)], dim=1),
                    "nvec": torch.cat([vh[:, 1:], seq("nvec").unsqueeze(1)], dim=1),
                    "priv": seq("priv"), "npriv": seq("npriv"),
                    "act": seq("act"), "rew": seq("rew"),
                    "gpow": gamma * (1.0 - terminal), "done": terminal,
                })
            if episodes:
                out = {k: torch.cat([ep[k] for ep in episodes], dim=0) for k in episodes[0]}
        # done envのキャッシュ行は次エピソードと混ざるが、開始位置検出(done flag)で区切れる
        return out, ep_infos

    # --- カリキュラム ---

    def maybe_curriculum(self) -> dict | None:
        """進級 or 再構築が必要なら {'stage':…, 'seed':…} を返す(プロセス再起動を要求)。"""
        advanced = self.curriculum.maybe_advance()
        if advanced or self.curriculum.needs_rebuild():
            seed = self.curriculum.next_course_seed(self.cfg.env.course_seed)
            return {"stage": self.curriculum.stage, "seed": seed,
                    "success_rate": self.curriculum.success_rate(), "advanced": advanced}
        return None

    def load_actor_weights(self, sd: dict):
        self.actor.load_state_dict(sd)
