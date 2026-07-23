"""学習の実行モード。

run_sync  — 1プロセス1GPU(デバッグ/スモーク)。collect→updateを交互に回す。
run_async — 2GPU: collector子プロセス(Genesis+レンダ+ResNet) / learner子プロセス(SAC)。
            親は監視のみ。カリキュラム進級・シーン再構築はcollectorがexit code 3で
            自主終了 → 親がcurriculum.jsonを読んで新stage/seedで再起動する
            (Genesisはプロセス内のシーン再構築が安全でないため)。

GPU割り当てはデバイス名の部分一致(WSL2は列挙順不定)。子プロセスは自分のmain冒頭で
CUDA_VISIBLE_DEVICESを設定する(torch/taichiのCUDA初期化前なら有効)。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _pick_gpu_index(name_substr: str, fallback: int) -> int:
    """デバイス名の部分一致でGPUを選ぶ。nvidia-smi経由なのでCUDAを初期化しない
    (親/呼び出しプロセスでCUDAを初期化するとCUDA_VISIBLE_DEVICESが効かなくなる)。"""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            idx, _, name = line.partition(",")
            if name_substr.lower() in name.strip().lower():
                return int(idx.strip())
    except Exception:
        pass
    return fallback


def _to_cpu(batch: dict) -> dict:
    return {k: v.detach().to("cpu", non_blocking=False) for k, v in batch.items()}


def _curriculum_path(cfg) -> Path:
    return Path(cfg.run.ckpt_dir) / "curriculum.json"


def _load_curriculum_state(cfg) -> tuple[int, int, int]:
    """(stage, seed, transitions) — collectorの再起動/レジューム引き継ぎ。"""
    p = _curriculum_path(cfg)
    if p.exists():
        d = json.loads(p.read_text())
        return (int(d.get("stage", 0)), int(d.get("seed", cfg.env.course_seed)),
                int(d.get("transitions", 0)))
    return cfg.env.stage, cfg.env.course_seed, 0


# ---------------------------------------------------------------- sync mode

def run_sync(cfg, resume: str | None = None, smoke: bool = False):
    # CUDA初期化前にプロセスのGPUを1枚に絞る(Genesis/taichiは可視デバイス0を使う)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_pick_gpu_index(cfg.hw.collector_gpu, 0)))
    import torch

    torch.set_float32_matmul_precision("high")  # Ampere: TF32行列積(数値影響は無視できる)

    from .collector import Collector
    from .learner import Learner
    from .checkpoint import load_checkpoint, find_latest

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    stage, seed, _ = _load_curriculum_state(cfg)

    learner = Learner(cfg, device)
    if resume:
        ck = find_latest(Path(cfg.run.ckpt_dir)) if resume == "auto" else Path(resume)
        if ck and Path(ck).exists():
            payload = load_checkpoint(ck, learner.agent)
            learner.updates = payload["learner_step"]
            learner.transitions = payload["env_transitions"]
            stage = payload.get("stage", stage)
            print(f"[train] resumed from {ck} (updates={learner.updates}, "
                  f"transitions={learner.transitions}, stage={stage})")

    total_target = 4000 if smoke else cfg.sac.total_transitions
    while learner.transitions < total_target:
        collector = Collector(cfg, device, stage, seed)
        collector.transitions = learner.transitions
        collector.load_actor_weights(learner.agent.actor.state_dict())
        collector.warmup()
        last_sync = time.time()
        last_stats = time.time()
        print(f"[train] collector up: stage={stage} seed={seed} envs={collector.N} "
              f"backend={collector.env.rig.backend}")

        rebuild = None
        while rebuild is None and learner.transitions < total_target:
            matured, succ, ep_infos = collector.step()
            if matured is not None:
                learner.add_transitions(matured)
            if succ is not None:
                learner.add_transitions(succ, success=True)
            for info in ep_infos:
                learner.logger.log_episode(learner.transitions, info)

            for _ in range(64):
                if learner.update_once() is None:
                    break
            collector.transitions = learner.transitions

            if time.time() - last_sync > cfg.sac.weight_sync_sec:
                collector.load_actor_weights(learner.agent.actor.state_dict())
                last_sync = time.time()
            if time.time() - last_stats > 30.0:
                stats = learner.logger.flush_episode_stats(learner.transitions)
                learner.maybe_checkpoint(stats)
                if stats:
                    print(f"[train] t={learner.transitions} upd={learner.updates} "
                          f"gates={stats.get('episode/gates_mean', 0):.2f} "
                          f"succ={stats.get('episode/success_rate', 0):.2f} stage={stage}")
                last_stats = time.time()

            rebuild = collector.maybe_curriculum()

        if rebuild:
            stage, seed = rebuild["stage"], rebuild["seed"]
            learner.update_success_ratio(stage)
            _curriculum_path(cfg).write_text(json.dumps(
                {"stage": stage, "seed": seed, "transitions": learner.transitions}))
            print(f"[train] curriculum: stage={stage} seed={seed} "
                  f"(success_rate={rebuild['success_rate']:.2f}) — rebuilding scene")
            # Genesisのシーンはプロセス内で完全破棄できないため、syncモードの再構築は
            # プロセス終了→docker restart(または呼び出し側ループ)に任せる
            learner.maybe_checkpoint(force=True)
            return 3

    learner.maybe_checkpoint(force=True)
    print(f"[train] done: transitions={learner.transitions} updates={learner.updates}")
    return 0


# ---------------------------------------------------------------- async mode

def collector_main(cfg, gpu_index: int, stage: int, seed: int, transitions0: int,
                   q_trans, q_weights, q_metrics, stop_ev, rebuild_ev=None, rank: int = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import torch

    torch.set_float32_matmul_precision("high")

    from .collector import Collector
    from .. import contracts as C

    device = torch.device("cuda", 0)
    collector = Collector(cfg, device, stage, seed, rank=rank)
    collector.transitions = transitions0
    collector.warmup()
    q_metrics.put({"type": "info", "msg": f"collector{rank} up: stage={stage} seed={seed} "
                                          f"envs={collector.N} backend={collector.env.rig.backend}"})
    next_eval = collector.transitions + cfg.run.eval_interval
    eval_frames = []
    eval_left = 0

    while not stop_ev.is_set():
        # rank0がカリキュラム再構築を宣言したら全collectorが自主終了(exit 3)して
        # orchestratorに新stage/seedで再起動してもらう
        if rebuild_ev is not None and rank != 0 and rebuild_ev.is_set():
            break
        deterministic = eval_left > 0
        matured, succ, ep_infos = collector.step(deterministic=deterministic)
        if deterministic and cfg.run.eval_video:
            eval_frames.append(collector._obs["rgb"][0].cpu().numpy())
            eval_left -= 1
            if eval_left == 0 and eval_frames:
                _save_eval_video(cfg, eval_frames, collector.transitions)
                eval_frames = []
        elif rank == 0 and collector.transitions >= next_eval:
            eval_left = int(20 * C.POLICY_HZ)
            next_eval += cfg.run.eval_interval

        if matured is not None:
            with collector.prof.section("queue_put"):
                q_trans.put(("main", _to_cpu(matured)))
        if succ is not None:
            with collector.prof.section("queue_put"):
                q_trans.put(("succ", _to_cpu(succ)))
        for info in ep_infos:
            q_metrics.put({"type": "episode", "transitions": collector.transitions, "info": info})

        # 最新のactor重みへ(latest-only)
        sd = None
        try:
            while True:
                sd = q_weights.get_nowait()
        except Exception:
            pass
        if sd is not None:
            collector.load_actor_weights({k: v.to(device) for k, v in sd.items()})

        ev = collector.maybe_curriculum()
        if ev is not None:  # rank0のみ(他rankはNone固定)
            q_metrics.put({"type": "curriculum", **ev})
            _curriculum_path(cfg).write_text(json.dumps(
                {"stage": ev["stage"], "seed": ev["seed"], "transitions": collector.transitions}))
            if rebuild_ev is not None:
                rebuild_ev.set()  # 他rankにも自主終了を伝える
            break

    raise SystemExit(3 if not stop_ev.is_set() else 0)


def _save_eval_video(cfg, frames, transitions):
    try:
        import imageio
        import numpy as np

        out = Path(cfg.run.ckpt_dir) / "videos" / f"eval_{transitions}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(out), fps=30, codec="libx264", quality=7) as w:
            for f in frames:
                w.append_data(np.asarray(f))
    except Exception as e:  # pragma: no cover
        print(f"[collector] eval video failed: {e}")


def learner_main(cfg, gpu_index: int, resume: str | None, q_trans, q_weights_list, q_metrics,
                 stop_ev):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import queue as pyqueue

    import torch

    torch.set_float32_matmul_precision("high")

    from .checkpoint import find_latest, load_checkpoint
    from .learner import Learner

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    learner = Learner(cfg, device)
    stage, _, _ = _load_curriculum_state(cfg)
    learner.update_success_ratio(stage)
    if resume:
        ck = find_latest(Path(cfg.run.ckpt_dir)) if resume == "auto" else Path(resume)
        if ck and Path(ck).exists():
            payload = load_checkpoint(ck, learner.agent)
            learner.updates = payload["learner_step"]
            learner.transitions = payload["env_transitions"]
            print(f"[learner] resumed from {ck} (updates={learner.updates})")

    last_weights = 0.0
    last_stats = time.time()
    while not stop_ev.is_set() and learner.transitions < cfg.sac.total_transitions:
        drained = 0
        while drained < 128:
            try:
                kind, batch = q_trans.get_nowait()
            except pyqueue.Empty:
                break
            except (FileNotFoundError, ConnectionError, EOFError, OSError) as e:
                # カリキュラム再構築でcollectorが自主終了した直後は、キュー在庫の
                # 共有テンソルFDの提供元(旧collectorのresource_sharer)が消えていて
                # 逆シリアライズに失敗する。在庫を捨てて続行(数バッチの損失は無害)。
                print(f"[learner] dropped stale batch from dead collector ({type(e).__name__})")
                continue
            learner.add_transitions(batch, success=(kind == "succ"))
            drained += 1

        if learner.update_once() is None and drained == 0:
            time.sleep(0.005)

        try:
            while True:
                m = q_metrics.get_nowait()
                if m["type"] == "episode":
                    learner.logger.log_episode(m["transitions"], m["info"])
                elif m["type"] == "curriculum":
                    learner.update_success_ratio(m["stage"])
                    print(f"[learner] curriculum → stage={m['stage']} seed={m['seed']} "
                          f"(success_rate={m['success_rate']:.2f})")
                elif m["type"] == "info":
                    print(f"[learner] {m['msg']}")
        except pyqueue.Empty:
            pass

        now = time.time()
        if now - last_weights > cfg.sac.weight_sync_sec:
            sd = learner.actor_weights_cpu()
            for q_weights in q_weights_list:  # collectorごとに専用キュー(latest-only)
                try:
                    while True:
                        q_weights.get_nowait()
                except pyqueue.Empty:
                    pass
                q_weights.put(sd)
            last_weights = now
        if now - last_stats > 30.0:
            stats = learner.logger.flush_episode_stats(learner.transitions)
            learner.maybe_checkpoint(stats)
            if stats:
                print(f"[learner] t={learner.transitions} upd={learner.updates} "
                      f"gates={stats.get('episode/gates_mean', 0):.2f} "
                      f"succ={stats.get('episode/success_rate', 0):.2f}")
            last_stats = now

    learner.maybe_checkpoint(force=True)


def run_async(cfg, resume: str | None = None):
    # 親プロセスではCUDAを一切初期化しない(子のCUDA_VISIBLE_DEVICESを有効に保つ)
    import torch.multiprocessing as tmp

    col_idx = _pick_gpu_index(cfg.hw.collector_gpu, 0)
    lrn_idx = _pick_gpu_index(cfg.hw.learner_gpu, 1)
    if col_idx == lrn_idx:
        print(f"[orchestrator] 警告: collector/learnerが同一GPU({col_idx})です")
    print(f"[orchestrator] collector=GPU{col_idx}('{cfg.hw.collector_gpu}') "
          f"learner=GPU{lrn_idx}('{cfg.hw.learner_gpu}')")

    n_col = max(1, int(cfg.hw.num_collectors))
    ctx = tmp.get_context("spawn")
    q_trans = ctx.Queue(maxsize=256)
    q_weights_list = [ctx.Queue(maxsize=4) for _ in range(n_col)]
    q_metrics = ctx.Queue(maxsize=2048)
    stop_ev = ctx.Event()
    rebuild_ev = ctx.Event()

    lp = ctx.Process(target=learner_main, name="learner",
                     args=(cfg, lrn_idx, resume, q_trans, q_weights_list, q_metrics, stop_ev))
    lp.start()

    def _join_all(cps, timeout):
        deadline = time.time() + timeout
        for cp in cps:
            cp.join(timeout=max(0.5, deadline - time.time()))
            if cp.is_alive():
                cp.terminate()

    exit_code = 1
    try:
        while True:
            stage, seed, trans0 = _load_curriculum_state(cfg)
            rebuild_ev.clear()
            cps = [ctx.Process(target=collector_main, name=f"collector{r}",
                               args=(cfg, col_idx, stage, seed, trans0, q_trans,
                                     q_weights_list[r], q_metrics, stop_ev, rebuild_ev, r))
                   for r in range(n_col)]
            for cp in cps:
                cp.start()
            while lp.is_alive() and all(cp.is_alive() for cp in cps):
                time.sleep(2.0)
            if not lp.is_alive():
                if lp.exitcode == 0:
                    print("[orchestrator] learner finished (total_transitions reached)")
                    exit_code = 0
                else:
                    print(f"[orchestrator] learner died (code={lp.exitcode})")
                    exit_code = lp.exitcode or 1
                stop_ev.set()
                _join_all(cps, timeout=30)
                break
            # いずれかのcollectorが終了。rebuild(exit 3)なら全員を回収して再起動
            _join_all(cps, timeout=60 if rebuild_ev.is_set() else 10)
            codes = [cp.exitcode for cp in cps]
            if rebuild_ev.is_set() or 3 in codes:
                print(f"[orchestrator] collector rebuild (codes={codes}) — "
                      "restarting with new stage/seed")
                continue
            if 0 in codes:
                exit_code = 0
                break
            print(f"[orchestrator] collector died (codes={codes}) — restarting in 10s")
            time.sleep(10.0)
    finally:
        stop_ev.set()
        for p in (lp,):
            if p.is_alive():
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
    return exit_code
