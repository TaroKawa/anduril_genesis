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


def _resolve_gpu(spec, fallback: int) -> int:
    """GPU指定を物理indexへ解決する。
      - 整数 / 数字文字列("0", 3) → そのindexを直接使う(同名GPU多数の環境向け)
      - それ以外の文字列 → デバイス名の部分一致(_pick_gpu_index)
    """
    if isinstance(spec, bool):
        return fallback
    if isinstance(spec, int):
        return spec
    s = str(spec).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    return _pick_gpu_index(s, fallback)


def _collector_gpu_indices(cfg) -> list[int]:
    """collectorを配置するGPU indexのリスト(collector_gpusが空なら collector_gpu 1枚)。"""
    gpus = getattr(cfg.hw, "collector_gpus", ()) or ()
    if gpus:
        return [_resolve_gpu(g, i) for i, g in enumerate(gpus)]
    return [_resolve_gpu(cfg.hw.collector_gpu, 0)]


def _to_cpu(batch: dict) -> dict:
    return {k: v.detach().to("cpu", non_blocking=False) for k, v in batch.items()}


def _put_interruptible(q, item, *evs, timeout: float = 0.5) -> bool:
    """Queueが満杯でも無限ブロックしない put。stop/rebuild イベントが立ったら item を
    捨てて False を返す(呼び出し側はクリーンに break して終了する)。

    ブロッキング put を避けることが要点: カリキュラム再構築で collector を terminate() した際、
    その collector が共有 mp.Queue の内部ロックを保持したまま殺されるとロックが orphan 化し、
    再起動後の全 collector が put で永久ブロック(futex_wait)してデッドロックする。putを
    中断可能にして「必ず自主終了(exit 3)」させることで terminate() 自体を不要にし、これを防ぐ。
    """
    import queue as pyqueue

    while True:
        try:
            q.put(item, timeout=timeout)
            return True
        except pyqueue.Full:
            if any(e is not None and e.is_set() for e in evs):
                return False


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
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_collector_gpu_indices(cfg)[0]))
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
    _put_interruptible(q_metrics, {"type": "info", "msg": f"collector{rank} up: stage={stage} "
                       f"seed={seed} envs={collector.N} backend={collector.env.rig.backend}"}, stop_ev)
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

        aborted = False
        if matured is not None:
            with collector.prof.section("queue_put"):
                aborted = not _put_interruptible(q_trans, ("main", _to_cpu(matured)),
                                                 stop_ev, rebuild_ev)
        if not aborted and succ is not None:
            with collector.prof.section("queue_put"):
                aborted = not _put_interruptible(q_trans, ("succ", _to_cpu(succ)),
                                                 stop_ev, rebuild_ev)
        if not aborted:
            for info in ep_infos:
                if not _put_interruptible(q_metrics,
                                          {"type": "episode", "transitions": collector.transitions,
                                           "info": info}, stop_ev, rebuild_ev):
                    aborted = True
                    break
        if aborted:  # stop/rebuild が立った → ロックを持ったまま殺されないうちにクリーン終了
            break

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
            _put_interruptible(q_metrics, {"type": "curriculum", **ev}, stop_ev)
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
                 stop_ev, progress=None):
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

        # learner(専有GPU)を遊ばせない: replay_ratio_cap に達するまで連続更新する。
        # can_update()==False になると update_once() が None を返して自然に止まる
        # (=UTD上限=過学習防止は replay_ratio_cap が担保。syncモードの range(64) 相当を非同期にも適用)。
        updated = 0
        for _ in range(128):
            if learner.update_once() is None:
                break
            updated += 1
        if updated == 0 and drained == 0:
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

        # orchestratorのストール監視へ進捗を通知(遷移数が一定時間進まなければコンテナ再起動される)
        if progress is not None:
            progress.value = int(learner.transitions)

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

    lrn_idx = _resolve_gpu(cfg.hw.learner_gpu, 1)
    col_indices = _collector_gpu_indices(cfg)
    n_col = max(1, int(cfg.hw.num_collectors))
    # rank r の collector を col_indices へ round-robin で割り当てる
    rank_gpu = [col_indices[r % len(col_indices)] for r in range(n_col)]
    if lrn_idx in rank_gpu:
        print(f"[orchestrator] 警告: learner(GPU{lrn_idx})とcollectorが同一GPUを共有します")
    print(f"[orchestrator] learner=GPU{lrn_idx}  "
          f"collectors={n_col}本 → GPU割当 {rank_gpu}")
    ctx = tmp.get_context("spawn")
    q_trans = ctx.Queue(maxsize=1024)  # 7collector分。溢れるとcollectorのput()がブロックして収集が止まる
    q_weights_list = [ctx.Queue(maxsize=4) for _ in range(n_col)]
    q_metrics = ctx.Queue(maxsize=2048)
    stop_ev = ctx.Event()
    rebuild_ev = ctx.Event()
    progress = ctx.Value("q", 0)  # learnerの累積遷移数。ストール監視用
    stall_timeout = float(os.environ.get("STALL_TIMEOUT_SEC", "600"))  # 進捗が止まってからコンテナ再起動するまで

    lp = ctx.Process(target=learner_main, name="learner",
                     args=(cfg, lrn_idx, resume, q_trans, q_weights_list, q_metrics, stop_ev, progress))
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
                               args=(cfg, rank_gpu[r], stage, seed, trans0, q_trans,
                                     q_weights_list[r], q_metrics, stop_ev, rebuild_ev, r))
                   for r in range(n_col)]
            # 起動をずらす: 全collectorが同時にtaichi/quadrantsのカーネルキャッシュ
            # (/root/.cache/quadrants)へアクセスするとロック競合で全員が再コンパイルし、
            # 再構築のたびにwarmup空白(GPU 0%)が長引く。rank0にまず生成させ、
            # 残りは populated キャッシュを読ませることで warmup を短縮する。
            stagger = float(os.environ.get("COLLECTOR_START_STAGGER_SEC", "12"))
            for i, cp in enumerate(cps):
                cp.start()
                if stagger > 0 and i < len(cps) - 1:
                    time.sleep(stagger)
            # ストール監視: 遷移数(progress)がstall_timeout秒進まなければデッドロック等と判断し、
            # 非0終了→restart:unless-stoppedでコンテナごと再起動(Queue作り直しで確実に復旧)。
            wd_val, wd_t = progress.value, time.time()
            stalled = False
            while lp.is_alive() and all(cp.is_alive() for cp in cps):
                time.sleep(2.0)
                v = progress.value
                if v != wd_val:
                    wd_val, wd_t = v, time.time()
                elif v > 0 and time.time() - wd_t > stall_timeout:
                    stalled = True
                    break
            if stalled:
                print(f"[orchestrator] STALL: 遷移数が t={wd_val} で {stall_timeout:.0f}s 進みません — "
                      "コンテナ再起動で復旧します(exit 42)")
                stop_ev.set()
                _join_all(cps, timeout=30)
                exit_code = 42
                break
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
            # いずれかのcollectorが終了。共有mp.Queueを再構築をまたいで使い回すと、
            # 協調終了(terminate)でキューの内部ロックがorphan化し再起動後の全collectorが
            # futex_wait_queueでデッドロックする。よってプロセス内では再起動せず、
            # orchestratorごと終了→restart:unless-stoppedでコンテナを丸ごと再起動し、
            # 新stage(rank0がcurriculum.jsonへ記録済み)+latest.pt から再開する(堅牢設計)。
            stop_ev.set()  # learnerにcheckpoint+終了を指示
            _join_all(cps, timeout=60 if rebuild_ev.is_set() else 10)
            codes = [cp.exitcode for cp in cps]
            if rebuild_ev.is_set() or 3 in codes:
                print(f"[orchestrator] カリキュラム再構築 (codes={codes}) — "
                      "exit 3 でコンテナ再起動 → 新stageで再開")
                exit_code = 3
            elif 0 in codes:
                exit_code = 0
            else:
                print(f"[orchestrator] collector 終了 (codes={codes}) — exit 1 でコンテナ再起動")
                exit_code = 1
            break
    finally:
        stop_ev.set()
        for p in (lp,):
            if p.is_alive():
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
    return exit_code
