#!/usr/bin/env bash
# 8×A100-40GB ゼロ学習ランナー(フォアグラウンド)。
#
# orchestrator はカリキュラム進級/シーン再構築時に exit 3 で自主終了する設計
# (genesis_rl/training/orchestrator.py:455-466)。`docker compose up` なら
# restart: unless-stopped が拾うが `docker compose run` は拾わないので、
# ここで exit 3(再構築) / 42(ストール) を拾って再起動する。
#
# checkpoints/ を空にしてから起動すれば初回はゼロスタート、
# 以降の再起動は --resume auto が latest.pt を拾って正しく継続する。
#
# 使い方:  ./run_scratch.sh 2>&1 | tee -a train_scratch.log
#
# ── num_envs=384 の根拠(2026-07-28 実測 / bench_collector --stage 5) ──
#   num_envs   tps/collector   step      本番ピークVRAM(cache充填後の推定)
#   384         810            474ms     27.5GB(実測)
#   512         ~850           568ms     ~34GB
#   640         ~895           698ms     ~40.5GB → 40GB機ではOOM
# env数を1.67倍にしても+10%しか増えない。ステップ内訳(384env/452ms)が
#   render 160.6ms(36%) / encode 88-122ms(20-27%) / physics 57ms(13%)
# でレンダ律速、かつレンダ画素はenv数に完全比例するため。解像度640x360は
# deploy(640x360 JPEG→224)とドメインを合わせる必須要件で下げられない(configs/train.yaml)。
# +5%のために VRAM 余裕を 12.5GB→6GB へ削るのは多日ランでは割に合わないので384を維持する。
# 参考: 本番7collector実測は744tps/collector、ソロベンチ810tpsとの差8%のみ
#       =キュー/learner同期はボトルネックではない(純粋にcollectorの演算律速)。
set -u
cd "$(dirname "$0")"

fail_streak=0
while true; do
  sudo docker compose run --rm train \
    uv run python -m genesis_rl.scripts.train \
      --config configs/train.yaml \
      --resume auto \
      --set env.num_envs=384 \
      --set env.course_pool=384 \
      --set sac.replay_capacity=2000000 \
      --set sac.success_capacity=400000 \
      --set sac.encoder_chunk=192
  code=$?
  case $code in
    3)  echo "[run] カリキュラム再構築 (exit 3) — 新stageで再開"; fail_streak=0 ;;
    42) echo "[run] ストール検知 (exit 42) — 再起動"; fail_streak=0 ;;
    0)  echo "[run] total_transitions 到達 — 終了"; exit 0 ;;
    130|143) echo "[run] 中断 (exit $code)"; exit $code ;;
    *)  fail_streak=$((fail_streak + 1))
        echo "[run] 異常終了 (exit $code) 連続${fail_streak}回"
        # 連続3回の異常終了は本物のバグ。contract不一致のような無限クラッシュループを防ぐ
        if [ "$fail_streak" -ge 3 ]; then
          echo "[run] 連続3回失敗。ログを確認してください。"
          exit $code
        fi ;;
  esac
  sleep 5
done
