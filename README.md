# anduril_genesis — Genesisドローンレース RL学習環境

AI Grand Prix Virtual Qualifier(DCLシム)向けの sim-to-sim 事前学習環境。
[Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) (v1.2.3固定) 上に
実コースを模したレース環境を構築し、**特権情報つき非対称SAC** で学習する。

- `src_anduril/` — 既存の本番シムMAVLinkクライアント(変更なし)
- `genesis_rl/` — 本パッケージ(環境・センサー・学習)
- `configs/train.yaml` — 学習設定
- `checkpoints/` — ckpt / TensorBoard / 動画 / progress.png(bind mount)

## 環境の忠実度(本番シム再現)

| 項目 | 実装 |
|---|---|
| アクション | `SET_ATTITUDE_TARGET`互換 (roll_rate, pitch_rate, yaw_rate [rad/s], thrust 0-1)。thrust∈[0.265, 0.40](下限=離陸閾値) |
| 推力モデル | 実測sysid: 比力 A = g·(thrust/0.2742)²、線形ドラッグ c=0.72 |
| 左手系 | cmd_rate_sign / gyro_out_sign = [-1,-1,-1](configで全軸切替可、`frames.py`) |
| タイミング | 物理120Hz / 決定30Hz(カメラ同期) / IMU 40Hz |
| カメラ | 640×360相当 fx=fy=320、**20°上チルト**、学習時は320×180レンダ→`to_resnet()`で224² |
| ゲート検出 | 真値投影+**近距離ほど増えるノイズ**+ドロップアウト+偽検出、**17.3ms遅延**(1フレーム遅れ+ジッタ) |
| 初期状態 | スタートゲート内・中心の0.3m下・**ピッチ-17.8°前傾**・静止 |
| コース | 18ゲート(2.7m/開口1.5m)+青発光リボン+柱+壁天井+急上昇区間、手続き生成+色DR |
| 報酬 | ゲート+50 / 完走+100 / 衝突-20終端 / 接近dense / 平滑化 |

## 使い方

### 0. ビルド
```bash
docker compose build
```

### 1. コースプレビュー(学習前の確認)
```bash
docker compose run --rm preview
# → checkpoints/course_preview.mp4 (FPV+チェイス+俯瞰。スクリプトパイロットが全ゲートを飛ぶ)
```
ホストで直接実行する場合(WSL2はスタブlibcuda回避のためLD_LIBRARY_PATH必須):
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python -m genesis_rl.scripts.preview --out checkpoints/course_preview.mp4
```

### 2. 疎通テスト
```bash
docker compose run --rm train uv run python -m genesis_rl.scripts.train --smoke
```

### 3. 学習(2GPU非同期: collector=3070Ti / learner=4060)
```bash
docker compose up -d train tensorboard
# 監視: http://localhost:6006 (TensorBoard) / checkpoints/progress.png
```
- カリキュラム進級/コース再構築時はコンテナがexit code 3で終了→`restart: unless-stopped`で自動再開(`--resume auto`)
- ckpt: `latest.pt`(10分毎) / `best_gates.pt` / `best_return.pt`
- 評価動画: `checkpoints/videos/eval_*.mp4`(25万遷移ごと、決定的方策)

### 4. Phase 2への引き継ぎ物(本番シムfine-tune用)
- `best_gates.pt` — actor + **TwinQObs**(実観測critic。特権テレメトリなしで持ち込める)
- `checkpoints/genesis_success.pt` — 成功エピソードの特徴空間遷移(RLPD混合用):
```bash
docker compose run --rm train uv run python -m genesis_rl.scripts.export_buffer --ckpt checkpoints/best_gates.pt
```

## カリキュラム(自動進級: trailing 200エピソード成功率)

| Stage | コース | 要求 | ノイズ | 進級閾値 |
|---|---|---|---|---|
| 0 | 直線8ゲート | ゲート1 | ×0.3 | 70% |
| 1 | 緩カーブ18 | 4ゲート | ×0.6 | 70% |
| 2 | フル生成 | 全18 | ×1.0+色DR | 60% |
| 3 | 32シードプール | 全18 | +動力学DR/クラッタ | 50% |
| 4 | 同上 | 全18 | +完走時間ボーナス | 終段 |

## アーキテクチャ

```
COLLECTOR (RTX 3070 Ti)                    LEARNER (RTX 4060)
 GenesisRaceEnv (バッチ, 320x180 FPV)       GPU常駐Replay 1M件 (fp16特徴512d)
 凍結ResNet18 → 512d特徴                    成功バッファ 250k (RLPD混合 0.5→0.25)
 actorコピー(推論) / n-step(3)              SAC: TwinQPriv(特権39d, DroQ) が actor駆動
   │ 遷移 (mp.Queue, CPU tensor)              TwinQObs(実観測) を並行学習(Phase2用)
   └──────────────►                          auto-α / replay-ratioガバナー(≤8)
   ◄──────────────  actor重み (2秒毎)
```

観測契約(`genesis_rl/contracts.py`、Phase 1/2でバイト互換・contract hashでckpt照合):
- actor: `rgb`(遅延FPV) + `vec` 55次元 = [gyro/4, accel/25, 検出u,v,vis,rel_dist,age, 通過one-hot40, last_action4]
- critic特権(Phase 1のみ): 39次元 = [pos, vel, rot6d, ω, 次3ゲート相対+方位, last_action, aux]

## テスト

```bash
uv run pytest genesis_rl/tests/   # 座標系符号・コース不変条件・投影・レイテンシ・n-step
```

## 既知の注意点
- WSL2ホスト直実行時は `LD_LIBRARY_PATH=/usr/lib/wsl/lib`(CUDAスタブ回避)。コンテナ内は不要
- Madronaバッチレンダラ(`gs_madrona`)が入っていれば `env.render.backend: batch` で64env一括レンダ。
  無ければ自動でsequential(≤16envレンダ+残りはカメラオフ)へフォールバック
- ポイントライトは使わない(Genesisラスタライザは8192²シャドウキューブマップを確保しVRAMが溢れる)
- 質量0.9kg/慣性は仮定値(比力モデルなので並進には影響しない。回転はK_rate DR ±60%で吸収)
