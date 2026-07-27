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
| アクション | `SET_ATTITUDE_TARGET`互換 (roll_rate, pitch_rate, yaw_rate [rad/s], thrust 0-1)。**レート±1 rad/s / thrust∈[0, 0.35]**(a=±1で端。a3=0は0.175=A 0.50gでホバーではない。ホバーは a3=+0.523) |
| 推力モデル | VQ1真値sysid: 比力 A = g·(thrust/0.2665)^1.66、**2次ドラッグ**(水平 c2=0.0427 / 鉛直 0.0186 [1/m]) |
| レート追従 | 軸別 k=[61,64,35] 1/s、指令遅延 5〜10ms。実シムは指令を約2.6 rad/sでクリップ(±1指令なら未到達) |
| 左手系 | cmd_rate_sign / gyro_out_sign = [-1,-1,-1](VQ1真値で確定、`frames.py`) |
| タイミング | 物理120Hz / 決定30Hz(カメラ同期) / IMU 120Hz(実機~100Hz) |
| カメラ | 640×360相当 fx=fy=320、**20°上チルト**、学習時は320×180レンダ→`to_resnet()`で224² |
| ゲート検出 | 真値投影+**近距離ほど増えるノイズ**+ドロップアウト+偽検出、**17.3ms遅延**(1フレーム遅れ+ジッタ) |
| 初期状態 | スタートゲート内・中心の0.3m下・**ピッチ-17.8°前傾**・静止 |
| コース | 18ゲート(2.7m/開口1.5m)。始点→終点へ縦方向に進行し横に蛇行(ターン最大90°・真横ゲートあり)、各ゲートは±12°まで面が傾く。急上昇区間+柱+壁天井、手続き生成+色DR |
| 青パス | 半透明(opacity 0.45)の発光リボンを**アクティブゲートから5ゲート先まで**表示。ゲート通過で通過区間が消え窓が進む(per-env動的)。0.6s周期・区間位相ずれで点滅 |
| 床グロー | 金色グローは**次に行くべきゲート1つの真下だけ**点灯(per-env動的) |
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

## VQ1レガシーシムの真値テレメトリで動力学を較正する(2026-07-27)

VQ1(レガシー)シムはテレメトリが復元されており、**ATTITUDE / LOCAL_POSITION_NED /
ODOMETRY / ACTUATOR / トラック情報(実ゲート幾何)** が全部取れる。VQ2は取れない。
そこでVQ1を *較正台* として使い、自作シムの動力学を真値で合わせる。

```bash
# 1) 実シムを飛ばして真値を録る(Windowsネイティブでも動く。--no-relay必須)
python -m genesis_rl.scripts.fly_dcl --sysid --sysid-plan fit \
    --record-dir runs/vq1_fit --no-video --no-relay --max-sec 200
#    → runs/vq1_fit/{steps,imu,truth,events}.jsonl + track.json
#    プラン: fit=全状態同定1本完結 / sat=レート飽和掃引 / rate,thrust,drag=単項目

# 2) 同定(複数runは結合して1つの回帰にかかる)
python -m genesis_rl.scripts.analyze_truth --dir runs/vq1_fit runs/vq1_sat

# 3) 「自作シムの精度」を数値化(真値状態から1秒回して位置/姿勢のズレを集計)
python -m genesis_rl.scripts.verify_model --dir runs/vq1_fit            # 現行config
python -m genesis_rl.scripts.verify_model --dir runs/vq1_fit --optimize # 軌道誤差最小化

# 4) VQ2へ持ち込む前の橋渡し確認(同じプランをVQ2でも録って比べる)
python -m genesis_rl.scripts.compare_builds --vq1 runs/vq1_fit --vq2 runs/vq2_fit
```

較正で判明した主要な事実(すべて `config.yaml` に反映済み):

| 項目 | 旧(IMU単独同定) | 新(VQ1真値) |
|---|---|---|
| 姿勢テレメトリの規約 | 不明(左手系と推測) | **ODOMETRYクォータニオンはy軸鏡像**。`(w,x,y,z)→(w,-x,y,-z)`で標準FRDになり、真値運動学とIMU accelが**差rms 0.01 m/s²**で一致 |
| 指令→回転の符号 | `[-1,-1,+1]` | **`[-1,-1,-1]`**(yawが逆だった=学習と実機でyawの回る向きが反転) |
| gyro観測の符号 | 学習`[1,1,-1]`×デプロイ`[-1,-1,-1]` | **学習`[-1,-1,-1]`×デプロイ`[1,1,1]`**(生HIGHRES_IMU = −標準FRDω) |
| ドラッグ | 線形 c=0.64(等方) | **body軸別のほぼ純2次**: 水平 c2=0.0427 / 鉛直 c2=0.0186 [1/m](残差 0.564→0.385 m/s²) |
| 推力比力 | A=g(t/0.2694)^1.84 | **A=g(t/0.2665)^1.66**(多項式にしても改善せず=形は正しい) |
| レート追従 | 単一 k=35 | **軸別 [61, 64, 35]**(yawだけτ≈29msで遅い)、遅延5〜10ms |
| レート飽和 | 未把握(±5rad/s前提) | **指令は約2.6rad/sでクリップ**(指令5でも達成2.5)→ `dynamics.rate_max=2.65` で再現。※方策の `action.rate_limits` は別途ユーザー指定で±1rad/s(飽和には当たらない) |
| IMUレート | 60Hz想定 | 飛行中**約100Hz** |
| スポーン姿勢 | pitch −17.8°(推定) | pitch −17.8°(標準FRDで確認)・yaw 180°・NED原点 |

**忠実度(4run結合・770窓・|v| 0〜18m/s・指令±5rad/sを含む)**
真値状態から1秒だけモデルを回した誤差:

| | 位置(中央/p90) | 速度 | 姿勢(中央/p90) |
|---|---|---|---|
| 旧パラメータ | 0.841 / 1.517 m | 1.520 m/s | 0.88° / 6.96° |
| **新較正** | **0.109 / 0.799 m** | **0.210 m/s** | **0.32° / 1.02°** |

`--optimize` は診断用。パラメータ間に縮退があり(遅延↔k_rate など)、中央値だけ下げて
裾を悪化させることがある。**採用値は analyze_truth の直接同定を一次ソースにする**。

### 発進(ピン解除)の仕様 — 2026-07-28 実測

`python -m genesis_rl.scripts.probe_start --no-relay` で条件を切り分けた結果:

| 条件 | 結果 |
|---|---|
| 何も送らない | **動かない**(位置も姿勢も3秒以上まったく変化なし。加速度計は 1.00g=重力反力) |
| thrust 0 / 0.01 / 0.05 / 0.10 | **動かない**。しかも**姿勢指令も一切効かない**(roll 0.5rad/s を3秒入れて0.0°) |
| 推力ランプ 0.08→0.24 | **0.183 で解除**(3回とも一致。遅延を差し引いて閾値 ≈ **0.18**) |
| thrust 0.20 / 0.24 / 0.265 / 0.30 | 動く |
| ARM しない | 動く → **ARM は解除に不要** |
| レースのカウントダウン | **無関係**。開始フラグの2.5〜3.0秒 *前* に解除されるし、フラグ後2秒に送り始めれば その時に解除される |
| 解除後に thrust 0.05 へ落とす | 普通に効く(roll応答 0.477 / 指令0.5、A も低下) → **閾値はスタート専用** |

つまり **発進の引き金は「thrust > 約0.18 の `SET_ATTITUDE_TARGET` を送ること」だけ**。
レース開始フラグは計時の開始点であって、物理的な拘束の解除ではない。
解除の遅れは指令到達から約0.07秒。

自作シム側に同じ挙動を実装済み(`env_physics.pin_start` / `pin_release_thrust` /
`pin_release_delay_s` / `pin_start_thrust`、
[genesis_race_env.py](anduril_genesis/genesis_rl/envs/genesis_race_env.py) の
`_apply_pin` / `_hold_pinned`)。拘束中は指令を無効化してスポーン姿勢へ固定し続けるので、
IMUは「支持された静止状態」として重力反力(1.00g)を出す = 実測と一致する。
途中スポーン(逆カリキュラム)は既に飛んでいる想定なので拘束しない。
状態機械は [tests/test_pin_start.py](anduril_genesis/genesis_rl/tests/test_pin_start.py) で固定。

**発進はルールベース**(方策には学ばせない)。拘束中は方策の出力を使わず
`pin_start_thrust`(=`action.takeoff_thrust`=0.265)を出す固定動作で解除する。実機deploy
([dcl/client.py](anduril_genesis/genesis_rl/dcl/client.py))がピン解除まで方策を呼ばず
`START_THRUST` を出し続けるのと同じ構造で、学習と実機で発進手順が一致する。
`action.thrust_center=0.175` は解除閾値0.18の *すぐ下* だが、発進が固定動作なので影響しない
(`pin_start_thrust` を閾値以下に下げると実シム同様に永久拘束になる — テストで固定済み)。

### VQ1→VQ2 の橋渡し検証(2026-07-28 実施・**同一プラントと確認**)

同じ指令列を両ビルドへ入れ、両方で観測できる信号(HIGHRES_IMU / ACTUATOR)だけを比較した
(`compare_builds.py`。指令列の一致は最大差 0.000000 で確認済み):

| 比較項目 | 結果 | 判定基準 |
|---|---|---|
| 生gyro(位相整合・20s平均) | 差rms 0.004〜0.006 rad/s(信号rms 0.16 の 3%) | — |
| 生accel(同) | 差rms 0.007〜0.064 m/s²(z信号 9.84 の 0.6%) | — |
| 最初の5秒 | gyro 0.0021 rad/s / accel 0.035 m/s² | 0.05 / 0.5 |
| 指令→gyro 定常ゲイン | roll 1.1% / pitch 0.6% / yaw 0.0% 差 | 5%以内 |
| 推力13水準の -accel_z | 全水準 0.1 m/s² 以内(最大 0.096) | 0.2以内 |

→ **VQ1真値で決めた較正値は補正なしでVQ2へ持ち込める**。
両ビルドはリビジョンも同じ(`AIGP_VQ1_3391` と `AIGP_3391`)で、VQ1レガシーは
「旧ビルドの復活」ではなく **現行ビルドのテレメトリを再有効化したもの** だった。
VQ2で違うのは *観測できる情報だけ* で、動力学は同一。

注意点:
- **`sat` プランのrunはパラメータ同定に混ぜない**(analyze_truth が meta の plan を見て自動除外)。
  飽和・大振幅の回復挙動は一次モデルから外れる(角加速度制限の疑い)ため、混ぜると
  cmd_gain 0.95→1.16 / k_rate 61→16 まで崩れる。飽和上限の測定専用。
- `analyze_truth --dir` に複数runを渡すと結合して1つの回帰にかかる。速度域(4m/s と 11m/s)を
  跨いだデータが揃って初めて線形/2次ドラッグが分離できる。
- ビルド間比較では**プラン定数(`deploy.measured_hover` / `rate_plant_gain_rads`)を間で変えない**。
  変えると指令列が変わって比較が無効になる(`compare_builds` の「== 0. 前提確認」で検出できる)。
  過去のrunに合わせたいときは `GENESIS_USER_CONFIG=<当時の値のconfig>` で走らせる。

実コース(`track.json`)も真値で取れる。生成コースとはスケールが大きく違う:

```
実VQ1コース: 6ゲート・全て2.72m角・面の傾きゼロ・間隔23〜39m(中央24m)・全長165m・26m下降
生成コース : 18ゲート・外形2.7m・面±12°・間隔3〜13m・120×50×10mホール
```
`course.course_from_track('runs/*/track.json')` で実コースをそのまま `CourseSpec` にできる
(`describe_track()` で要約表示)。ゲート見かけサイズ=距離が桁で違うので、視覚系の
sim2simギャップは動力学より大きい可能性が高い。

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
