# -*- coding: utf-8 -*-
"""本番シム(DCL / AI Grand Prix Virtual Qualifier)推論クライアント。

anduril_genesis 単体で完結する(Spakona_PyAIPilotExample への依存なし)。
シムの通信プロトコル(Spakonaサンプルから最小移植):
  - MAVLink UDP 14550 (udpin): HIGHRES_IMU受信 / ENCAPSULATED_DATA(レース状態) /
    SET_ATTITUDE_TARGET(レートモード)送信 / ARM / シムリセット(cmd 31000)
  - 映像 UDP 5600: チャンク分割JPEG (header "<IHHIIQ") → 640x360 BGR

観測は学習契約(genesis_rl/contracts.py)と同一に組み立てる:
  gyro/accel: HIGHRES_IMU生値 / RATE_SCALE, ACCEL_SCALE
  gate 5次元: YOLOX-x でゲートbbox検出(--gate-detector hsv でHSVフォールバック)
              → [u_n, v_n, vis, rel_dist, age_n]
              rel_dist = 1 - bbox面積/GATE_AREA_MAX (YOLOX互換規約)
  one-hot   : レース状態の active_gate_index(=通過済みゲート数)
  last_action / 画像特徴: LoadedPolicy(scripts/eval_video.py と共用、新旧actor自動判別)

衝突はSpakonaと同じ比力スパイク(|a|>40m/s²、ピン解除後)で検知し、シムを
リセットして再アーム・履歴クリアで続行する。
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np

MAVLINK_CMD_SIM_RESET = 31000
ENCAPSULATED_RACE_STATUS_MSG_ID = 1
COLLISION_ACCEL_MPS2 = 40.0
# 本番HIGHRES_IMUのgyroは学習側の観測規約(frames.ProductionSigns.gyro_out_sign=-1、
# Spakona estimatorのgyro_sign=[-1,-1,-1]と同一)に対して符号反転している。生値のまま
# vecに入れると方策のレート帰還が正帰還になり発進直後から転がる(2026-07-23 実機ログで確定:
# 指令→gyro応答比が全軸-2.5→-1倍で+2.5=負帰還)。ここで学習規約へ揃える。
GYRO_OBS_SIGN = (-1.0, -1.0, -1.0)
ACCEL_OBS_SIGN = (1.0, 1.0, 1.0)   # accelは整合(実機ログで level rest≈(0,0,-9.81)を確認)
# 実シムのレートループは指令の約2.5倍の角速度を出す(振幅0.02〜0.4で線形、sysid v2で確定:
# runs/sysid2_rate_0723 / sysid2_yaw_0723)。旧ckpt(Genesisが指令=達成レート≈1倍のプラント
# で学習)はデプロイ側で送信レートをこのゲインで割って達成レートを方策の意図値へ揃える。
# 新ckpt(config.py drone.cmd_gainでプラントごと模擬して学習)は除算不要 —
# GenesisPilotがckptのcfgスナップショットから自動判別する。
RATE_CMD_GAIN = (2.44, 2.46, 2.18)
# rev3390: SET_ATTITUDE_TARGET.type_mask の拡張bit16を立てると、シムは body_rates を
# 正しい物理 rad/s として解釈する(公式サンプル check/controller.py で判明)。立てない
# レガシー解釈こそが上記 ~2.5倍ゲインの正体。
# 【bit16付き開ループ同定で確定 2026-07-23 runs/sysid_bit16_0723】3軸すべてで:
#   ・大きさ: レート定常ゲイン |達成/指令| = roll 0.99 / pitch 0.99 / yaw 0.89(≈1.0、線形)。
#     → レガシーの ~2.5倍は解釈bitの欠落が原因。bit16でほぼ1:1になる。
#   ・符号: 生gyroが指令と *同符号* になる(指令-0.3→生gyro-0.30)。レガシーは逆符号だった
#     ため、既存の観測符号 GYRO_OBS_SIGN=-1 / EnvConfig.signs_* はレガシー挙動に合わせて
#     逆算されている。→ bit16 を採用すると符号系も破綻するので **ドロップイン不可**。
#     採用にはGYRO_OBS_SIGN→+1・signs再導出・cmd_gain→(1,1,0.89)相当・**fresh再学習**が必要。
# よって既定は False(現行のレガシー較正のまま=無回帰)。True は上記一式を揃えた
# 新ckpt専用のオプトインとして残す。送信スケールは train_cmd_gain/deploy_plant_gain で一意
# (GenesisPilot参照。False時は従来の RATE_CMD_GAIN 除算と数値的に等価)。
DCL_BODY_RATES_RADS_BIT = 16
USE_RAD_PER_SEC_BODY_RATES = True   # 【bit16 完全移行 2026-07-23】符号再導出(config.signs_cmd
                                    # →(-1,-1,+1))+ cmd_gain→(1,1,0.89) + RATE_LIMITS更新 +
                                    # fresh再学習 と一式で採用。旧ckptは False に戻して使用。
# bit16 ON時の実シムのレート・プラントゲイン|達成/指令|(runs/sysid_bit16_0723)。
# 送信スケール(train_cmd_gain/deploy_gain)の分母に使う。ckptの cmd_gain がこれと一致すれば
# 送信スケール=1.0(そのまま送る)。OFF時はレガシーの RATE_CMD_GAIN を使う。
RATE_PLANT_GAIN_RADS = (1.0, 1.0, 0.89)
HOVER_THRUST = 0.2742          # contracts.HOVER_THRUST(フェイルセーフ用)
CONTROL_HZ = 250.0             # コマンド送信レート(シム仕様)
VIDEO_W, VIDEO_H = 640, 360


# ---------------------------------------------------------------- MAVLink

class MavlinkIO:
    """MAVLink受信スレッド + コマンド送信。shared: imu / race / collision。"""

    def __init__(self, shared: dict, ip: str = "127.0.0.1", port: int = 14550):
        from pymavlink import mavutil

        self.mavutil = mavutil
        self.shared = shared
        self.boot_ms = int(time.time() * 1000)
        print(f"MAVLink: waiting for heartbeat on udpin:{ip}:{port} ...", flush=True)
        self.conn = mavutil.mavlink_connection(f"udpin:{ip}:{port}")
        self.conn.wait_heartbeat()
        print(f"MAVLink: connected to system {self.conn.target_system}", flush=True)

        self._last_hb_ms = 0
        self._last_collision_t = 0.0
        # レース開始判定の残骸検出(Spakona mavlink_rx.py の移植)
        self._initial_race_start = None
        self._stale_initial_start = False
        self._last_sim_boot_ms = None
        self.is_running = True
        self.thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.thread.start()

    # --- 受信 ---

    def _rx_loop(self):
        while self.is_running:
            try:
                msg = self.conn.recv_match(blocking=False)
            except ConnectionResetError:
                print("WARNING: MAVLink connection reset", flush=True)
                return
            if msg is None:
                time.sleep(0.001)
                continue
            t = msg.get_type()
            if t == "HIGHRES_IMU":
                self._on_imu(msg)
            elif t == "ENCAPSULATED_DATA":
                raw = bytes(msg.data)
                if raw and raw[0] == ENCAPSULATED_RACE_STATUS_MSG_ID:
                    self._on_race_status(raw)

    def _on_imu(self, msg):
        accel = (msg.xacc, msg.yacc, msg.zacc)
        gyro = (msg.xgyro, msg.ygyro, msg.zgyro)
        self.shared["imu"] = {"accel": accel, "gyro": gyro, "t": msg.time_usec * 1e-6}
        log = self.shared.get("imu_log")
        if log is not None:   # 高レートIMUログ(sysid解析用。dequeはGILでスレッド安全)
            log.append({"t_rx_wall": time.time(), "t_sim": msg.time_usec * 1e-6,
                        "gyro": gyro, "accel": accel})
        race = self.shared.get("race") or {}
        # 発進グレース: ピン解除直後は拘束反力の残り(~17g)が乗るため衝突判定しない
        released_at = race.get("released_at")
        in_grace = released_at is None or (time.time() - released_at) < 1.5
        if race.get("pin_released") and not in_grace:
            a = (accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2) ** 0.5
            if a > COLLISION_ACCEL_MPS2 and time.time() - self._last_collision_t > 5.0:
                self._last_collision_t = time.time()
                print(f"COLLISION detected: |a|={a:.0f} m/s^2", flush=True)
                self.shared["collision"] = {"t_wall": time.time(), "handled": False}

    def _on_race_status(self, raw):
        (_, sim_boot_ms, race_start_ms, race_finish_ns,
         active_gate_index, _) = struct.unpack_from("<BQqqIq", raw)

        # シム時計の巻き戻り = リセット発生 → 残骸判定をやり直す
        if self._last_sim_boot_ms is not None and sim_boot_ms < self._last_sim_boot_ms - 5000:
            self._initial_race_start = None
            self._stale_initial_start = False
        self._last_sim_boot_ms = sim_boot_ms

        # 初回から開始時刻が過去 = 前回runの残骸。値が変化するまで信用しない
        if self._initial_race_start is None:
            self._initial_race_start = race_start_ms
            self._stale_initial_start = (race_start_ms is not None and race_start_ms > 0
                                         and sim_boot_ms >= race_start_ms)
        start_trusted = (race_start_ms is not None and race_start_ms > 0
                         and (not self._stale_initial_start
                              or race_start_ms != self._initial_race_start))
        pin_released = start_trusted and sim_boot_ms >= race_start_ms
        start_pending = start_trusted and sim_boot_ms < race_start_ms
        finished = start_trusted and race_finish_ns is not None and race_finish_ns > 0
        prev = self.shared.get("race") or {}
        released_at = prev.get("released_at")
        if pin_released and not prev.get("pin_released"):
            released_at = time.time()   # 発進の立ち上がり(衝突グレースの起点)
        elif not pin_released:
            released_at = None
        self.shared["race"] = {
            "active_gate_index": int(active_gate_index),
            "pin_released": bool(pin_released),
            "start_pending": bool(start_pending),
            "race_finished": bool(finished),
            "released_at": released_at,
            "t_wall": time.time(),
        }

    # --- 送信 ---

    def send_rates(self, roll_rate: float, pitch_rate: float, yaw_rate: float, thrust: float):
        m = self.mavutil.mavlink
        # 姿勢無視=レート制御。rad/s bit(rev3390)を立てると body_rates を物理 rad/s と解釈。
        type_mask = (m.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
                     | (DCL_BODY_RATES_RADS_BIT if USE_RAD_PER_SEC_BODY_RATES else 0))
        self.conn.mav.set_attitude_target_send(
            int(time.time() * 1000) - self.boot_ms,
            self.conn.target_system, self.conn.target_component,
            type_mask,
            [1, 0, 0, 0], roll_rate, pitch_rate, yaw_rate, thrust)

    def heartbeat_if_due(self):
        now_ms = int(time.time() * 1000)
        if now_ms - self._last_hb_ms < 500:  # 2Hz
            return
        self._last_hb_ms = now_ms
        m = self.mavutil.mavlink
        self.conn.mav.heartbeat_send(m.MAV_TYPE_GCS, m.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def arm(self):
        m = self.mavutil.mavlink
        self.conn.mav.command_long_send(self.conn.target_system, self.conn.target_component,
                                        m.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)

    def sim_reset(self):
        self.conn.mav.command_long_send(self.conn.target_system, self.conn.target_component,
                                        MAVLINK_CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0)

    def close(self):
        self.is_running = False


# ---------------------------------------------------------------- 映像 + ゲート検出

def gate_detect_hsv(img_bgr: np.ndarray, gate_area_max: float | None = None) -> dict:
    """HSVオレンジ抽出でゲートbboxを検出(YOLOX代替、Spakona gate_color.py移植)。

    最大連結成分のbboxを使う(通常は最も近い=アクティブゲート)。
    returns {visible, center(px,py∈[0,1]), rel_dist}
    """
    import cv2

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 50), (25, 255, 255))       # 赤〜オレンジ帯
    m2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))    # 赤の折り返し帯
    mask = m1 | m2
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return {"visible": 0, "center": (0.5, 0.5), "rel_dist": 1.0}
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 12:  # 数px未満はノイズ
        return {"visible": 0, "center": (0.5, 0.5), "rel_dist": 1.0}
    w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[i]
    H, W = img_bgr.shape[:2]
    from ..contracts import GATE_AREA_MAX
    gam = GATE_AREA_MAX if gate_area_max is None else gate_area_max
    rel = float(np.clip(1.0 - (w * h) / gam, 0.0, 1.0))
    return {"visible": 1, "center": (float(cx / W), float(cy / H)), "rel_dist": rel}


class VideoRX:
    """映像UDP受信スレッド。チャンク分割JPEGを組み立て → obs_rgb/obs_gate を publish。

    obs_rgb: 224x224 RGB uint8(学習時のto_resnet入力と同じ全面リサイズ)。
    録画: 生フレーム+HUDを mp4 へ逐次書き出し(out指定時)。
    """

    def __init__(self, shared: dict, ip: str = "0.0.0.0", port: int = 5600,
                 out_mp4: str | None = None, gate_detect_fn=None):
        self.shared = shared
        self.ip, self.port = ip, port
        self._writer = None
        self._out = out_mp4
        # ゲート検出器: 既定は YOLOX(run() が注入)。None のときは HSV フォールバック。
        self._detect = gate_detect_fn or gate_detect_hsv
        self.frames = 0
        self.packets = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.5)
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        import cv2

        header_fmt = "<IHHIIQ"
        header_sz = struct.calcsize(header_fmt)
        frames: dict = {}
        sock = self.sock
        print(f"Video: listening on udp:{self.ip}:{self.port} ...", flush=True)

        while self.is_running:
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            self.packets += 1
            fid, cid, total, jpeg_size, _, _ = struct.unpack(header_fmt, packet[:header_sz])
            f = frames.setdefault(fid, {"chunks": {}, "total": total})
            f["chunks"][cid] = packet[header_sz:]
            if len(f["chunks"]) < f["total"]:
                continue
            buf = bytearray()
            ok = True
            for i in range(f["total"]):
                if i not in f["chunks"]:
                    ok = False
                    break
                buf.extend(f["chunks"][i])
            del frames[fid]
            if not ok:
                continue
            img = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            self._process(img)

    def _process(self, img_bgr):
        import cv2

        now = time.time()
        det = self._detect(img_bgr)
        det["t_wall"] = now
        self.shared["obs_gate"] = det
        rgb224 = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        self.shared["obs_rgb"] = {"t_wall": now, "rgb": rgb224}
        self.frames += 1

        if self._out:
            if self._writer is None:
                import imageio
                self._writer = imageio.get_writer(self._out, fps=30, codec="libx264", quality=7)
            hud = img_bgr.copy()
            race = self.shared.get("race") or {}
            cmd = self.shared.get("last_cmd")
            if det["visible"]:
                u, v = int(det["center"][0] * hud.shape[1]), int(det["center"][1] * hud.shape[0])
                cv2.drawMarker(hud, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            txt = f"gate={race.get('active_gate_index', '-')} pin={int(bool(race.get('pin_released')))}"
            if cmd is not None:
                txt += f" thr={cmd[3]:.3f}"
            cv2.putText(hud, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self._writer.append_data(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB))

    def close(self):
        """受信を止め、動画を確実にクローズする(呼び出し側スレッドで行う)。"""
        self.is_running = False
        self.thread.join(timeout=3.0)
        try:
            self.sock.close()
        except OSError:
            pass
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# ---------------------------------------------------------------- パイロット

class ScriptedRatePilot:
    """方策を使わず既知の指令列を送る開ループ同定用パイロット(sysid v2)。

    プラント(実シムの動力学)を方策から切り離して測る。schedule要素は
    (継続秒, roll, pitch, yaw, thrust)。prefixを1回再生後、loopを繰り返す。
    解析は scripts/analyze_sysid.py(--record-dir の steps.jsonl + imu.jsonl)。

    プラン:
      rate:   3軸×複数振幅の対称レートパルス → ゲインの線形性/飽和・時定数・遅延
      yaw:    yaw微小振幅の掃引 → 不感帯/小信号ゲインの解像(rateプランで非線形を確認済み)
      thrust: 推力階段(上昇/下降の対称ペアでv_z蓄積を抑制)→ 比力曲線 A(thrust)
      drag:   前傾のまま加速→水平化→惰性減速 → 線形ドラッグ c(体x比力の指数減衰)

    幾何の前提(パルス設計にのみ使用、同定結果には影響しない):
      実測レートゲイン≈2.5(runs/sysid_0723_0321)、スポーン前傾-17.8°。
      水平化(機首上げ)は pitch=+0.1 を 1.24s(≈17.8°)。
      ※DCL実機で確認済み(runs/sysid_rate_0723): pitch指令-0.1では前傾が深くなり
        前進加速が継続する(体x比力が単調増加)→ 機首上げは+側。
    """

    HOVER = 0.2742          # contracts.HOVER_THRUST(A=g想定)
    TILT_HOVER = 0.281      # 17.8°前傾時に鉛直成分がgになる推力(=hover/√cos17.8°)
    LEVEL = (1.24, 0.0, +0.1, 0.0, 0.281)   # 前傾-17.8°→水平(達成レート≈0.25rad/s)

    def __init__(self, plan: str = "rate"):
        from .. import contracts as C

        self.C = C
        self.plan = plan
        self.last_action = np.zeros(C.ACTION_DIM, dtype=np.float32)  # 記録用(方策なしなので0)
        self.last_vec = np.zeros(C.VEC_DIM, dtype=np.float32)
        self.t0 = None
        h = self.HOVER
        climb = (1.0, 0.0, 0.0, 0.0, 0.29)      # 高度マージン確保(+1m弱)

        if plan == "rate":
            self.prefix = [climb, self.LEVEL, (0.7, 0, 0, 0, h)]
            body = []
            # 振幅を上げるほどパルスを短く(大バンクからの復帰を安全に)。
            # 振幅を外側ループに: 途中で墜落しても全軸の小振幅データが先に揃う。
            amps = [(0.05, 0.8), (0.1, 0.8), (0.2, 0.6), (0.4, 0.4)]
            for amp, dur in amps:
                for ax in range(3):
                    for sgn in (+1.0, -1.0):
                        rpy = [0.0, 0.0, 0.0]
                        rpy[ax] = sgn * amp
                        body.append((dur, *rpy, h))
                        body.append((0.5, 0.0, 0.0, 0.0, h))
            self.loop = body
        elif plan == "roll":
            # roll物理方向の判定: バンク~10°を2.5s保持 → 横比力 f_y = -c·v_y の符号が
            # バンク方向を示す(+rollでf_y<0なら「+指令=右バンク」)。左右対称に実施。
            self.prefix = [climb, self.LEVEL, (0.7, 0, 0, 0, h)]
            th = 0.277   # 10°バンク時の高度維持推力
            self.loop = [
                (0.9, +0.08, 0.0, 0.0, h), (2.5, 0.0, 0.0, 0.0, th),
                (0.9, -0.08, 0.0, 0.0, h), (1.5, 0.0, 0.0, 0.0, h),
                (0.9, -0.08, 0.0, 0.0, h), (2.5, 0.0, 0.0, 0.0, th),
                (0.9, +0.08, 0.0, 0.0, h), (1.5, 0.0, 0.0, 0.0, h),
            ]
        elif plan == "yaw":
            self.prefix = [climb, self.LEVEL, (0.7, 0, 0, 0, h)]
            body = []
            for amp in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25, 0.40):
                for sgn in (+1.0, -1.0):
                    body.append((0.9, 0.0, 0.0, sgn * amp, h))
                    body.append((0.4, 0.0, 0.0, 0.0, h))
            self.loop = body
        elif plan == "thrust":
            self.prefix = [climb, self.LEVEL, (0.7, 0, 0, 0, h)]
            # (上げ, 対称の下げ)ペア: 下げ側は実測曲線 A≈g*(t/0.269)^1.64 で净v_z≈0に
            # なるよう選定(初回計測 runs/sysid2_thrust_0723 のフィット)。想定が外れても
            # ドリフトするだけで、解析側はv_z推定で補正する。方策レンジ[0.265,0.40]重視。
            pairs = [(0.30, 0.235), (0.32, 0.211), (0.34, 0.183), (0.37, 0.132),
                     (0.25, 0.297), (0.26, 0.288)]
            body = [(0.8, 0.0, 0.0, 0.0, h)]
            for up, dn in pairs:
                body += [(0.6, 0.0, 0.0, 0.0, up), (0.6, 0.0, 0.0, 0.0, dn),
                         (0.7, 0.0, 0.0, 0.0, h)]
            body += [(0.5, 0.0, 0.0, 0.0, 0.40), (0.55, 0.0, 0.0, 0.0, 0.12),
                     (0.7, 0.0, 0.0, 0.0, h)]
            self.loop = body
        elif plan == "drag":
            # スポーンの前傾-17.8°をそのまま加速に使う(prefixで水平化しない)
            self.prefix = [climb]
            self.loop = [
                (3.5, 0.0, 0.0, 0.0, self.TILT_HOVER),    # 前傾加速(→準終端速度)
                self.LEVEL,                                # 水平化
                (5.0, 0.0, 0.0, 0.0, h),                   # 惰性減速: f_x∝e^{-ct}
                (1.24, 0.0, -0.1, 0.0, 0.281),             # 前傾に戻す(ループ整合)
            ]
        else:
            raise ValueError(f"unknown sysid plan: {plan}")
        self.prefix_total = sum(s[0] for s in self.prefix)
        self.loop_total = sum(s[0] for s in self.loop)
        print(f"SYSID plan={plan}: prefix {self.prefix_total:.1f}s + "
              f"loop {self.loop_total:.1f}s", flush=True)

    def reset(self):
        self.t0 = None

    def cmd_at(self, elapsed: float) -> tuple[float, float, float, float]:
        """経過秒→コマンド(壁時計にもシム時間にも使える。Genesis側の再生検証と共用)。"""
        if elapsed < self.prefix_total:
            sched, el = self.prefix, elapsed
        else:
            sched = self.loop
            el = (elapsed - self.prefix_total) % self.loop_total   # loopは繰り返し再生
        acc = 0.0
        seg = sched[-1]
        for s in sched:
            acc += s[0]
            if el < acc:
                seg = s
                break
        return float(seg[1]), float(seg[2]), float(seg[3]), float(seg[4])

    def decide(self, shared: dict, warmup: bool = False):
        if self.t0 is None:
            self.t0 = time.time()
        return self.cmd_at(time.time() - self.t0)


class GenesisPilot:
    """学習方策(新旧構造自動判別)で観測→物理コマンドを計算する。"""

    def __init__(self, ckpt_path: str):
        import torch

        from .. import contracts as C
        from ..scripts.eval_video import LoadedPolicy

        self.C = C
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = LoadedPolicy(ckpt_path, self.device, num_envs=1)
        self.action_map = C.ActionMap()
        # 送信スケール = 学習時プラントゲイン / デプロイ時プラントゲイン(軸別)。
        #   学習時: ckpt の cfg スナップショットの drone.cmd_gain(欠落=旧ckptは 1.0)。
        #           方策は「達成レート = cmd_gain × 指令」を前提に最適化されている。
        #   デプロイ時: rad/s bit ON なら実シムは指令どおり≈1.0、レガシーなら ~2.5倍。
        # 送信レートにこのスケールを掛けると、実シムの達成レートが方策の意図レートに一致する。
        #   例) 旧ckpt(1.0)+rad/s → ×1.0 / 旧ckpt+レガシー → ×1/2.5 /
        #       cmd_gain=2.44 ckpt+rad/s → ×2.44 / cmd_gain=2.44 ckpt+レガシー → ×1.0
        try:
            train_gain = np.asarray(
                self.policy.cfg_snapshot["env"]["drone"]["cmd_gain"], np.float32)
        except (KeyError, TypeError, ValueError):
            train_gain = np.ones(3, np.float32)
        deploy_gain = np.asarray(
            RATE_PLANT_GAIN_RADS if USE_RAD_PER_SEC_BODY_RATES else RATE_CMD_GAIN,
            np.float32)
        self.rate_scale = (train_gain / deploy_gain).astype(np.float32)
        print(f"GenesisPilot: rate_scale = train{tuple(np.round(train_gain, 2))} / "
              f"deploy{tuple(np.round(deploy_gain, 2))} = "
              f"{tuple(np.round(self.rate_scale, 3))} "
              f"(rad/s bit {'ON' if USE_RAD_PER_SEC_BODY_RATES else 'OFF'})", flush=True)
        self.last_action = np.zeros(C.ACTION_DIM, dtype=np.float32)
        self.last_vec = np.zeros(C.VEC_DIM, dtype=np.float32)   # 記録用: 直近に方策へ渡したvec
        self._reset_flag = torch.zeros(1, dtype=torch.bool, device=self.device)
        # CUDA初期化を離陸前に済ませる(初回推論の~100msスパイク回避)
        for _ in range(10):
            self.decide({}, warmup=True)
        self.reset()
        print(f"GenesisPilot: warmup done (device={self.device})", flush=True)

    def reset(self):
        self.last_action[:] = 0.0
        self._reset_flag[:] = True  # 次のdecideで履歴をクリア(新構造のみ意味を持つ)

    def _build_vec(self, shared: dict, now: float) -> np.ndarray:
        C = self.C
        vec = np.zeros(C.VEC_DIM, dtype=np.float32)
        imu = shared.get("imu") or {}
        gyro = np.asarray(imu.get("gyro", (0.0, 0.0, 0.0)), np.float32) * np.asarray(GYRO_OBS_SIGN, np.float32)
        accel = np.asarray(imu.get("accel", (0.0, 0.0, -9.81)), np.float32) * np.asarray(ACCEL_OBS_SIGN, np.float32)
        vec[C.VEC_GYRO] = gyro / C.RATE_SCALE
        vec[C.VEC_ACCEL] = accel / C.ACCEL_SCALE
        og = shared.get("obs_gate") or {}
        age_s = now - float(og.get("t_wall", 0.0))
        visible = bool(og.get("visible", 0)) and age_s <= C.GATE_OBS_MAX_AGE_S
        if visible:
            px, py = og.get("center", (0.5, 0.5))
            vec[C.VEC_GATE] = (np.clip(px * 2 - 1, -1.5, 1.5), np.clip(py * 2 - 1, -1.5, 1.5),
                               1.0, np.clip(og.get("rel_dist", 1.0), 0.0, 1.0),
                               np.clip(age_s / C.GATE_OBS_MAX_AGE_S, 0.0, 1.0))
        else:
            vec[C.VEC_GATE] = (0.0, 0.0, 0.0, 1.0, 1.0)
        # Genesis規約(genesis_race_env.py: onehot[max(active_gate-1,0)])に合わせる。
        # DCLのactive_gate_indexはGenesisのactive_gateと同義(狙うゲートの0始まりindex、
        # ゲート通過でインクリメント)。生値をそのまま使うと2本目以降で+1ズレる
        # (runs/sysid_0723_0321でgate0→1遷移を確認しoff-by-one確定)。
        agi = int((shared.get("race") or {}).get("active_gate_index", 0))
        passed = max(agi - 1, 0)
        if 0 <= passed < C.MAX_GATES:
            vec[C.VEC_ONEHOT.start + passed] = 1.0
        vec[C.VEC_LAST_ACTION] = self.last_action
        return vec

    def decide(self, shared: dict, warmup: bool = False) -> tuple[float, float, float, float]:
        """観測を組み立てて1決定。returns (roll_rate, pitch_rate, yaw_rate, thrust)。"""
        torch = self.torch
        now = time.time()
        rgb = (shared.get("obs_rgb") or {}).get("rgb")
        if rgb is None:
            rgb = np.zeros((224, 224, 3), np.uint8)
        rgb_t = torch.as_tensor(np.ascontiguousarray(rgb)).unsqueeze(0).to(self.device)
        vec = self._build_vec(shared, now)
        vec_t = torch.as_tensor(vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.policy.act(rgb_t, vec_t, self._reset_flag)
        self._reset_flag[:] = False
        if not warmup:
            self.last_action = a[0].cpu().numpy().astype(np.float32)
            self.last_vec = vec
        cmd = self.action_map.to_command(a)[0].cpu().numpy()
        # 送信レート = 方策レート × rate_scale(達成レートを方策の意図値へ揃える。上の解説参照)
        s = self.rate_scale
        return float(cmd[0] * s[0]), float(cmd[1] * s[1]), float(cmd[2] * s[2]), float(cmd[3])


# ---------------------------------------------------------------- Windowsリレー

def spawn_win_relay(mavlink_port: int, video_port: int):
    """WSLのinterop経由でWindows側リレー(win_relay.py)を起動する。

    シムはWindowsの127.0.0.1へ送信するため、WSLネイティブで受けるには
    Windows側での転送が必要(詳細はwin_relay.pyのdocstring)。
    returns (Popen|None)。python.exeが見つからない場合はNone(手動起動を案内)。
    """
    import os
    import shutil
    import subprocess

    wsl_ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True
                            ).stdout.split()[0]
    if shutil.which("python.exe") is None:
        print("WARNING: python.exe が見つかりません。Windows側で手動でリレーを起動してください:\n"
              f"  python \\\\wsl.localhost\\{os.environ.get('WSL_DISTRO_NAME', '<distro>')}"
              f"\\{os.path.abspath(os.path.join(os.path.dirname(__file__), 'win_relay.py')).replace('/', chr(92))}"
              f" --target {wsl_ip}", flush=True)
        return None
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "win_relay.py")
    # WSLパスはinterop実行時にWindows側からUNCパスとして見える
    distro = os.environ.get("WSL_DISTRO_NAME", "Ubuntu-22.04")
    unc = "\\\\wsl.localhost\\" + distro + script.replace("/", "\\")
    proc = subprocess.Popen(
        ["python.exe", unc, "--target", wsl_ip,
         "--mavlink-port", str(mavlink_port), "--video-port", str(video_port)])
    print(f"[dcl] Windowsリレー起動 (pid={proc.pid}, target={wsl_ip})", flush=True)
    return proc


# ---------------------------------------------------------------- ゲート検出器

DEFAULT_YOLOX_CKPT = "YOLOX_outputs_x/yolox_x_custom/best_ckpt.pth"


def make_gate_detector(kind: str, yolox_ckpt: str, gate_area_max: float | None = None):
    """ゲート検出関数 detect(img_bgr)->dict を返す。

    kind="yolox": YOLOX-x を GPU/CPU にロード。重みが無い/初期化失敗時は
    警告して HSV にフォールバックする(飛行自体は止めない)。
    kind="hsv":   従来の HSV 色検出。
    gate_area_max: rel_dist正規化のoverride(実bbox較正用。Noneで契約既定150000)。
    """
    import functools

    hsv = functools.partial(gate_detect_hsv, gate_area_max=gate_area_max)
    if kind == "hsv":
        return hsv

    import os

    if not os.path.exists(yolox_ckpt):
        print(f"WARNING: YOLOX重みが見つかりません ({yolox_ckpt}) → HSV検出にフォールバック",
              flush=True)
        return hsv
    try:
        import torch

        from .yolox_gate import GateYOLOX
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return GateYOLOX(yolox_ckpt, dev, gate_area_max=gate_area_max).detect
    except Exception as e:
        print(f"WARNING: YOLOX初期化に失敗 ({type(e).__name__}: {e}) → HSV検出にフォールバック",
              flush=True)
        return hsv


# ---------------------------------------------------------------- メインループ

def run(ckpt: str, mavlink_ip="0.0.0.0", mavlink_port=14550,
        video_port=5600, out_mp4: str | None = "flight_dcl.mp4",
        max_sec: float = 0.0, reset_on_collision: bool = True,
        relay: bool = True, gate_detector: str = "yolox",
        yolox_ckpt: str = DEFAULT_YOLOX_CKPT,
        record_dir: str | None = None, sysid: bool = False,
        sysid_plan: str = "rate",
        gate_area_max: float | None = None) -> None:
    import collections
    import signal

    from .. import contracts as C

    # SIGTERM(docker stop等)もCtrl+Cと同じ後始末パスを通す。
    # SIGINTは明示的にデフォルトハンドラへ戻す: バックグラウンド起動(&)や
    # nohup系ではSIG_IGN継承でCtrl+C/killが無視され、止められなくなるため。
    def _sigterm(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    shared: dict = {}
    if sysid:
        pilot = ScriptedRatePilot(plan=sysid_plan)   # 方策を外した開ループ同定
        print("SYSID mode: 方策なし・既知コマンド列を送出します", flush=True)
    else:
        pilot = GenesisPilot(ckpt)           # 重いロードを接続前に済ませる(後始末不要フェーズ)
    gate_fn = make_gate_detector(gate_detector, yolox_ckpt, gate_area_max)  # YOLOX/HSV も接続前にロード
    recorder = None
    if record_dir:
        from .recorder import FlightRecorder
        recorder = FlightRecorder(record_dir, meta={
            "ckpt": ckpt, "contract_hash": C.contract_hash(),
            "gate_detector": gate_detector, "policy_hz": C.POLICY_HZ,
            "sysid": sysid, "sysid_plan": sysid_plan if sysid else None})
        shared["imu_log"] = collections.deque(maxlen=200_000)  # rxスレッドが積む
    relay_proc = None
    mav = None
    video = None

    def reset_sim_and_wait(reason=""):
        for attempt in range(1, 4):
            print(f"Resetting sim{reason} (attempt {attempt})...", flush=True)
            mav.sim_reset()
            end = time.time() + 8.0
            while time.time() < end:
                mav.heartbeat_if_due()
                race = shared.get("race")
                if race and (race.get("start_pending") or race.get("pin_released")):
                    print("Race scheduled.", flush=True)
                    pilot.reset()
                    return True
                time.sleep(0.1)
        print("WARNING: race did not get scheduled after resets.", flush=True)
        pilot.reset()
        return False

    # ここから先はどの時点でCtrl+Cされても finally が後始末する
    # (特にWindowsリレーの孤児化を防ぐ: 残ると次回起動時にポート衝突する)
    try:
        relay_proc = spawn_win_relay(mavlink_port, video_port) if relay else None
        mav = MavlinkIO(shared, mavlink_ip, mavlink_port)   # ハートビート待ちでブロックし得る
        video = VideoRX(shared, port=video_port, out_mp4=out_mp4, gate_detect_fn=gate_fn)

        print("Arming drone...", flush=True)
        mav.arm()
        # 起動時にレースが動いていなければリセットして新レースをスケジュールさせる
        t_wait = time.time() + 5.0
        while time.time() < t_wait:
            mav.heartbeat_if_due()
            race = shared.get("race")
            if race and (race.get("start_pending") or race.get("pin_released")):
                break
            time.sleep(0.1)
        else:
            reset_sim_and_wait(" (startup: no race scheduled)")
            time.sleep(1.0)
            mav.arm()
        print("Starting Genesis-RL DCL loop... (Ctrl+C to stop)", flush=True)
        t_start = time.time()
        cmd = (0.0, 0.0, 0.0, HOVER_THRUST)
        next_policy_t = 0.0
        next_tx_t = 0.0
        last_status_t = 0.0
        while True:
            now = time.time()
            mav.heartbeat_if_due()

            race = shared.get("race") or {}
            flying = bool(race.get("pin_released"))
            if flying and now >= next_policy_t:      # 30Hzで方策決定
                next_policy_t = max(next_policy_t + 1.0 / C.POLICY_HZ, now)
                cmd = pilot.decide(shared)
                shared["last_cmd"] = cmd
                if recorder is not None:
                    col = shared.get("collision")
                    recorder.record(
                        rgb224=(shared.get("obs_rgb") or {}).get("rgb"),
                        vec=pilot.last_vec, raw_action=pilot.last_action, cmd=cmd,
                        shared=shared,
                        collision=bool(col and not col.get("handled")))
            if not flying:
                cmd = (0.0, 0.0, 0.0, HOVER_THRUST)  # ピン中/待機はフェイルセーフ

            if now >= next_tx_t:                     # 250Hzで送信
                next_tx_t = max(next_tx_t + 1.0 / CONTROL_HZ, now)
                mav.send_rates(*cmd)

            if recorder is not None:                 # 高レートIMUを逐次書き出し
                q = shared.get("imu_log")
                while q:
                    recorder.record_imu(q.popleft())

            col = shared.get("collision")
            if col and not col.get("handled"):
                col["handled"] = True
                # steps.jsonl(30Hz)は同一イテレーション内のpopで衝突を見逃すため、
                # イベントとして時刻を独立に残す(analyze_sysidのデータ除外に使う)
                if recorder is not None:
                    recorder.record_event({"type": "collision", "t_wall": col["t_wall"]})
                if reset_on_collision:
                    print("COLLISION -> reset & re-arm", flush=True)
                    reset_sim_and_wait(" (collision)")
                    time.sleep(3.0)
                    mav.arm()
                shared.pop("collision", None)

            if now - last_status_t > 5.0:
                last_status_t = now
                og = shared.get("obs_gate") or {}
                print(f"[dcl] t={now - t_start:6.1f}s pin={int(flying)} "
                      f"gate={race.get('active_gate_index', '-')} "
                      f"det={int(og.get('visible', 0))} rel={og.get('rel_dist', 1.0):.2f} "
                      f"frames={video.frames} pkts={video.packets} thr={cmd[3]:.3f}", flush=True)

            if max_sec and now - t_start > max_sec:
                print(f"Time limit ({max_sec:.0f}s); stopping.", flush=True)
                break
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopping... (終了処理中: 以降のCtrl+Cは無視されます)", flush=True)
    finally:
        # 後始末中の追加Ctrl+Cで動画クローズ等が中断されないようにする
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # 各ステップを独立にガード(1つの失敗で残り、特にリレー終了、を飛ばさない)
        if mav is not None:
            try:
                mav.sim_reset()
            except Exception as e:
                print(f"(cleanup) sim_reset failed: {e}", flush=True)
            try:
                mav.close()
            except Exception:
                pass
        if video is not None:
            try:
                video.close()   # 受信停止+mp4クローズ(数秒かかることがある)
                if out_mp4:
                    print(f"video saved: {out_mp4}", flush=True)
            except Exception as e:
                print(f"(cleanup) video close failed: {e}", flush=True)
        if relay_proc is not None:
            try:
                relay_proc.terminate()
                relay_proc.wait(timeout=3.0)
            except Exception:
                try:
                    relay_proc.kill()
                except Exception:
                    pass
        if recorder is not None:
            try:
                recorder.close()
            except Exception as e:
                print(f"(cleanup) recorder close failed: {e}", flush=True)
        print("DCL client exited.", flush=True)
