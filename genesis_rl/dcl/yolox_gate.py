# -*- coding: utf-8 -*-
"""YOLOX-x によるゲート検出(HSV ``gate_detect_hsv`` の本番相当・置き換え)。

学習済み重み ``YOLOX_outputs_x/yolox_x_custom/best_ckpt.pth``(yolox-x /
num_classes=1 / depth=1.33 width=1.25)を使い、FPV フレームからゲート bbox を
検出する。出力は HSV 版と同一契約:

    {"visible": 0/1, "center": (px, py ∈[0,1]), "rel_dist": float∈[0,1]}

    center   = bbox 中心を元画像 W,H で正規化(原点=左上)。
    rel_dist = clip(1 - bbox面積 / GATE_AREA_MAX, 0, 1)  … contracts と同一規約。
               面積は元画像(640x360)ピクセル空間で測るので GATE_AREA_MAX の
               較正がそのまま効く。

`yolox` パッケージには依存せず、必要なネットワーク定義(Megvii YOLOX,
Apache-2.0)を自己完結で内蔵する。torch / torchvision のみ使用。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision


# ============================================================ network blocks

class BaseConv(nn.Module):
    """Conv2d + BatchNorm + SiLU。"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """4分割スタック → BaseConv(入力チャンネル x4)。"""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x):
        top_left = x[..., ::2, ::2]
        bot_left = x[..., 1::2, ::2]
        top_right = x[..., ::2, 1::2]
        bot_right = x[..., 1::2, 1::2]
        x = torch.cat((top_left, bot_left, top_right, bot_right), dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.conv2 = BaseConv(hidden, out_channels, 3, 1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        self.conv2 = BaseConv(hidden * (len(kernel_sizes) + 1), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        return self.conv2(x)


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden, 1, 1)
        self.conv3 = BaseConv(2 * hidden, out_channels, 1, 1)
        self.m = nn.Sequential(
            *[Bottleneck(hidden, hidden, shortcut, expansion=1.0) for _ in range(n)])

    def forward(self, x):
        x_1 = self.m(self.conv1(x))
        x_2 = self.conv2(x)
        return self.conv3(torch.cat((x_1, x_2), dim=1))


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5")):
        super().__init__()
        self.out_features = out_features
        base = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base, ksize=3)
        self.dark2 = nn.Sequential(
            BaseConv(base, base * 2, 3, 2),
            CSPLayer(base * 2, base * 2, n=base_depth))
        self.dark3 = nn.Sequential(
            BaseConv(base * 2, base * 4, 3, 2),
            CSPLayer(base * 4, base * 4, n=base_depth * 3))
        self.dark4 = nn.Sequential(
            BaseConv(base * 4, base * 8, 3, 2),
            CSPLayer(base * 8, base * 8, n=base_depth * 3))
        self.dark5 = nn.Sequential(
            BaseConv(base * 8, base * 16, 3, 2),
            SPPBottleneck(base * 16, base * 16),
            CSPLayer(base * 16, base * 16, n=base_depth, shortcut=False))

    def forward(self, x):
        outs = {}
        x = self.stem(x); outs["stem"] = x
        x = self.dark2(x); outs["dark2"] = x
        x = self.dark3(x); outs["dark3"] = x
        x = self.dark4(x); outs["dark4"] = x
        x = self.dark5(x); outs["dark5"] = x
        return {k: v for k, v in outs.items() if k in self.out_features}


class YOLOPAFPN(nn.Module):
    def __init__(self, depth, width, in_features=("dark3", "dark4", "dark5"),
                 in_channels=(256, 512, 1024)):
        super().__init__()
        self.backbone = CSPDarknet(depth, width)
        self.in_features = in_features
        c = [int(ch * width) for ch in in_channels]
        n = round(3 * depth)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(c[2], c[1], 1, 1)
        self.C3_p4 = CSPLayer(2 * c[1], c[1], n=n, shortcut=False)
        self.reduce_conv1 = BaseConv(c[1], c[0], 1, 1)
        self.C3_p3 = CSPLayer(2 * c[0], c[0], n=n, shortcut=False)
        self.bu_conv2 = BaseConv(c[0], c[0], 3, 2)
        self.C3_n3 = CSPLayer(2 * c[0], c[1], n=n, shortcut=False)
        self.bu_conv1 = BaseConv(c[1], c[1], 3, 2)
        self.C3_n4 = CSPLayer(2 * c[1], c[2], n=n, shortcut=False)

    def forward(self, x):
        feats = self.backbone(x)
        x2, x1, x0 = [feats[f] for f in self.in_features]

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat([self.upsample(fpn_out0), x1], 1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat([self.upsample(fpn_out1), x2], 1))
        pan_out1 = self.C3_n3(torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1))
        pan_out0 = self.C3_n4(torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1))
        return (pan_out2, pan_out1, pan_out0)


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, strides=(8, 16, 32),
                 in_channels=(256, 512, 1024)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.n_anchors = 1
        hidden = int(256 * width)

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        for ch in in_channels:
            self.stems.append(BaseConv(int(ch * width), hidden, 1, 1))
            self.cls_convs.append(nn.Sequential(
                BaseConv(hidden, hidden, 3, 1), BaseConv(hidden, hidden, 3, 1)))
            self.reg_convs.append(nn.Sequential(
                BaseConv(hidden, hidden, 3, 1), BaseConv(hidden, hidden, 3, 1)))
            self.cls_preds.append(nn.Conv2d(hidden, self.n_anchors * num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(hidden, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(hidden, self.n_anchors, 1, 1, 0))

    def forward(self, xin):
        outputs = []
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_output = self.cls_preds[k](cls_conv(x))
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            outputs.append(torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1))
        hw = [o.shape[-2:] for o in outputs]
        # [B, n_ch, H, W] → [B, sum(H*W), n_ch]
        flat = torch.cat([o.flatten(start_dim=2) for o in outputs], dim=2).permute(0, 2, 1)
        return self._decode(flat, hw, flat.dtype, flat.device)

    def _decode(self, outputs, hw, dtype, device):
        grids, strides = [], []
        for (hsize, wsize), stride in zip(hw, self.strides):
            yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize), indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            strides.append(torch.full((1, grid.shape[1], 1), stride))
        grids = torch.cat(grids, 1).to(dtype=dtype, device=device)
        strides = torch.cat(strides, 1).to(dtype=dtype, device=device)
        xy = (outputs[..., 0:2] + grids) * strides
        wh = torch.exp(outputs[..., 2:4]) * strides
        return torch.cat([xy, wh, outputs[..., 4:]], dim=-1)


class YOLOX(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_yolox_x(num_classes: int = 1) -> YOLOX:
    """yolox-x (depth=1.33, width=1.25) を組み立てる。"""
    depth, width = 1.33, 1.25
    in_channels = (256, 512, 1024)
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width=width, in_channels=in_channels)
    return YOLOX(backbone, head)


# ============================================================ pre / post process

def _preproc(img_bgr: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """アスペクト比維持のレターボックス化(余白=114)。CHW float32 と縮尺 r を返す。"""
    import cv2

    padded = np.full((input_size[0], input_size[1], 3), 114.0, dtype=np.float32)
    r = min(input_size[0] / img_bgr.shape[0], input_size[1] / img_bgr.shape[1])
    nw, nh = int(img_bgr.shape[1] * r), int(img_bgr.shape[0] * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded[:nh, :nw] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r


def _postprocess(pred: torch.Tensor, num_classes: int, conf_thre: float, nms_thre: float):
    """decode 済み出力(1画像分)から NMS 後の検出 [K,7] を返す。無ければ None。

    行: [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
    """
    p = pred[0]  # [N, 5+num_classes]  (cx, cy, w, h, obj, cls...)
    box = torch.empty_like(p[:, :4])
    box[:, 0] = p[:, 0] - p[:, 2] / 2
    box[:, 1] = p[:, 1] - p[:, 3] / 2
    box[:, 2] = p[:, 0] + p[:, 2] / 2
    box[:, 3] = p[:, 1] + p[:, 3] / 2

    class_conf, class_pred = torch.max(p[:, 5:5 + num_classes], dim=1, keepdim=True)
    scores = p[:, 4] * class_conf.reshape(-1)
    mask = scores >= conf_thre
    if not bool(mask.any()):
        return None
    dets = torch.cat((box, p[:, 4:5], class_conf, class_pred.float()), dim=1)[mask]
    keep = torchvision.ops.batched_nms(
        dets[:, :4], dets[:, 4] * dets[:, 5], dets[:, 6], nms_thre)
    return dets[keep]


# ============================================================ detector

# ゲート色検証(andu_ddrnet tools/infer_video.py の orange_ratio と同方式)。
# 検出ボックス内のゲート色画素割合が GATE_ORANGE_THR 未満なら誤検出として捨てる。
# 参照色は2色: 公式オレンジ #FA3C0F(andu_ddrnetの値)+ 実飛行フレーム実測のピンク寄り赤
# (RGB 235,100,95。現行シムのゲートはブルームでピンク化しており、公式色だけだと本物の
#  ゲートまで弾く: runs/20260720_212416 の frame_000050 で ratio 0.08 < 0.2 だった)。
# 判定は両参照色の max(どちらかに20%以上合えばゲート)。シアンのリボン等の誤検出
# (Lab a<0)はどちらの参照からも遠く、正しく除外される(同 frame_000190 で実測)。
GATE_COLORS_BGR = ((15, 60, 250), (95, 100, 235))
ORANGE_AB_TOL = 28               # Lab色度(a,b)距離のしきい値(明るさ変化に頑健)
GATE_ORANGE_THR = 0.2

_GATE_ABS: list | None = None


def _gate_abs():
    import cv2

    global _GATE_ABS
    if _GATE_ABS is None:
        _GATE_ABS = [cv2.cvtColor(np.array([[c]], dtype=np.uint8),
                                  cv2.COLOR_BGR2LAB)[0, 0, 1:].astype(np.int16)
                     for c in GATE_COLORS_BGR]
    return _GATE_ABS


def orange_ratio(frame_bgr: np.ndarray, box) -> float:
    """検出ボックス内のゲート色画素の割合(0-1、参照色ごとのmax)。"""
    import cv2

    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    lab = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.int16)
    best = 0.0
    for ab in _gate_abs():
        da = lab[:, :, 1] - int(ab[0])
        db = lab[:, :, 2] - int(ab[1])
        dist = np.sqrt((da * da + db * db).astype(np.float32))
        best = max(best, float(np.count_nonzero(dist <= ORANGE_AB_TOL)) / float(dist.size))
    return best


class GateYOLOX:
    """YOLOX-x でゲートを検出し、HSV 版と同一契約の dict を返す。

    しきい値・選択ロジックは実飛行で使っている andu_ddrnet のパイプライン
    (tools/infer_video.py: conf 0.5 / NMS 0.45 / min_area 1500px² /
     面積最大の1個のみ採用 / オレンジ画素率>=0.2 で誤検出除去)と同一に揃える。
    重い推論のため VideoRX の処理スレッドで生成・実行される想定。
    """

    def __init__(self, ckpt_path: str, device: torch.device,
                 num_classes: int = 1, input_size: tuple[int, int] = (640, 640),
                 conf_thre: float = 0.4, nms_thre: float = 0.45,
                 fp16: bool | None = None, gate_area_max: float | None = None,
                 min_area: float = 500.0, max_side: float = 30000.0,
                 orange_thr: float = GATE_ORANGE_THR):
        # conf/min_area は andu_ddrnet(0.5/1500)から緩和: 学習側 SimGateDetector の契約は
        # 45mまで検出で、遠方ゲートはbboxが小さく(40mで~450px²)スコアも低い(実測0.3台)。
        # 0.5/1500のままだとゲート通過直後の次ゲート(遠い)が長時間不可視になり、
        # 方策が航法手掛かりを失う(runs/dcl_onehot_fix: agi=1区間の可視率20%)。
        # 誤検出はオレンジ率チェック(orange_thr)が引き続き除去する。
        from ..contracts import GATE_AREA_MAX

        self.device = device
        self.input_size = input_size
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.num_classes = num_classes
        self.min_area = float(min_area)
        self.max_side = float(max_side)
        self.orange_thr = float(orange_thr)
        # rel_dist = 1 - bbox面積/gate_area_max。実bboxはGenesis投影(s_px²)より小さいため
        # 較正が必要(overrideで下げると接近時にrel_distが早く下がる)。既定は契約値。
        self.gate_area_max = float(GATE_AREA_MAX if gate_area_max is None else gate_area_max)
        self.fp16 = (device.type == "cuda") if fp16 is None else fp16

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model = build_yolox_x(num_classes)
        model.load_state_dict(state, strict=True)   # 構造不一致はここで即エラー
        model.eval().to(device)
        if self.fp16:
            model.half()
        self.model = model
        print(f"GateYOLOX: loaded {ckpt_path} (yolox-x, classes={num_classes}, "
              f"device={device}, fp16={self.fp16})", flush=True)

    @torch.no_grad()
    def detect(self, img_bgr: np.ndarray, return_box: bool = False) -> dict:
        """return_box=True で可視化用に box(x1,y1,x2,y2 px) と score も付与する
        (契約キー visible/center/rel_dist は不変)。"""
        H, W = img_bgr.shape[:2]
        padded, r = _preproc(img_bgr, self.input_size)
        x = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        x = x.half() if self.fp16 else x.float()
        out = self.model(x).float()
        dets = _postprocess(out, self.num_classes, self.conf_thre, self.nms_thre)
        if dets is None or dets.shape[0] == 0:
            res = {"visible": 0, "center": (0.5, 0.5), "rel_dist": 1.0}
            if return_box:
                res["box"] = None
                res["score"] = 0.0
            return res

        # andu_ddrnet 準拠の後処理: 元画像座標へ戻し min_area/max_side でフィルタ →
        # 面積最大の1個だけ採用 → ボックス内オレンジ率で本物のゲートか検証。
        # raw(クリップ前)座標も保持する: 至近でゲート枠が画面外へはみ出すと、クリップ後の
        # bbox中心は画面中心へ引き寄せられ「整列済み」という偽信号になる。学習側
        # SimGateDetector は投影中心が画面内(10pxマージン)なら可視・その中心を返す契約なので、
        # 中心と面積は raw から取り、可視判定も「raw中心が画面内か」で行う(§可視判定を参照)。
        d = dets.cpu().numpy()
        raw = d[:, :4] / r
        boxes = raw.copy()
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, H)
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        keep = np.ones(len(d), dtype=bool)
        if self.min_area > 0:
            keep &= (bw * bh) >= self.min_area
        if self.max_side > 0:
            keep &= np.maximum(bw, bh) <= self.max_side
        # 可視判定(学習契約と同一): rawのbbox中心が画面内(10pxマージン)にあること。
        # ゲート通過直後に出る「視野を覆う巨大box」は、機体がゲート面の内側/外側にいて
        # 中心が画面外へ抜けるため自動的に落ちる(旧・端3辺ルールの置き換え。あちらは
        # 正対至近=本来可視のケースまで捨てていた)。
        raw_cx = (raw[:, 0] + raw[:, 2]) * 0.5
        raw_cy = (raw[:, 1] + raw[:, 3]) * 0.5
        keep &= (raw_cx > 10) & (raw_cx < W - 10) & (raw_cy > 10) & (raw_cy < H - 10)
        if not keep.any():
            res = {"visible": 0, "center": (0.5, 0.5), "rel_dist": 1.0}
            if return_box:
                res["box"] = None
                res["score"] = 0.0
            return res
        idx = np.where(keep)[0]
        i = int(idx[np.argmax((bw * bh)[idx])])          # 面積最大の1個
        best = d[i]
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        # ゲート色検証。遠方・斜めのゲートはbbox内に背景(開口越しの暗部)が多くオレンジ率が
        # 下がるため、面積依存で閾値を緩める: 小box(<4000px²、目安10m超)は0.5×thr。
        # シアンリボン等の誤検出(実測 area~7400/ratio0.16)は通常サイズ帯なのでフル閾値のまま。
        orange = orange_ratio(img_bgr, (x1, y1, x2, y2)) if self.orange_thr > 0 else 1.0
        area_box = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        thr = self.orange_thr * (0.5 if area_box < 4000.0 else 1.0)
        if orange < thr:
            res = {"visible": 0, "center": (0.5, 0.5), "rel_dist": 1.0}
            if return_box:
                res["box"] = (x1, y1, x2, y2)
                res["score"] = float(best[4] * best[5])
                res["orange"] = float(orange)
            return res
        # 中心と面積は raw(クリップ前)から取る。学習側 SimGateDetector は
        #   center = ゲート中心の投影 / rel_dist = 1 - s_px²/GATE_AREA_MAX, s_px = FX·2.7/d
        # で、どちらも画面外へはみ出しても幾何どおりに増え続ける(s_pxのみIMG_Wでクランプ)。
        # クリップ後の値を使うと至近でrel_distが頭打ちになり中心も画面中心へ寄る。
        rx1, ry1, rx2, ry2 = (float(v) for v in raw[i])
        cx, cy = (rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5
        area_raw = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
        rel = float(np.clip(1.0 - area_raw / self.gate_area_max, 0.0, 1.0))
        res = {"visible": 1, "center": (float(cx / W), float(cy / H)), "rel_dist": rel}
        if return_box:
            res["box"] = (x1, y1, x2, y2)
            res["score"] = float(best[4] * best[5])
            res["orange"] = float(orange)
        return res
