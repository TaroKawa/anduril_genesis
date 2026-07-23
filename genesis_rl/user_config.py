# -*- coding: utf-8 -*-
"""ユーザー調整値(物理/アクション/符号/センサ/デプロイ)を top-level `config.yaml` から読む。

各モジュールは `uc(section, key, default)` で自分の定数を初期化する:
    from .user_config import uc
    RATE_LIMITS = uc("action", "rate_limits", (1.0, 1.0, 1.0))

config.yaml が存在しない / セクションやキーが無い場合は default(=現行のハードコード値)を
返すので、ファイルが無くても従来どおり動く(無回帰)。config.yaml のパスは既定でリポジトリ
直下 `config.yaml`、環境変数 `GENESIS_USER_CONFIG` で上書き可能。

※ここは「人が触る物理/アクション/符号/センサ/デプロイのノブ」の単一ソース。学習ハイパラ
  (sac/curriculum/run/hw と env のエピソード設定)は従来どおり configs/train.yaml + config.py。
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml


def _config_path() -> Path:
    env = os.environ.get("GENESIS_USER_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "config.yaml"


@functools.lru_cache(maxsize=1)
def _load() -> dict:
    p = _config_path()
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def uc(section: str, key: str, default):
    """config.yaml の section.key を返す。無ければ default。default が tuple なら list→tuple 変換。"""
    sec = _load().get(section) or {}
    if key not in sec:
        return default
    v = sec[key]
    if isinstance(default, tuple) and isinstance(v, (list, tuple)):
        return tuple(v)
    return v


def config_source() -> str:
    """読み込んだ config.yaml のパス(存在すれば)/ 'defaults' を返す(ログ用)。"""
    p = _config_path()
    return str(p) if p.exists() else "defaults(no config.yaml)"
