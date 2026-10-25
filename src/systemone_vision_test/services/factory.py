"""モデル名から API クライアントを作る。

ローカルの clef / clef-flash は :class:`~systemone_vision_test.services.clef_client.ClefClient`、
インターネット経由の jev は :class:`~systemone_vision_test.services.jev_client.JevClient` を使う。
呼び出し側（GUI・統計バッチ）はモデル名だけを渡せばよい。
"""

from __future__ import annotations

from systemone_vision_test import config
from systemone_vision_test.services.clef_client import ClefClient
from systemone_vision_test.services.jev_client import JevClient


def is_online_model(model: str) -> bool:
    """インターネット経由で使うモデルかどうか。

    ``jev`` のような選択名（:data:`config.ONLINE_MODELS`）と、API に送るモデル名
    （:data:`config.JEV_MODEL`。既定 ``jev-latest``）のどちらでも ``True`` を返す。

    Args:
        model: モデル名（選択名または API のモデル名）。

    Returns:
        インターネット経由（API キーが必要）なら ``True``。
    """
    return model in config.ONLINE_MODELS or model == config.JEV_MODEL


def create_client(
    model: str,
    *,
    base_url: str | None = None,
    timeout_sec: float | None = None,
) -> ClefClient:
    """モデル名に合ったクライアントを作る。

    Args:
        model: 使用するモデル名。
        base_url: 接続先（``None`` ならモデルの種類ごとの既定値）。
        timeout_sec: 1 リクエストのタイムアウト（秒。``None`` なら既定値）。

    Returns:
        使用するクライアント。
    """
    if is_online_model(model):
        return JevClient(model=model, base_url=base_url, timeout_sec=timeout_sec)
    return ClefClient(
        base_url=base_url or config.CLEF_BASE_URL,
        model=model,
        timeout_sec=timeout_sec or config.REQUEST_TIMEOUT_SEC,
    )
