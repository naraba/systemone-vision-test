"""ローカルの clef / clef-flash（System One 互換 API）のクライアント。

送信・応答の解釈は共通部分（:class:`~systemone_vision_test.services.systemone.SystemOneClient`）
にあり、この module は**ローカル固有の差分**だけを持つ。

* 接続先はローカルの Ollama（``http://localhost:11434/v1/systemone``）。認証は不要
* 画像は ``images`` 配列に Base64 を入れて渡す（インターネット経由の jev は別形式）
* 環境のプロキシ設定は使わない（``trust_env = False``。ローカル接続のため）
"""

from __future__ import annotations

from typing import Any

import requests

from systemone_vision_test import config
from systemone_vision_test.domain.questions import (
    DIRECTION_QUESTION,
    DIRECTION_STATE_TEXT,
    DirectionQuestion,
)
from systemone_vision_test.services.systemone import SystemOneClient


class ClefClient(SystemOneClient):
    """ローカルの System One 互換 API（clef / clef-flash）のクライアント。

    接続は**永続セッション（keep-alive）で使い回す**（:mod:`systemone_vision_test.services.systemone`
    の説明を参照）。

    Args:
        base_url: API のエンドポイント
            （既定は :data:`systemone_vision_test.config.CLEF_BASE_URL`）。
        model: 使用するモデル名（既定は :data:`systemone_vision_test.config.DEFAULT_MODEL`）。
        timeout_sec: 1 リクエストのタイムアウト（秒）。
        session: 使用する ``requests.Session``（既定は内部で生成。テスト用に差し替え可能）。
    """

    def __init__(
        self,
        base_url: str = config.CLEF_BASE_URL,
        model: str = config.DEFAULT_MODEL,
        timeout_sec: float = config.REQUEST_TIMEOUT_SEC,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(base_url=base_url, model=model, timeout_sec=timeout_sec, session=session)

    def inspection_payload(
        self,
        image_base64: str,
        *,
        question: DirectionQuestion = DIRECTION_QUESTION,
    ) -> dict[str, Any]:
        """方向判定のリクエストボディを返す（画像は ``images`` 配列で渡す）。

        Args:
            image_base64: PNG を Base64 エンコードした文字列（データ URI は付けない）。
            question: 使う質問（言い回しや回答キーを差し替えられる）。

        Returns:
            リクエストボディ。
        """
        return {
            "model": self.model,
            "images": [image_base64],
            "state": {"text": DIRECTION_STATE_TEXT},
            "questions": {question.key: question.question()},
        }
