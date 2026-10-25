"""System One 形式の API クライアントの共通部分（HTTP・認証・応答の解釈）。

ローカルの clef（:mod:`systemone_vision_test.services.clef_client`）と、
インターネット経由の jev（:mod:`systemone_vision_test.services.jev_client`）が共有する。

* 接続は**永続セッション（keep-alive）で使い回す**。毎回新しい接続を張ると、
  ``localhost`` の名前解決と IPv6 への接続試行で約 2 秒待たされる
  （:data:`systemone_vision_test.config.PREFER_IPV4_HOST` の説明も参照）
* 送信はここだけが行い、ペイロードの組み立てと接続確認の判定はサブクラスが担当する
* 応答の解釈は :mod:`systemone_vision_test.services.answers` に委譲する
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

import requests

from systemone_vision_test import config
from systemone_vision_test.domain.questions import DIRECTION_QUESTION, DirectionQuestion
from systemone_vision_test.services.answers import (
    DirectionAnswer,
    answer_key_for,
    extract_answer,
    response_body,
    response_json,
)

logger = logging.getLogger(__name__)

#: 接続確認に使う質問（色を答えさせる）。仕様は ``docs/spec.md`` を参照。
CONNECTION_QUESTION_KEY = "check_color"
CONNECTION_STATE_TEXT = "トマトやりんごは熟すと鮮やかな赤色になります。"
CONNECTION_INSTRUCTION = "stateの内容から、説明されている物体のメインの色はどれですか？"
CONNECTION_CRITERIA = {
    "red": "赤色、またはそれに近い暖色",
    "blue": "青色、またはそれに近い冷色",
}
#: 接続確認で期待する回答
CONNECTION_OK_CHOICE = "red"


def prefer_ipv4_url(base_url: str) -> str:
    """``localhost`` を IPv4 表記（``127.0.0.1``）に置き換えた URL を返す。

    Windows では ``localhost`` がまず IPv6(::1) に接続を試み、Ollama が IPv4 のみを
    待ち受けていると約 2 秒待ってから IPv4 にフォールバックする。これを避ける。

    Args:
        base_url: 元の URL。

    Returns:
        ホスト名が ``localhost`` なら IPv4 表記に置き換えた URL。それ以外はそのまま。
    """
    parts = urllib.parse.urlsplit(base_url)
    if parts.hostname != "localhost":
        return base_url
    netloc = config.PREFER_IPV4_HOST
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urllib.parse.urlunsplit(parts._replace(netloc=netloc))


class SystemOneClient:
    """System One 形式の API クライアント（共通部分）。

    送信・認証ヘッダ・エラー処理・応答の解釈を担当する。ペイロードの組み立てと
    接続確認の判定はサブクラスが実装する。

    Args:
        base_url: API のエンドポイント。
        model: 使用するモデル名。
        timeout_sec: 1 リクエストのタイムアウト（秒）。モデルロードに備えて長めにする。
        session: 使用する ``requests.Session``（既定は内部で生成。テスト用に差し替え可能）。

    Attributes:
        base_url: 指定されたエンドポイント。
        request_url: 実際に接続する URL（``localhost`` を IPv4 表記に置き換えたもの）。
        model: 使用するモデル名（GUI から切り替え可能）。
        timeout_sec: 1 リクエストのタイムアウト（秒）。
    """

    #: 環境のプロキシ設定（``trust_env``）を使うかどうか。ローカルは使わず、インターネットは使う。
    trust_env = False

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_sec: float,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url
        self.request_url = prefer_ipv4_url(base_url)
        self.model = model
        self.timeout_sec = timeout_sec
        if session is None:
            session = requests.Session()
            session.trust_env = self.trust_env
        self._session = session

    # -- 読み取り用 ---------------------------------------------------------

    @property
    def session(self) -> requests.Session:
        """使用している HTTP セッション（診断・テスト用）。"""
        return self._session

    @property
    def models_url(self) -> str:
        """モデル一覧（``GET``）の URL。接続先から組み立てる。"""
        return f"{self.request_url.removesuffix('/systemone')}/models"

    # -- 公開 API -----------------------------------------------------------

    def check_connection(self) -> bool:
        """API に接続できるかを確認する。

        Returns:
            接続できて期待どおりの応答（色の回答）が得られたら ``True``。
            通信エラー、タイムアウト、想定外の応答なら ``False``。
        """
        data = self._request_json(self.connection_payload(), what="接続性確認")
        if data is None:
            return False
        answer = extract_answer(data, CONNECTION_QUESTION_KEY)
        return answer is not None and answer.choice.lower() == CONNECTION_OK_CHOICE

    def inspect_landolt_c(
        self,
        image_base64: str,
        *,
        question: DirectionQuestion = DIRECTION_QUESTION,
    ) -> DirectionAnswer | None:
        """ランドルト環画像を判定させ、回答を返す。

        Args:
            image_base64: PNG を Base64 エンコードした文字列（データ URI は付けない）。
            question: 使う質問（言い回しや回答キーを差し替えられる）。

        Returns:
            判定結果。通信エラーや想定外の応答なら ``None``。``choice`` は質問のキーに
            寄せた値（方向への変換は :meth:`DirectionQuestion.direction_of`）。
        """
        data = self._request_json(
            self.inspection_payload(image_base64, question=question), what="判定リクエスト"
        )
        if data is None:
            return None

        answer = extract_answer(data, question.key)
        if answer is None:
            logger.warning(
                "応答から %s を取り出せませんでした: %s", question.key, response_json(data)
            )
            return None
        return DirectionAnswer(
            choice=answer_key_for(question, answer.choice),
            probabilities=answer.probabilities,
            confidence=answer.confidence,
        )

    # -- ペイロード ---------------------------------------------------------

    def connection_payload(self) -> dict[str, Any]:
        """接続性確認のリクエストボディを返す（色を答えさせる質問）。"""
        return {
            "model": self.model,
            "state": {"text": CONNECTION_STATE_TEXT},
            "questions": {
                CONNECTION_QUESTION_KEY: {
                    "type": "choice",
                    "instructions": CONNECTION_INSTRUCTION,
                    "criteria": dict(CONNECTION_CRITERIA),
                }
            },
        }

    def inspection_payload(
        self,
        image_base64: str,
        *,
        question: DirectionQuestion = DIRECTION_QUESTION,
    ) -> dict[str, Any]:
        """方向判定のリクエストボディを返す（画像の渡し方はサブクラスが決める）。

        Raises:
            NotImplementedError: サブクラスが実装していない場合。
        """
        raise NotImplementedError

    def request_headers(self, payload: dict[str, Any]) -> dict[str, str]:
        """リクエストのヘッダを返す（既定は認証ヘッダのみ）。

        Args:
            payload: 送るリクエストボディ。

        Returns:
            付与するヘッダ。
        """
        return self.auth_headers()

    def auth_headers(self) -> dict[str, str]:
        """認証ヘッダを返す（既定は無し）。"""
        return {}

    # -- 内部処理 -----------------------------------------------------------

    def _request_json(self, payload: dict[str, Any], *, what: str) -> dict[str, Any] | None:
        """リクエストを送って JSON を返す（失敗しても例外を投げない）。

        Args:
            payload: 送るリクエストボディ。
            what: ログに出す処理名（``接続性確認`` など）。

        Returns:
            応答の JSON。通信エラー・HTTP エラー・JSON でない応答なら ``None``。
        """
        try:
            return self._post(payload)
        except requests.RequestException as exc:
            logger.warning("%sに失敗しました: %s", what, exc)
        except ValueError as exc:  # 応答が JSON でない
            logger.warning("%sの応答を JSON として読めません: %s", what, exc)
        return None

    def _get_json(self, url: str, *, what: str) -> dict[str, Any] | None:
        """``GET`` して JSON を返す（失敗しても例外を投げない）。

        Args:
            url: 取得する URL。
            what: ログに出す処理名（``モデル一覧の取得`` など）。

        Returns:
            応答の JSON。通信エラー・HTTP エラー・JSON でない応答なら ``None``。
        """
        try:
            response = self._session.get(url, headers=self.auth_headers(), timeout=self.timeout_sec)
        except requests.RequestException as exc:
            logger.warning("%sに失敗しました: %s", what, exc)
            return None
        if response.status_code != 200:
            logger.warning(
                "%sに失敗しました: HTTP %s / %s（%s）",
                what,
                response.status_code,
                response_body(response),
                url,
            )
            return None
        try:
            return response.json()
        except ValueError as exc:
            logger.warning("%sの応答を JSON として読めません: %s", what, exc)
            return None

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """リクエストを送信して JSON を返す。

        Raises:
            requests.RequestException: 通信に失敗した場合。
            requests.HTTPError: 4xx / 5xx が返った場合（応答本文をログに出す）。
            ValueError: 応答が JSON でない場合。
        """
        response = self._session.post(
            self.request_url,
            json=payload,
            headers=self.request_headers(payload),
            timeout=self.timeout_sec,
        )
        if response.status_code >= 400:
            # 原因（モデル名・キー・レート制限など）が本文にしか無いことが多い
            logger.warning(
                "HTTP %s が返りました（%s）: %s",
                response.status_code,
                self.request_url,
                response_body(response),
            )
            raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
        return response.json()
