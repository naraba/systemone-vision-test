"""jev（TypeSafe / OpenRouter の System One API）のクライアント。

仕様: https://docs.typesafe.ai/images ・ https://api.typesafe.ai/openapi.json

送信・応答の解釈は共通部分（:class:`~systemone_vision_test.services.systemone.SystemOneClient`
と :mod:`systemone_vision_test.services.answers`）にあり、この module は**インターネット経由の
差分**だけを持つ。

* **接続先**: 既定は **OpenRouter 経由**の ``https://openrouter.ai/api/v1/systemone``
  （:data:`systemone_vision_test.config.JEV_BASE_URL`）。TypeSafe に直接つなぐ場合は
  ``https://api.typesafe.ai/v1/systemone`` を指定する
* **認証**: ``Authorization: Bearer <API キー>``
* **画像**: TypeSafe 直接は ``assets`` に入れて参照名を state に書き、
  ``TypeSafe-Danger-Zone-Images: true`` ヘッダを付ける。OpenRouter は ``assets`` を
  **無視する**ため、state を配列にして ``image_url``（data URI）で渡す
* **接続確認**: ``GET <接続先の /models>`` で認証と接続を確かめ、System One へ 1 回
  小さなリクエストを送って応答の形まで確認する
* **プロキシ**: インターネット経由なので環境のプロキシ設定を使う（``trust_env = True``）

**API キーはリポジトリに入れない。** 環境変数 ``JEV_API_KEY``、または git 管理外の
:data:`systemone_vision_test.config.JEV_API_KEY_FILE`（既定 ``.jev_api_key``）で与える。
未設定のときは通信せずに接続 NG を返し、設定先をログに出す（キーの値そのものはログに出さない）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests

from systemone_vision_test import config
from systemone_vision_test.domain.questions import (
    DIRECTION_QUESTION,
    DIRECTION_STATE_TEXT,
    DirectionQuestion,
)
from systemone_vision_test.services.answers import response_json
from systemone_vision_test.services.systemone import SystemOneClient

logger = logging.getLogger(__name__)

#: 画像入りのリクエストに必要なヘッダ（TypeSafe の仕様。値は常に ``true``）。
#: これが無いと、画像や Box 質問を含むリクエストは 400 になる。
DANGER_ZONE_IMAGE_HEADER = "TypeSafe-Danger-Zone-Images"


def resolve_credential(
    environ: dict[str, str] | None = None,
    path: os.PathLike[str] | str | None = None,
) -> tuple[str, str]:
    """API キーと、その取得元（キーの値は含まない説明）を返す。

    環境変数 :data:`systemone_vision_test.config.JEV_API_KEY_ENV` が優先で、無ければ
    キーファイル :data:`systemone_vision_test.config.JEV_API_KEY_FILE` を読む
    （空行と ``#`` 始まりの行は無視する）。

    Args:
        environ: 環境変数の代わりに見る辞書（既定は ``os.environ``。テスト用）。
        path: キーファイルの場所（既定は :data:`systemone_vision_test.config.JEV_API_KEY_FILE`）。

    Returns:
        ``(API キー, 取得元)``。取得元は ``環境変数 JEV_API_KEY`` /
        ``キーファイル .jev_api_key`` / ``未設定`` のいずれか。
    """
    target = os.environ if environ is None else environ
    key = (target.get(config.JEV_API_KEY_ENV) or "").strip()
    if key:
        return key, f"環境変数 {config.JEV_API_KEY_ENV}"

    source = Path(config.JEV_API_KEY_FILE if path is None else path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        return "", "未設定"
    for line in text.splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("#"):
            return candidate, f"キーファイル {source.name}"
    return "", "未設定"


def load_api_key(
    environ: dict[str, str] | None = None,
    path: os.PathLike[str] | str | None = None,
) -> str:
    """API キーを環境変数 → キーファイルの順に探して返す。

    Args:
        environ: 環境変数の代わりに見る辞書（既定は ``os.environ``。テスト用）。
        path: キーファイルの場所（既定は :data:`systemone_vision_test.config.JEV_API_KEY_FILE`）。

    Returns:
        API キー。見つからなければ空文字。
    """
    return resolve_credential(environ, path)[0]


def api_model_name(selector: str | None = None) -> str:
    """選択名から、リクエストの ``model`` に載せる名前を決める。

    ``--models jev`` のような選択名（レポートに出す名前）は、そのまま API に送ると
    ``typesafe/jev`` のように解決されて存在しないモデルになる。オンラインのモデルは
    :data:`systemone_vision_test.config.JEV_MODEL`（既定 ``jev-latest``）を送り、
    ``typesafe/jev-1.13`` のような API のモデル名を直接指定したときはそのまま使う。

    Args:
        selector: 計画や GUI で選ばれた名前（``None`` なら既定）。

    Returns:
        リクエストの ``model`` に載せる名前。
    """
    if selector is None or selector in config.ONLINE_MODELS:
        return config.JEV_MODEL
    return selector


def data_uri(image_base64: str) -> str:
    """画像の data URI を作る（OpenRouter の ``image_url`` に渡す形）。

    Args:
        image_base64: Base64 文字列（データ URI は付けない）。

    Returns:
        ``data:image/png;base64,...`` の形の文字列（形式は
        :data:`systemone_vision_test.config.JEV_IMAGE_FORMAT`）。
    """
    extension = config.JEV_IMAGE_FORMAT.strip().lower()
    mime = "image/jpeg" if extension in ("jpg", "jpeg") else f"image/{extension or 'png'}"
    return f"data:{mime};base64,{image_base64}"


class JevClient(SystemOneClient):
    """jev API のクライアント（インターネット経由・API キー認証）。

    ローカルの clef と同じ System One 形式を使うが、認証ヘッダを付け、環境のプロキシ設定を
    使い、接続先・モデル名・タイムアウトは jev 用の設定値を使う。

    Args:
        base_url: 接続先（``None`` なら :data:`systemone_vision_test.config.JEV_BASE_URL`）。
        model: 使用するモデル名。``None`` か選択名（``jev``）なら
            :data:`systemone_vision_test.config.JEV_MODEL`（既定 ``jev-latest``）を使い、
            ``typesafe/jev-1.13`` のような API のモデル名はそのまま使う（:func:`api_model_name`）。
        timeout_sec: 1 リクエストのタイムアウト（秒。``None`` なら jev の既定値）。
        session: 使用する ``requests.Session``（既定は内部で生成。テスト用に差し替え可能）。
        credential: API キー（``None`` なら環境変数 → キーファイルの順に探す）。

    Attributes:
        credential: 使用する API キー（未設定なら空文字）。
        credential_source: API キーの取得元（キーの値は含まない）。
        base_url_source: 接続先の由来（環境変数か config の既定値か）。
    """

    trust_env = True

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        session: requests.Session | None = None,
        credential: str | None = None,
    ) -> None:
        super().__init__(
            base_url=config.JEV_BASE_URL if base_url is None else base_url,
            model=api_model_name(model),
            timeout_sec=config.JEV_REQUEST_TIMEOUT_SEC if timeout_sec is None else timeout_sec,
            session=session,
        )
        if credential is None:
            self.credential, self.credential_source = resolve_credential()
        else:
            self.credential, self.credential_source = credential, "引数で指定"
        # 接続先が環境変数で上書きされているかどうか（設定の取り違えを診断しやすくする）
        from_env = os.environ.get(config.JEV_BASE_URL_ENV)
        self.base_url_source = (
            f"環境変数 {config.JEV_BASE_URL_ENV}" if from_env else "config の既定値"
        )

    # -- 設定 ---------------------------------------------------------------

    @property
    def provider(self) -> str:
        """接続先の種類（``typesafe`` / ``openrouter``）。

        :data:`systemone_vision_test.config.JEV_PROVIDER` が空なら、接続先の URL から推定する。
        """
        if config.JEV_PROVIDER:
            return config.JEV_PROVIDER
        return "openrouter" if "openrouter" in self.base_url else "typesafe"

    @property
    def configured(self) -> bool:
        """接続先と API キーが設定済みかどうか。"""
        return bool(self.base_url) and bool(self.credential)

    def settings_summary(self) -> str:
        """ログに出す設定の要約（API キーの値は含まない）。"""
        return (
            f"接続先={self.request_url or '(空)'}（{self.base_url_source}）"
            f" / provider={self.provider} / モデル={self.model}"
            f" / API キー={self.credential_source}"
        )

    def auth_headers(self) -> dict[str, str]:
        """認証ヘッダを返す（API キーが未設定なら空）。"""
        if not self.credential:
            return {}
        return {config.JEV_AUTH_HEADER: config.JEV_AUTH_SCHEME.format(key=self.credential)}

    def request_headers(self, payload: dict[str, Any]) -> dict[str, str]:
        """リクエストのヘッダを返す。

        画像（``assets``）を含むリクエストには ``TypeSafe-Danger-Zone-Images: true``
        が必要なので、ここで付ける（OpenRouter は state 配列で渡すため付かない）。

        Args:
            payload: 送るリクエストボディ。

        Returns:
            認証ヘッダ（TypeSafe 直接で画像入りのときは Danger Zone ヘッダも）。
        """
        headers = dict(super().request_headers(payload))
        if payload.get("assets") and config.JEV_SEND_DANGER_ZONE_HEADER:
            headers[DANGER_ZONE_IMAGE_HEADER] = "true"
        return headers

    # -- 公開 API -----------------------------------------------------------

    def check_connection(self) -> bool:
        """接続先と API キーで API に接続できるかを確認する。

        トークンを消費しない ``GET``（モデル一覧）で認証と接続を確かめ、続けて System One の
        エンドポイントへ小さなリクエストを 1 回送って、応答の形まで確認する。

        Returns:
            どちらも通れば ``True``。設定不足・通信エラー・想定外の応答なら ``False``。
        """
        settings = self.settings_summary()
        if not self.configured:
            logger.warning(
                "jev の設定が足りません（%s）。接続先は config.JEV_BASE_URL か環境変数 %s、"
                "API キーは環境変数 %s かキーファイル %s（git 管理外）で設定してください",
                settings,
                config.JEV_BASE_URL_ENV,
                config.JEV_API_KEY_ENV,
                config.JEV_API_KEY_FILE.name,
            )
            return False
        logger.info("jev の設定: %s", settings)

        if not self._check_model_list():
            return False

        what = "System One エンドポイントへの接続確認"
        data = self._request_json(self.connection_payload(), what=what)
        if data is None:
            logger.warning("jev の設定を確認してください（%s）", settings)
            return False
        if not isinstance(data.get("answers"), dict) or not data["answers"]:
            logger.warning(
                "jev の応答に answers が含まれていません: %s（%s）",
                response_json(data),
                settings,
            )
            return False
        logger.info("jev に接続できました（応答のモデル=%s）", data.get("model", "(不明)"))
        return True

    def inspection_payload(
        self,
        image_base64: str,
        *,
        question: DirectionQuestion = DIRECTION_QUESTION,
    ) -> dict[str, Any]:
        """方向判定のリクエストボディを返す（接続先ごとに画像の渡し方を変える）。

        * **TypeSafe 直接**: 画像を ``assets`` に入れ、**その参照名を state に書く**
          （TypeSafe は参照名が state / instructions / criteria のどこかに現れることを
          要求する）。画像を含むリクエストには Danger Zone ヘッダが要る
        * **OpenRouter**: ``assets`` は受け付けられず**無視される**ので、state を配列にして
          ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`` で渡す
          （OpenRouter のエラー応答が示す形式）

        Args:
            image_base64: PNG を Base64 エンコードした文字列（データ URI は付けない）。
            question: 使う質問（言い回しや回答キーを差し替えられる）。

        Returns:
            リクエストボディ。
        """
        questions = {question.key: question.question()}
        if self.provider == "openrouter":
            return {
                "model": self.model,
                "state": [
                    {"type": "text", "text": DIRECTION_STATE_TEXT},
                    {"type": "image_url", "image_url": {"url": data_uri(image_base64)}},
                ],
                "questions": questions,
            }

        reference = config.JEV_IMAGE_ASSET_REFERENCE
        return {
            "model": self.model,
            "state": {"text": DIRECTION_STATE_TEXT, "image": reference},
            "assets": {
                reference: {
                    "type": "image",
                    "format": config.JEV_IMAGE_FORMAT,
                    "data_base64": image_base64,
                }
            },
            "questions": questions,
        }

    # -- 内部処理 -----------------------------------------------------------

    def _check_model_list(self) -> bool:
        """モデル一覧を取得し、利用するモデルが載っているかを確かめる。

        OpenRouter の一覧は名前空間が違い（``typesafe/jev-...``）、素の ``jev-latest`` は
        載らないので照合しない（接続できることだけ確かめる）。

        Returns:
            一覧を取得できて問題が無ければ ``True``。取得できなければ ``False``。
        """
        data = self._get_json(self.models_url, what="モデル一覧の取得")
        if data is None:
            return False
        if self.provider == "openrouter":
            return True

        names = _model_names(data)
        if names and config.JEV_MODEL not in names:
            logger.warning(
                "jev のモデル %s が一覧にありません（例: %s）。必要なら環境変数 JEV_MODEL で"
                "指定してください",
                config.JEV_MODEL,
                ", ".join(names[:5]),
            )
        return True


def _names_of(item: Any) -> list[str]:
    """モデル一覧の 1 件分（辞書または文字列）から名前を取り出す。"""
    if isinstance(item, str):
        return [item]
    if not isinstance(item, dict):
        return []
    for name in ("name", "id", "model"):
        value = item.get(name)
        if isinstance(value, str):
            return [value]
    return []


def _model_names(data: Any) -> list[str]:
    """モデル一覧（``GET /v1/models``）の応答からモデル名を集める（形の違いを吸収する）。"""
    if isinstance(data, dict):
        candidates = [data[key] for key in ("models", "data", "items") if key in data] or [data]
    elif isinstance(data, list):
        candidates = [data]
    else:
        return []

    names: list[str] = []
    for candidate in candidates:
        items = candidate if isinstance(candidate, list) else [candidate]
        for item in items:
            names += _names_of(item)
    return names
