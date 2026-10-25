"""clef API クライアント（services.clef_client）のテスト。

``requests.Session`` をフェイクに差し替えるため、実際のサーバは不要。
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

from systemone_vision_test import config
from systemone_vision_test.services.clef_client import ClefClient
from systemone_vision_test.services.systemone import prefer_ipv4_url

#: 想定どおりの応答（接続確認と方向判定の両方のキーを含む）
GOOD_RESPONSE = {
    "answers": {
        "check_color": {"choice": "red"},
        "landolt_c_direction": {
            "choice": "right",
            "probabilities": {"up": 0.01, "down": 0.02, "left": 0.03, "right": 0.94},
            "confidence": 0.94,
        },
    }
}


class _FakeResponse:
    """``requests.Response`` の最小限の代替。

    Args:
        payload: 返す JSON。
        status_code: 返すステータス（4xx / 5xx の模擬に使う）。
    """

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        """固定の JSON を返す。"""
        return self._payload


class FakeSession:
    """``requests.Session`` の代替。送信内容を記録する。

    Args:
        payload: 返す JSON。
        error: 送信時に発生させる例外。
    """

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload if payload is not None else GOOD_RESPONSE
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.trust_env = True

    def post(self, url, json=None, timeout=None, headers=None):
        """送信内容（ヘッダ含む）を記録して固定の応答を返す。"""
        self.calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.payload)


@pytest.fixture
def session() -> FakeSession:
    """フェイクセッション。"""
    return FakeSession()


@pytest.fixture
def client(session: FakeSession) -> ClefClient:
    """フェイクセッションを使うクライアント。"""
    return ClefClient(session=session)


# ---------------------------------------------------------------------------
# URL（localhost の IPv6 待ちを避ける）
# ---------------------------------------------------------------------------


def test_localhost_is_replaced_with_ipv4():
    """``localhost`` は IPv4 表記に置き換える（Windows の 2 秒待ちを避ける）。"""
    assert prefer_ipv4_url("http://localhost:11434/v1/systemone") == (
        f"http://{config.PREFER_IPV4_HOST}:11434/v1/systemone"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/v1/systemone",
        "http://192.0.2.10:11434/v1/systemone",  # TEST-NET-1（ドキュメント用の予約域）
        "http://example.com/v1/systemone",
    ],
)
def test_other_hosts_are_kept(url):
    """localhost 以外のホストはそのまま使う。"""
    assert prefer_ipv4_url(url) == url


def test_client_requests_ipv4_url(client, session):
    """実際の接続は IPv4 表記の URL に対して行う。"""
    client.check_connection()

    assert session.calls[-1]["url"] == client.request_url
    assert "localhost" not in client.request_url
    assert client.base_url == config.CLEF_BASE_URL  # 表示用は元のまま


def test_default_session_disables_environment_proxies():
    """既定のセッションは環境のプロキシ設定を使わない。"""
    client = ClefClient()

    assert client.session.trust_env is False


def test_session_is_reused_across_calls(client, session):
    """接続は使い回される（毎回張り直さない）。"""
    client.check_connection()
    client.inspect_landolt_c("BASE64-IMAGE")
    client.check_connection()

    assert len(session.calls) == 3  # すべて同じセッションで送られている
    assert session.calls[-1]["url"] == (f"http://{config.PREFER_IPV4_HOST}:11434/v1/systemone")


# ---------------------------------------------------------------------------
# リクエストボディ
# ---------------------------------------------------------------------------


def test_connection_payload_structure(client, session):
    """接続性確認のリクエストが docs/spec.md の形になっている。"""
    client.check_connection()

    payload = session.calls[-1]["json"]
    assert payload["model"] == config.DEFAULT_MODEL
    assert set(payload["state"]) == {"text"}
    assert payload["state"]["text"] == "トマトやりんごは熟すと鮮やかな赤色になります。"
    assert payload["questions"]["check_color"]["type"] == "choice"
    assert set(payload["questions"]["check_color"]["criteria"]) == {"red", "blue"}


def test_inspection_payload_structure(client, session):
    """方向判定のリクエストが docs/spec.md の形になっている。"""
    client.inspect_landolt_c("BASE64-IMAGE")

    payload = session.calls[-1]["json"]
    assert payload["images"] == ["BASE64-IMAGE"]  # state ではなくトップレベル
    assert payload["state"] == {"text": "視力検査用のランドルト環の画像"}
    question = payload["questions"]["landolt_c_direction"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(config.DIRECTIONS)


def test_payload_uses_selected_model():
    """選択したモデルが両方のリクエストに反映される。"""
    session = FakeSession()
    client = ClefClient(model="clef-flash", session=session)
    client.check_connection()
    client.inspect_landolt_c("BASE64-IMAGE")

    assert [call["json"]["model"] for call in session.calls] == ["clef-flash", "clef-flash"]


def test_timeout_is_configurable():
    """タイムアウトを指定できる（既定は config の値）。"""
    default_session = FakeSession()
    ClefClient(session=default_session).check_connection()
    assert default_session.calls[-1]["timeout"] == config.REQUEST_TIMEOUT_SEC

    short_session = FakeSession()
    ClefClient(timeout_sec=5.0, session=short_session).check_connection()
    assert short_session.calls[-1]["timeout"] == 5.0


def test_available_models_include_clef_variants():
    """既定モデルは選択肢に含まれる。"""
    assert config.DEFAULT_MODEL in config.AVAILABLE_MODELS
    assert set(config.LOCAL_MODELS) == {"clef", "clef-flash"}
    assert (*config.LOCAL_MODELS, *config.ONLINE_MODELS) == config.AVAILABLE_MODELS
    assert config.ONLINE_MODELS == ("jev",)
    # 統計の既定は 3 モデルすべて（jev を含む）
    assert config.BENCHMARK_MODELS == config.AVAILABLE_MODELS
    assert config.BENCHMARK_MODELS == ("clef", "clef-flash", "jev")


# ---------------------------------------------------------------------------
# 応答の解釈
# ---------------------------------------------------------------------------


def test_check_connection_true_when_red(client):
    """``choice`` が red なら接続 OK。"""
    assert client.check_connection() is True


@pytest.mark.parametrize("choice", ["blue", "", None])
def test_check_connection_false_for_other_answers(client, session, choice):
    """期待した色以外の応答は接続 NG。"""
    session.payload = {"answers": {"check_color": {"choice": choice}}}

    assert client.check_connection() is False


def test_check_connection_false_on_error():
    """通信エラーは接続 NG（例外を投げない）。"""
    client = ClefClient(session=FakeSession(error=requests.ConnectionError("接続不可")))

    assert client.check_connection() is False


def test_inspect_returns_direction_answer(client):
    """判定結果を ``DirectionAnswer`` として返す。"""
    answer = client.inspect_landolt_c("BASE64-IMAGE")

    assert answer is not None
    assert answer.choice == "right"
    assert answer.probabilities["right"] == pytest.approx(0.94)
    assert answer.confidence == pytest.approx(0.94)


def test_inspect_returns_none_on_error():
    """通信エラーは ``None``（例外を投げない）。"""
    client = ClefClient(session=FakeSession(error=requests.Timeout("タイムアウト")))

    assert client.inspect_landolt_c("BASE64-IMAGE") is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"answers": {}},
        {"answers": {"other": {"choice": "up"}}},
    ],
)
def test_inspect_returns_none_for_unexpected_payload(client, session, payload):
    """想定外の応答は ``None``。"""
    session.payload = payload

    assert client.inspect_landolt_c("BASE64-IMAGE") is None
