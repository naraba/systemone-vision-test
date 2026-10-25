"""実際の clef API に対する統合テスト（既定では実行しない）。

**環境に依存させない**ため、次の 2 段構えにしている。

* 既定の ``pytest`` 実行では ``integration`` マーカーで除外される
* ``pytest -m integration`` で実行しても、Ollama が応答しなければ自動でスキップされる

明示的に実行する場合::

    pytest -m integration
"""

from __future__ import annotations

import urllib.parse

import pytest
import requests

from systemone_vision_test import config
from systemone_vision_test.domain.landolt import (
    GenerationMode,
    LandoltCParams,
    encode_png_base64,
    generate_landolt_c_image,
)
from systemone_vision_test.services.clef_client import ClefClient

#: サーバの生存確認に使うタイムアウト（モデルのロードを待たない短い値）
PROBE_TIMEOUT_SEC = 2.0


def is_clef_server_available(
    base_url: str = config.CLEF_BASE_URL, timeout_sec: float = PROBE_TIMEOUT_SEC
) -> bool:
    """Ollama の API サーバが応答するかを短いタイムアウトで確認する。

    モデルのロードで待たされないよう、``/api/tags`` を叩いて判定する。

    Args:
        base_url: clef のエンドポイント（ホストとポートの判定に使う）。
        timeout_sec: 判定のタイムアウト（秒）。

    Returns:
        サーバが応答したら ``True``。
    """
    parts = urllib.parse.urlsplit(base_url)
    probe_url = f"{parts.scheme}://{parts.netloc}/api/tags"
    try:
        response = requests.get(probe_url, timeout=timeout_sec)
    except requests.RequestException:
        return False
    return response.ok


@pytest.fixture(scope="module")
def live_client() -> ClefClient:
    """実際のサーバに接続するクライアント（未起動ならスキップ）。"""
    if not is_clef_server_available():
        pytest.skip(f"clef API ({config.CLEF_BASE_URL}) に接続できないためスキップします")
    return ClefClient()


# ---------------------------------------------------------------------------
# 生存確認の判定ロジック（環境に依存しない）
# ---------------------------------------------------------------------------


@pytest.mark.allow_network
@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:1/v1/systemone",  # 到達できないポート
        "http://192.0.2.1/v1/systemone",  # TEST-NET-1（必ず到達不可）
    ],
)
def test_probe_reports_unreachable_server(base_url):
    """接続できないエンドポイントは ``False`` と判定する。"""
    assert is_clef_server_available(base_url, timeout_sec=1.0) is False


@pytest.mark.allow_network
def test_probe_handles_invalid_url():
    """URL が壊れていても例外を投げず ``False`` を返す。"""
    assert is_clef_server_available("not-a-url", timeout_sec=1.0) is False


# ---------------------------------------------------------------------------
# 実際のサーバに対するテスト（自動スキップあり）
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_connection_is_available(live_client):
    """clef API に接続できる。"""
    assert live_client.check_connection() is True


@pytest.mark.integration
def test_clear_mode_is_judged_correctly(live_client):
    """清明モード（視力 1.0）では 4 方向すべて正しく判定できる。"""
    results = []
    for direction in config.DIRECTIONS:
        spec = LandoltCParams(vision=1.0, direction=direction, mode=GenerationMode.CLEAR)
        answer = live_client.inspect_landolt_c(encode_png_base64(generate_landolt_c_image(spec)))
        assert answer is not None, f"{direction} の判定に失敗しました"
        results.append(answer.choice == direction)

    assert all(results), f"判定結果: {results}"


@pytest.mark.integration
@pytest.mark.parametrize("mode", [GenerationMode.MYOPIA, GenerationMode.CATARACT])
def test_degraded_modes_reduce_confidence(live_client, mode):
    """劣化モードは清明モードほど確信を持って判定できない。"""
    confidences = []
    for direction in config.DIRECTIONS:
        spec = LandoltCParams(vision=1.0, direction=direction, mode=mode)
        answer = live_client.inspect_landolt_c(encode_png_base64(generate_landolt_c_image(spec)))
        assert answer is not None
        confidences.append(answer.confidence)

    assert max(confidences) < 0.99
