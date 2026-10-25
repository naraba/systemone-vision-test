"""プログラムが外部へ通信する経路を限定できていることを確かめるテスト。

方針: **HTTP を扱うのは System One API のクライアントだけ**とし、それ以外の module は
ネットワークに触れない（ローカル外の通信は System One のエンドポイントのみ）。
"""

from __future__ import annotations

import ast
from pathlib import Path

from systemone_vision_test import config
from systemone_vision_test.services.clef_client import ClefClient
from systemone_vision_test.services.factory import create_client
from systemone_vision_test.services.jev_client import JevClient
from systemone_vision_test.services.systemone import prefer_ipv4_url

SRC = Path(__file__).resolve().parents[1] / "src" / "systemone_vision_test"

#: HTTP を触ってよい module（System One API のクライアントとその共通部分）
ALLOWED_HTTP_MODULES = {
    "services/systemone.py",
    "services/clef_client.py",
    "services/jev_client.py",
}

#: 使ってはいけない通信ライブラリ（``requests`` 以外の経路を作らない）
FORBIDDEN_IMPORTS = ("urllib.request", "http.client", "socket", "httpx", "aiohttp")

#: 使ってはいけない Qt の通信クラス（GUI から外部へ出さない。ローカルの音声再生は除く）
FORBIDDEN_QT_NAMES = (
    "QNetworkAccessManager",
    "QNetworkRequest",
    "QNetworkReply",
    "QWebEngineView",
    "QWebEnginePage",
    "QWebSocket",
    "QTcpSocket",
    "QUdpSocket",
    "QHostInfo",
    "QMediaStreamsControl",
)


def iter_modules() -> list[Path]:
    """``src`` 配下の module を返す。"""
    return sorted(SRC.rglob("*.py"))


def imports_of(path: Path) -> set[str]:
    """module が import している名前を集める（``import x`` / ``from x import y``）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def touches_http(path: Path) -> bool:
    """``requests`` を使っている module かどうか（送信に限らない）。"""
    return any(name == "requests" or name.startswith("requests.") for name in imports_of(path))


def test_only_the_api_clients_use_requests():
    """``requests`` を使う module は API クライアントだけに限定されている。"""
    users = {path.relative_to(SRC).as_posix() for path in iter_modules() if touches_http(path)}

    assert users <= ALLOWED_HTTP_MODULES, f"HTTP を使う module が増えています: {sorted(users)}"


def test_no_other_network_libraries_are_used():
    """``requests`` 以外の通信ライブラリを import していない。"""
    offenders: dict[str, set[str]] = {}
    for path in iter_modules():
        found = {
            name
            for name in imports_of(path)
            if any(
                name == forbidden or name.startswith(f"{forbidden}.")
                for forbidden in FORBIDDEN_IMPORTS
            )
        }
        if found:
            offenders[path.relative_to(SRC).as_posix()] = found

    assert offenders == {}


def test_qt_network_classes_are_not_used():
    """Qt の通信クラス（ネットワーク・WebEngine）を使っていない。"""
    offenders: dict[str, set[str]] = {}
    for path in iter_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            node.attr if isinstance(node, ast.Attribute) else node.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Attribute, ast.Name))
        }
        found = names & set(FORBIDDEN_QT_NAMES)
        if found:
            offenders[path.relative_to(SRC).as_posix()] = found

    assert offenders == {}


def test_clients_only_target_the_configured_endpoints():
    """クライアントの接続先は config の System One エンドポイントだけ。"""
    local = ClefClient()
    online = JevClient(credential="dummy-key")

    assert local.request_url == prefer_ipv4_url(config.CLEF_BASE_URL)
    assert online.request_url == config.JEV_BASE_URL
    assert create_client("clef").request_url == prefer_ipv4_url(config.CLEF_BASE_URL)
    assert create_client("jev").request_url == config.JEV_BASE_URL
    # モデル一覧の取得も同じ接続先から導かれる（別ホストへは出ない）
    assert online.models_url == f"{config.JEV_BASE_URL.removesuffix('/systemone')}/models"
