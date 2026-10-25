"""pytest の共通フィクスチャ。

* GUI のテストは画面なし（offscreen）で動かし、モーダルダイアログで停止しないようにする
* 既定のテストが実際のネットワークへ出ないよう HTTP 呼び出しを禁止する
  （``integration`` / ``allow_network`` マーカーが付いたテストだけが例外）
* jev の設定（接続先・モデル・種類・API キー）は環境変数で上書きできるため、
  テストのあいだは既定値に固定する（開発中に設定した値でテスト結果が変わらないようにする）
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import requests

# QApplication を作る前に設定する必要があるため、Qt の import より前に置く
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: I001

from systemone_vision_test import config
from systemone_vision_test.domain.questions import DIRECTION_QUESTION, DirectionQuestion
from systemone_vision_test.services.answers import DirectionAnswer
from systemone_vision_test.services.clef_client import ClefClient
from systemone_vision_test.services.sound import SoundEffects


#: jev の設定を読む環境変数。テストでは既定値で検証するため、すべて外す。
JEV_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "JEV_BASE_URL",
    "JEV_MODEL",
    "JEV_PROVIDER",
    "JEV_API_KEY",
    "JEV_API_KEY_ENV",
    "JEV_API_KEY_FILE",
    "JEV_IMAGE_ASSET_REFERENCE",
    "JEV_IMAGE_FORMAT",
    "JEV_TIMEOUT_SEC",
    "JEV_AUTH_HEADER",
    "JEV_AUTH_SCHEME",
    "JEV_SEND_DANGER_ZONE_HEADER",
)

#: :mod:`systemone_vision_test.config` の既定値
#: （環境変数で上書きされていても、テストではこの値を使う）
DEFAULT_JEV_BASE_URL = "https://openrouter.ai/api/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"


@pytest.fixture(autouse=True)
def jev_settings_are_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """jev の設定を既定値に固定する。

    :mod:`systemone_vision_test.config` は import 時に環境変数を読むので、``JEV_PROVIDER`` や
    ``JEV_BASE_URL`` を設定したままテストを実行すると結果が変わってしまう（子プロセスへ
    渡る環境も同様）。環境変数を外し、読み込み済みの設定値も既定に戻す。
    """
    for name in JEV_ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "JEV_BASE_URL", DEFAULT_JEV_BASE_URL, raising=False)
    monkeypatch.setattr(config, "JEV_MODEL", DEFAULT_JEV_MODEL, raising=False)
    monkeypatch.setattr(config, "JEV_PROVIDER", "", raising=False)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """テスト全体で共有する ``QApplication`` を用意する。

    Returns:
        アプリケーションインスタンス。
    """
    app = QApplication.instance() or QApplication([])
    # モーダルダイアログでテストが止まらないように差し替える
    QMessageBox.critical = staticmethod(lambda *args, **kwargs: None)  # type: ignore[assignment]
    QMessageBox.information = staticmethod(lambda *args, **kwargs: None)  # type: ignore[assignment]
    return app


@pytest.fixture
def process_events(qapp: QApplication):
    """指定秒数のあいだイベントループを回す関数を返す。"""

    def _process(seconds: float = 0.2) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)

    return _process


@pytest.fixture
def artifacts_dir() -> Path:
    """テストの出力（画像など）を保存するディレクトリを返す。"""
    directory = config.ARTIFACTS_DIR / "tests"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(autouse=True)
def _forbid_real_network(request, monkeypatch):
    """既定のテストが実際のネットワークへ出ないようにする。

    API 呼び出しはスタブ（``monkeypatch`` で差し替え）を使うこと。
    実サーバを使うテストには ``integration``、到達不可アドレスへの確認など
    環境に依存しないネットワーク I/O には ``allow_network`` を付ける。
    """
    markers = ("integration", "allow_network")
    if any(request.node.get_closest_marker(name) for name in markers):
        return

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "テストから実際のネットワーク呼び出しが発生しました。"
            "API 呼び出しはスタブに差し替えてください。"
        )

    # requests のすべての入口（モジュール関数と Session）を塞ぐ
    monkeypatch.setattr(requests, "post", _forbidden)
    monkeypatch.setattr(requests, "get", _forbidden)
    monkeypatch.setattr(requests.Session, "request", _forbidden)


@pytest.fixture(autouse=True)
def _reset_sound_shared():
    """各テストで効果音の共有インスタンスを初期化する。"""
    SoundEffects.reset_shared()
    yield
    SoundEffects.reset_shared()


class FakeClefClient(ClefClient):
    """応答を即座に返すダミークライアント。

    Args:
        answer: 判定で返す方向。``None`` なら通信エラーを模擬する。
        probability: 回答方向の確率。
        delay_sec: 1 リクエストあたりの擬似的な所要時間（秒）。
    """

    def __init__(
        self,
        answer: str | None = "up",
        probability: float = 0.9,
        delay_sec: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.answer = answer
        self.probability = probability
        self.delay_sec = delay_sec
        self.inspect_calls = 0
        self.check_calls = 0
        #: 直近の判定で使われた質問（呼び出し側の指定が届いているかの確認用）
        self.last_question = DIRECTION_QUESTION

    def check_connection(self) -> bool:
        self.check_calls += 1
        time.sleep(self.delay_sec)
        return self.answer is not None

    def inspect_landolt_c(
        self,
        image_base64: str,
        *,
        question: DirectionQuestion = DIRECTION_QUESTION,
    ) -> DirectionAnswer | None:
        self.inspect_calls += 1
        self.last_question = question
        time.sleep(self.delay_sec)
        if self.answer is None:
            return None
        probabilities = dict.fromkeys(config.DIRECTIONS, 0.01)
        probabilities[self.answer] = self.probability
        return DirectionAnswer(
            choice=self.answer,
            probabilities=probabilities,
            confidence=self.probability,
        )


@pytest.fixture
def fake_client_factory():
    """``FakeClefClient`` を作るファクトリを返す。"""

    def _create(**kwargs) -> FakeClefClient:
        return FakeClefClient(**kwargs)

    return _create
