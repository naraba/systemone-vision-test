"""clef API への通信をバックグラウンドで行うワーカー。

モデルロードには数十秒かかることがあるため、GUI（メインスレッド）を
ブロックしないよう別スレッドで実行する。
"""

from __future__ import annotations

import time

from PySide6.QtCore import QThread, Signal

from systemone_vision_test.services.answers import DirectionAnswer
from systemone_vision_test.services.systemone import SystemOneClient


class ClefCheckWorker(QThread):
    """clef API の接続性確認を実行するワーカー。

    Args:
        client: 使用するクライアント。
        parent: 親オブジェクト。

    Signals:
        checked: 接続できたかどうか。
    """

    checked = Signal(bool)

    def __init__(self, client: SystemOneClient, parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        """接続性を確認し、結果を通知する。"""
        self.checked.emit(self._client.check_connection())


class ClefInspectWorker(QThread):
    """ランドルト環画像の判定を実行するワーカー。

    Args:
        client: 使用するクライアント。
        image_base64: PNG を Base64 エンコードした文字列。
        parent: 親オブジェクト。

    Signals:
        inspected: ``(判定結果または None, 応答時間ミリ秒)``。
    """

    inspected = Signal(object, float)

    def __init__(self, client: SystemOneClient, image_base64: str, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._image_base64 = image_base64

    def run(self) -> None:
        """判定リクエストを送信し、結果と応答時間を通知する。"""
        started = time.perf_counter()
        answer: DirectionAnswer | None = self._client.inspect_landolt_c(self._image_base64)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.inspected.emit(answer, elapsed_ms)
