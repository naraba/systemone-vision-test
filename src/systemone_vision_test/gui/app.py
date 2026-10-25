"""アプリケーションの起動（エントリーポイント）。"""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from systemone_vision_test import config
from systemone_vision_test.gui.main_window import MainWindow
from systemone_vision_test.services.sound import configure_audio_backend

logger = logging.getLogger(__name__)


def create_application(argv: list[str] | None = None) -> QApplication:
    """``QApplication`` を用意する（既にあればそれを返す）。

    音声バックエンドの環境変数は QtMultimedia の初期化時に読まれるため、
    ``QApplication`` を作る前に :func:`configure_audio_backend` を呼ぶ。

    Args:
        argv: コマンドライン引数（``None`` なら ``sys.argv``）。

    Returns:
        アプリケーションインスタンス。
    """
    logger.info("音声バックエンド: %s", configure_audio_backend())
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName(config.WINDOW_TITLE)
    return app


def main(argv: list[str] | None = None) -> int:
    """アプリケーションを起動する。

    Args:
        argv: コマンドライン引数（``None`` なら ``sys.argv``）。

    Returns:
        終了コード。
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = create_application(argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - 直接実行用
    raise SystemExit(main())
