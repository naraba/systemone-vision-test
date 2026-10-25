"""System One による視力検査もどき。

パッケージ構成:

* :mod:`systemone_vision_test.config` — パス・定数・既定値（唯一の真実の源）
* :mod:`systemone_vision_test.domain` — GUI に依存しない中核ロジック
* :mod:`systemone_vision_test.services` — 外部 I/O（clef API 通信、効果音）
* :mod:`systemone_vision_test.gui` — PySide6 による画面

起動方法::

    python -m systemone_vision_test
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
