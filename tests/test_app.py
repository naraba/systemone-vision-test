"""アプリケーション起動（gui.app）のテスト。"""

from __future__ import annotations

from systemone_vision_test import config
from systemone_vision_test.gui import app as app_module


def test_create_application_configures_audio_backend(monkeypatch):
    """``QApplication`` を作る前に音声バックエンドを設定している。"""
    calls: list[str] = []
    monkeypatch.setattr(app_module, "configure_audio_backend", lambda: calls.append("called"))

    app_module.create_application([])

    assert calls == ["called"]


def test_create_application_sets_window_title(qapp):
    """アプリケーション名（ウィンドウタイトル）が設定される。"""
    app = app_module.create_application([])

    assert app.applicationName() == config.WINDOW_TITLE


def test_create_application_reuses_existing_instance(qapp):
    """既に ``QApplication`` があれば作り直さない。"""
    assert app_module.create_application([]) is qapp
