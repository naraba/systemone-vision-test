"""メインウィンドウ（gui.main_window）の統合的なテスト。

通信はダミークライアントに差し替え、イベントループを手動で回して検証する。
"""

from __future__ import annotations

import pytest

from systemone_vision_test import config
from systemone_vision_test.domain.inspection import InspectionEngine
from systemone_vision_test.domain.landolt import GenerationMode
from systemone_vision_test.gui.main_window import MainWindow
from systemone_vision_test.services.sound import SoundEffect


class RecordingSounds:
    """再生要求を記録するダミーの効果音プレイヤ。"""

    def __init__(self) -> None:
        self.played: list[SoundEffect] = []
        self.stop_calls = 0

    def available_effects(self) -> tuple[SoundEffect, ...]:
        return ()

    def play_correct(self) -> bool:
        self.played.append(SoundEffect.CORRECT)
        return True

    def play_wrong(self) -> bool:
        self.played.append(SoundEffect.WRONG)
        return True

    def stop_all(self) -> None:
        self.stop_calls += 1


class RecordingSession:
    """``requests.Session`` の代替（送信内容を記録する）。

    Args:
        payload: 返す JSON。
    """

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []
        self.trust_env = True

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        return _Response(self.payload)


class _Response:
    """``requests.Response`` の最小限の代替。"""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        """固定の JSON を返す。"""
        return self._payload


@pytest.fixture
def window(qapp, fake_client_factory):
    """ダミークライアントを挿した ``MainWindow`` を作る。"""

    def _create(**client_kwargs) -> MainWindow:
        w = MainWindow(client=fake_client_factory(**client_kwargs))
        w.show()
        qapp.processEvents()
        return w

    yield _create


@pytest.fixture
def fast_verdict(monkeypatch):
    """正否の表示維持時間を短くしてテストを速くする。"""
    monkeypatch.setattr(config, "VERDICT_HOLD_MS", 1)


def widget_y(widget) -> int:
    """親（操作パネル）から見た縦位置を返す。"""
    return widget.mapTo(widget.parentWidget(), widget.rect().topLeft()).y()


# ---------------------------------------------------------------------------
# 起動直後の状態
# ---------------------------------------------------------------------------


def test_initial_state_is_idle(window):
    """起動直後は「検査開始」で、操作できる状態。"""
    w = window()
    try:
        assert w.control_panel.btn_start.text() == "検査開始"
        assert w.control_panel.btn_start.isEnabled()
        assert w.inspection_active is False
        assert w.engine.state.status.value == "idle"
        assert w.control_panel.combo_model.currentText() == config.DEFAULT_MODEL
    finally:
        w.close()


def test_default_window_shows_canvas_at_native_size(window, process_events):
    """既定のウィンドウサイズでは、キャンバスが原寸（1:1）で表示される。"""
    w = window()
    try:
        process_events(0.3)
        w.canvas.show_blank()
        process_events(0.2)
        pixmap = w.canvas.displayed_pixmap()
        viewport = w.canvas.viewport()

        assert (viewport.width(), viewport.height()) == (
            config.CANVAS_SIZE,
            config.CANVAS_SIZE,
        )
        assert (pixmap.width(), pixmap.height()) == (
            config.CANVAS_SIZE,
            config.CANVAS_SIZE,
        )
    finally:
        w.close()


def test_window_default_size_is_derived_from_canvas():
    """既定ウィンドウサイズはキャンバス原寸から導出されている。"""
    assert config.WINDOW_WIDTH == config.CONTROL_PANEL_WIDTH + config.CANVAS_SIZE
    assert config.WINDOW_HEIGHT == config.CANVAS_SIZE


def test_sample_image_is_generated(window, process_events):
    """サンプル画像が表示される。"""
    w = window()
    try:
        process_events(0.4)
        pixmap = w.control_panel.sample_label.pixmap()
        assert pixmap is not None and not pixmap.isNull()
    finally:
        w.close()


def test_sample_pixmap_fits_display_area(window, process_events):
    """表示するサンプル画像は表示エリアに収まる大きさになる。"""
    w = window()
    try:
        process_events(0.4)
        pixmap = w.control_panel.sample_label.pixmap()
        area = w.control_panel.sample_viewport_size()

        assert pixmap is not None
        assert pixmap.width() <= area.width()
        assert pixmap.height() <= area.height()
        # 表示エリアの大きさに合わせて縮小されている（キャンバスは 640x640）
        assert pixmap.width() < config.CANVAS_SIZE
    finally:
        w.close()


def test_sample_updates_when_parameters_change(window, qapp, process_events):
    """設定値を変えるとサンプル画像が作り直される。"""
    w = window()
    try:
        w.control_panel.combo_mode.setCurrentText(GenerationMode.MYOPIA.value)
        qapp.processEvents()
        process_events(0.3)
        sharp = w.control_panel.sample_label.pixmap().toImage().copy()

        w.control_panel._sliders["gaussian_ksize"].slider.setValue(121)
        qapp.processEvents()
        process_events(0.3)
        blurred = w.control_panel.sample_label.pixmap().toImage().copy()

        assert sharp != blurred
    finally:
        w.close()


# ---------------------------------------------------------------------------
# 検査の開始・実行・終了
# ---------------------------------------------------------------------------


def test_start_checks_connection_then_runs(window, qapp, process_events):
    """接続確認を経て検査が始まり、ボタンが「検査終了」になる。"""
    w = window()
    try:
        w.control_panel.btn_start.click()
        qapp.processEvents()
        assert w.control_panel.btn_start.text() == "接続確認中..."
        assert not w.control_panel.btn_start.isEnabled()

        process_events(1.0)
        assert w.inspection_active is True
        assert w.control_panel.btn_start.text() == "検査終了"
        assert w.control_panel.btn_start.isEnabled()
    finally:
        w.close()


def test_inspection_does_not_block_event_loop(window, qapp, process_events):
    """接続確認中もイベントループが回り続ける（メインスレッドを止めない）。"""
    from PySide6.QtCore import QTimer

    w = window(delay_sec=0.5)
    try:
        ticks = {"count": 0}
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.__setitem__("count", ticks["count"] + 1))
        timer.start(20)

        w.control_panel.btn_start.click()
        process_events(1.5)

        assert ticks["count"] > 10
        assert w.inspection_active is True
    finally:
        w.close()


def test_controls_are_disabled_while_running(window, qapp, process_events):
    """検査中はモデル・モード・スライダーが操作できない。"""
    w = window()
    try:
        w.control_panel.combo_mode.setCurrentText(GenerationMode.MYOPIA.value)
        qapp.processEvents()
        w.control_panel.btn_start.click()
        process_events(1.0)

        assert w.control_panel.combo_model.isEnabled() is False
        assert w.control_panel.combo_mode.isEnabled() is False
        assert w.control_panel.btn_reset_defaults.isEnabled() is False
    finally:
        w.close()


def test_end_button_finishes_immediately(window, qapp, process_events):
    """「検査終了」で即座に終わり、記録は残さない（再開しない）。"""
    w = window()
    try:
        w.control_panel.btn_start.click()
        process_events(1.0)
        assert w.inspection_active is True

        w.control_panel.btn_start.click()
        qapp.processEvents()

        assert w.inspection_active is False
        assert w.control_panel.btn_start.text() == "検査開始"
        assert w.control_panel.combo_mode.isEnabled() is True
        assert w.engine.state.status.value == "idle"
        assert w.canvas.top_overlay.isVisible() is False
    finally:
        w.close()


def test_late_result_is_ignored_after_end(window, process_events):
    """検査終了後に届いた結果は無視する（次に進まない）。"""
    w = window(delay_sec=0.4)
    try:
        w.control_panel.btn_start.click()
        process_events(1.0)
        assert w.inspection_active is True
        calls_before = w.client.inspect_calls

        w.control_panel.btn_start.click()  # 判定の途中で終了する
        process_events(1.5)

        assert w.inspection_active is False
        assert w.client.inspect_calls == calls_before
        assert w.control_panel.btn_start.text() == "検査開始"
    finally:
        w.close()


def test_top_overlay_shows_model_mode_and_time(window, qapp, process_events):
    """検査中は上辺のオーバーレイにモデル・モード・応答時間が出る。"""
    w = window()
    try:
        w.control_panel.combo_mode.setCurrentText(GenerationMode.ASTIGMATISM.value)
        qapp.processEvents()
        w.control_panel.btn_start.click()
        process_events(1.5)

        assert w.canvas.top_overlay.isVisible() is True
        assert w.canvas.top_overlay.lbl_model.text() == f"モデル: {config.DEFAULT_MODEL}"
        assert w.canvas.top_overlay.lbl_mode.text() == "生成モード: 乱視"
        assert w.canvas.top_overlay.lbl_response_time.text().startswith("API応答時間:")
        assert w.canvas.top_overlay.graph.answer is not None
    finally:
        w.close()


def test_verdict_plays_matching_sound(qapp, process_events, fake_client_factory):
    """正誤判定に応じた効果音を鳴らす。"""
    sounds = RecordingSounds()
    w = MainWindow(client=fake_client_factory(delay_sec=0.05), sounds=sounds)
    w.show()
    try:
        w.control_panel.btn_start.click()
        process_events(1.5)

        assert len(sounds.played) == 1
        expected = (
            SoundEffect.CORRECT if w.canvas.top_overlay.graph.is_correct else SoundEffect.WRONG
        )
        assert sounds.played[0] is expected
    finally:
        w.close()


def test_sounds_are_stopped_at_inspection_boundaries(qapp, process_events, fake_client_factory):
    """検査の開始・終了で効果音を止める（前の検査の音を持ち越さない）。"""
    sounds = RecordingSounds()
    w = MainWindow(client=fake_client_factory(delay_sec=0.05), sounds=sounds)
    w.show()
    try:
        w.control_panel.btn_start.click()
        process_events(1.5)
        assert sounds.stop_calls >= 1  # 検査開始時に停止

        w.control_panel.btn_start.click()
        process_events(0.3)
        assert sounds.stop_calls >= 2  # 検査終了時にも停止
    finally:
        w.close()


def test_inspection_shows_final_result(window, process_events, fast_verdict):
    """検査が終わると中央のオーバーレイに視力値が出る。"""
    w = window(delay_sec=0.01)
    try:
        w.engine = InspectionEngine(vision_levels=(0.1,))  # 1 段階だけの検査
        w.control_panel.btn_start.click()
        process_events(3.0)

        assert w.inspection_active is False
        assert w.canvas.result_overlay.isVisible() is True
        assert w.canvas.result_overlay.label.text().startswith("視力：")
        assert w.control_panel.btn_start.text() == "検査開始"
    finally:
        w.close()


def test_final_result_uses_engine_value(window):
    """確定した視力値がそのまま表示される。"""
    w = window()
    try:
        w.engine = InspectionEngine(vision_levels=(1.5, 1.0))
        w.engine.start()
        w.engine.record_result("up", True)  # 1.0 -> 1.5
        w.engine.record_result("up", True)  # 上限に到達して終了

        w._show_final_result()

        assert w.canvas.result_overlay.isVisible() is True
        assert w.canvas.result_overlay.label.text() == "視力：1.5"
    finally:
        w.close()


# ---------------------------------------------------------------------------
# モデル選択
# ---------------------------------------------------------------------------


def test_selecting_online_model_switches_the_client(qapp):
    """jev を選ぶとインターネット経由のクライアントに切り替わり、clef に戻すと戻る。"""
    from systemone_vision_test.services.clef_client import ClefClient
    from systemone_vision_test.services.jev_client import JevClient

    w = MainWindow(client=ClefClient(), sounds=RecordingSounds())
    try:
        w.control_panel.combo_model.setCurrentText("jev")
        qapp.processEvents()
        assert isinstance(w.client, JevClient)
        assert w.canvas.top_overlay.lbl_model.text() == "モデル: jev"

        w.control_panel.combo_model.setCurrentText("clef-flash")
        qapp.processEvents()
        assert isinstance(w.client, ClefClient)
        assert not isinstance(w.client, JevClient)
        assert w.client.model == "clef-flash"
    finally:
        w.close()


def test_model_selector_lists_models_in_order(window, qapp):
    """モデルの選択肢と既定値、生成モードより上にあること。"""
    w = window()
    try:
        combo = w.control_panel.combo_model
        items = [combo.itemText(i) for i in range(combo.count())]

        assert items == list(config.AVAILABLE_MODELS)
        assert combo.currentText() == config.DEFAULT_MODEL

        qapp.processEvents()
        assert widget_y(combo) < widget_y(w.control_panel.combo_mode)
    finally:
        w.close()


def test_model_selection_reaches_the_api_payload(qapp):
    """選択したモデルが接続確認と判定の両方のリクエストに使われる。"""
    from systemone_vision_test.services.clef_client import ClefClient

    session = RecordingSession({"answers": {"check_color": {"choice": "red"}}})
    w = MainWindow(client=ClefClient(session=session), sounds=RecordingSounds())
    w.show()
    try:
        w.control_panel.combo_model.setCurrentText("clef-flash")
        qapp.processEvents()
        assert w.client.model == "clef-flash"
        assert w.canvas.top_overlay.lbl_model.text() == "モデル: clef-flash"

        w.client.check_connection()
        w.client.inspect_landolt_c("BASE64-IMAGE")
        assert [call["json"]["model"] for call in session.calls] == [
            "clef-flash",
            "clef-flash",
        ]
    finally:
        w.close()


# ---------------------------------------------------------------------------
# 操作パネル（パラメータ）
# ---------------------------------------------------------------------------


def test_only_current_mode_sliders_are_visible(window, qapp):
    """選択中のモードで使うスライダーだけが表示される。"""
    w = window()
    panel = w.control_panel
    try:
        panel.combo_mode.setCurrentText(GenerationMode.CLEAR.value)
        qapp.processEvents()
        assert panel.settings_frame.isVisible() is False

        panel.combo_mode.setCurrentText(GenerationMode.MYOPIA.value)
        qapp.processEvents()
        visible = {key for key, row in panel._sliders.items() if row.slider.isVisible()}
        assert visible == {"gaussian_ksize"}
    finally:
        w.close()


def test_parameters_are_kept_per_mode(window, qapp):
    """モードごとにパラメータを独立して保持し、切り替えると復元する。"""
    w = window()
    panel = w.control_panel
    try:
        panel.combo_mode.setCurrentText(GenerationMode.CATARACT.value)
        qapp.processEvents()
        panel._sliders["box_ksize"].slider.setValue(11)
        qapp.processEvents()
        assert panel.parameters(GenerationMode.CATARACT)["box_ksize"] == 11

        panel.combo_mode.setCurrentText(GenerationMode.MYOPIA.value)
        qapp.processEvents()
        panel.combo_mode.setCurrentText(GenerationMode.CATARACT.value)
        qapp.processEvents()

        assert panel.parameters(GenerationMode.CATARACT)["box_ksize"] == 11
        assert panel._sliders["box_ksize"].slider.value() == 11
    finally:
        w.close()


def test_reset_defaults_restores_mode_defaults(window, qapp):
    """「デフォルトに戻す」で現在のモードの既定値に戻る。"""
    from systemone_vision_test.domain.landolt import default_parameters

    w = window()
    panel = w.control_panel
    try:
        panel.combo_mode.setCurrentText(GenerationMode.CATARACT.value)
        qapp.processEvents()
        panel._sliders["contrast"].slider.setValue(90)
        qapp.processEvents()

        panel.btn_reset_defaults.click()
        qapp.processEvents()

        assert panel.parameters(GenerationMode.CATARACT) == default_parameters(
            GenerationMode.CATARACT
        )
    finally:
        w.close()
