"""アプリケーションのメインウィンドウ（各部の組み立てと検査フローの進行）。"""

from __future__ import annotations

import random

import numpy as np
from PySide6.QtCore import QSize, QThread, QTimer
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QMessageBox, QWidget

from systemone_vision_test import config
from systemone_vision_test.domain.inspection import InspectionEngine
from systemone_vision_test.domain.landolt import (
    GenerationMode,
    LandoltCParams,
    encode_png_base64,
    generate_landolt_c_image,
    generate_sample_image,
)
from systemone_vision_test.gui.canvas_view import CanvasView
from systemone_vision_test.gui.control_panel import ControlPanel
from systemone_vision_test.gui.images import fit_to_box, to_pixmap
from systemone_vision_test.gui.workers import ClefCheckWorker, ClefInspectWorker
from systemone_vision_test.services.answers import DirectionAnswer
from systemone_vision_test.services.clef_client import ClefClient
from systemone_vision_test.services.factory import create_client, is_online_model
from systemone_vision_test.services.sound import SoundEffects
from systemone_vision_test.services.systemone import SystemOneClient


class MainWindow(QMainWindow):
    """アプリケーションのメインウィンドウ。

    役割は「各部品の組み立て」と「検査フローの進行」のみとし、パラメータの保持は
    :class:`ControlPanel`、判定ロジックは :class:`InspectionEngine`、通信は
    :class:`ClefClient`、表示は :class:`CanvasView` に委譲する。

    Args:
        client: clef API のクライアント（既定は新規作成）。
        sounds: 効果音プレイヤ（既定はアプリ共通インスタンス）。
    """

    #: サンプル画像を表示エリアに収めるときに空ける余白（px）
    SAMPLE_MARGIN = 10

    def __init__(
        self,
        client: SystemOneClient | None = None,
        sounds: SoundEffects | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(config.WINDOW_TITLE)
        self.resize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

        self.engine = InspectionEngine()
        self.client = client or ClefClient()
        self.sounds = sounds or SoundEffects.shared()
        self.sounds.available_effects()  # 起動時に素材の有無を確認して警告する

        self._inspection_active = False
        self._workers: list[QThread] = []  # 実行中ワーカーの参照保持（GC 防止）

        self._verdict_timer = QTimer(self)
        self._verdict_timer.setSingleShot(True)
        self._verdict_timer.timeout.connect(self._on_verdict_timeout)

        self._build_ui()
        self._connect_signals()

        # レイアウト確定後にサンプル画像を生成する
        QTimer.singleShot(100, self._refresh_sample)

    # -- 公開プロパティ -----------------------------------------------------

    @property
    def inspection_active(self) -> bool:
        """検査が実行中かどうか。"""
        return self._inspection_active

    # -- 構築 ---------------------------------------------------------------

    def _build_ui(self) -> None:
        """ウィンドウを組み立てる。"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.control_panel = ControlPanel()
        self.canvas = CanvasView()
        layout.addWidget(self.control_panel)
        layout.addWidget(self.canvas)

    def _connect_signals(self) -> None:
        """各部品のシグナルを処理につなぐ。"""
        self.control_panel.start_requested.connect(self._on_start_requested)
        self.control_panel.model_changed.connect(self._on_model_changed)
        self.control_panel.mode_changed.connect(self._on_mode_changed)
        self.control_panel.parameters_changed.connect(self._on_parameters_changed)
        self.canvas.top_overlay.set_model(self.control_panel.selected_model())

    # -- 操作パネルからの通知 -----------------------------------------------

    def _on_model_changed(self, model: str) -> None:
        """使用モデルが切り替わったときの処理。"""
        if is_online_model(model) != is_online_model(self.client.model):
            # ローカル（clef）とインターネット経由（jev）ではクライアントの種類が変わる
            self.client = create_client(model)
        else:
            self.client.model = model
        self.canvas.top_overlay.set_model(model)

    def _on_mode_changed(self, mode: GenerationMode) -> None:
        """生成モードが切り替わったときの処理。

        Args:
            mode: 新しく選択されたモード。
        """
        self._refresh_sample(mode)

    def _on_parameters_changed(
        self, mode: GenerationMode, parameters: dict[str, float | int]
    ) -> None:
        """設定値（スライダー）が変わったときの処理。

        表示中のモードの設定が変わったときだけサンプルを作り直す。

        Args:
            mode: 設定が変わったモード。
            parameters: そのモードの新しいパラメータ（実値）。
        """
        if mode is self.control_panel.selected_mode():
            self._show_sample(generate_sample_image(mode, direction="right", **parameters))

    # -- サンプル表示 -------------------------------------------------------

    def _refresh_sample(self, mode: GenerationMode | None = None) -> None:
        """左パネルのサンプル画像を現在の設定で作り直す。

        Args:
            mode: 対象のモード。``None`` なら操作パネルで選択中のモード。
        """
        target = self.control_panel.selected_mode() if mode is None else mode
        image = generate_sample_image(
            target, direction="right", **self.control_panel.parameters(target)
        )
        self._show_sample(image)

    def _show_sample(self, image: np.ndarray) -> None:
        """サンプル画像を表示エリアに収まる大きさにして表示する。

        表示エリアより大きければ縮小し、小さければ拡大する（縦横比は維持）。

        Args:
            image: 表示するグレースケール画像。
        """
        area = self.control_panel.sample_viewport_size()
        box = QSize(area.width() - self.SAMPLE_MARGIN, area.height() - self.SAMPLE_MARGIN)
        self.control_panel.set_sample_pixmap(fit_to_box(to_pixmap(image), box))

    # -- 検査の開始と終了 ---------------------------------------------------

    def _on_start_requested(self) -> None:
        """「検査開始／検査終了」ボタンが押されたときの処理。"""
        if self._inspection_active:
            self._end_inspection()
            return

        # 接続性の確認は時間がかかるためバックグラウンドで実行する
        self.control_panel.show_checking()
        worker = ClefCheckWorker(self.client)
        worker.checked.connect(self._on_connection_checked)
        self._start_worker(worker)

    def _on_connection_checked(self, connected: bool) -> None:
        """接続性の確認が完了したときの処理。

        Args:
            connected: 接続できたかどうか。
        """
        if not connected:
            QMessageBox.critical(self, "接続エラー", "clef の API エンドポイントが利用できません。")
            self.control_panel.show_idle()
            return
        self._start_inspection()

    def _start_inspection(self) -> None:
        """検査を開始する。"""
        self._inspection_active = True
        self.engine.start()
        self.sounds.stop_all()  # 前の検査の音を持ち越さない
        self.control_panel.show_running()
        self.canvas.top_overlay.hide()
        self.canvas.hide_result()
        self.canvas.show_blank()
        QTimer.singleShot(100, self._next_step)

    def _end_inspection(self) -> None:
        """「検査終了」ボタンが押されたときの処理。

        検査はここで終了し、記録は保持しない（再開もしない）。実行中の通信があっても、
        その結果は :attr:`_inspection_active` で無視する。
        """
        self._stop_inspection()
        self.engine.reset()
        self.control_panel.show_idle()

    def _abort_inspection(self) -> None:
        """通信エラーなどで検査を中断する。"""
        self._stop_inspection()
        self.engine.reset()
        self.control_panel.show_idle()

    def _stop_inspection(self) -> None:
        """検査の進行を止め、検査中の表示を片付ける（終了と中断の共通処理）。"""
        self._inspection_active = False
        self._verdict_timer.stop()
        self.sounds.stop_all()
        self.canvas.top_overlay.hide()
        self.canvas.hide_result()
        self.canvas.show_blank()

    # -- 検査フロー ---------------------------------------------------------

    def _next_step(self) -> None:
        """次の視力値の検査を 1 回分実行する。"""
        if not self._inspection_active:
            return
        if self.engine.is_finished:
            self._show_final_result()
            return

        direction = random.choice(config.DIRECTIONS)
        mode = self.control_panel.selected_mode()
        spec = LandoltCParams(
            vision=self.engine.current_vision, direction=direction, mode=mode
        ).with_parameters(self.control_panel.parameters(mode))
        image = generate_landolt_c_image(spec)

        overlay = self.canvas.top_overlay
        overlay.set_model(self.client.model)
        overlay.set_mode(mode)
        overlay.set_response_time_ms(None)
        overlay.clear_result()
        overlay.show()
        self.canvas.set_image(image)

        QTimer.singleShot(100, lambda: self._send_to_clef(image, direction))

    def _send_to_clef(self, image: np.ndarray, correct_direction: str) -> None:
        """判定リクエストをバックグラウンドで開始する。

        Args:
            image: 判定させる画像。
            correct_direction: 本来の正解方向。
        """
        if not self._inspection_active:
            return
        worker = ClefInspectWorker(self.client, encode_png_base64(image))
        worker.inspected.connect(
            lambda answer, elapsed_ms: self._on_inspected(answer, elapsed_ms, correct_direction)
        )
        self._start_worker(worker)

    def _on_inspected(
        self, answer: DirectionAnswer | None, elapsed_ms: float, correct_direction: str
    ) -> None:
        """判定結果を受け取ったときの処理。

        Args:
            answer: 判定結果。通信エラーなら ``None``。
            elapsed_ms: 判定にかかった時間（ミリ秒）。
            correct_direction: 本来の正解方向。
        """
        if not self._inspection_active:
            return  # 検査が終了／中断されている場合は結果を無視する

        if answer is None:
            QMessageBox.critical(
                self, "通信エラー", f"{self.client.model} API との通信に失敗しました。"
            )
            self._abort_inspection()
            return

        is_correct = answer.choice == correct_direction
        if is_correct:
            self.sounds.play_correct()
        else:
            self.sounds.play_wrong()

        self.engine.record_result(answer.choice, is_correct, answer.probabilities)
        self.canvas.top_overlay.set_response_time_ms(elapsed_ms)
        self.canvas.top_overlay.set_result(
            answer.probabilities, answer.choice, correct_direction, is_correct
        )
        # 正否は中央のオーバーレイではなく棒グラフで示し、3 秒間表示を維持する
        self._verdict_timer.start(config.VERDICT_HOLD_MS)

    def _on_verdict_timeout(self) -> None:
        """正否の表示を維持する時間が経過したときの処理。"""
        self.sounds.stop_all()  # 次の判定までに音を止めておく（残りを持ち越さない）
        QTimer.singleShot(100, self._next_step)

    def _show_final_result(self) -> None:
        """検査終了時の視力値を表示する。"""
        self._inspection_active = False
        self.sounds.stop_all()  # 最後の音を残さない
        self.canvas.show_result(self.engine.final_vision)
        self.control_panel.show_idle()

    # -- ワーカーの管理 -----------------------------------------------------

    def _start_worker(self, worker: QThread) -> None:
        """ワーカーを起動し、終了するまで参照を保持する。

        Args:
            worker: 起動するワーカー。
        """
        self._workers.append(worker)
        worker.finished.connect(lambda: self._workers.remove(worker))
        worker.start()

    def closeEvent(self, event) -> None:
        """ウィンドウ終了時、実行中のワーカーの完了を待つ。

        Args:
            event: クローズイベント。
        """
        for worker in list(self._workers):
            worker.wait(5000)
        super().closeEvent(event)
