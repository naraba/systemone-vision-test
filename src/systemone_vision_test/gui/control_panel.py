"""左の操作用パネル（モデル選択・生成モード選択・サンプル表示・設定値調整）。"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from systemone_vision_test import config
from systemone_vision_test.domain.landolt import (
    GenerationMode,
    ParameterSpec,
    all_default_parameters,
    parameter_specs,
)
from systemone_vision_test.gui import style


@dataclass
class _SliderRow:
    """パラメータ 1 つ分のラベルとスライダー。"""

    spec: ParameterSpec
    label: QLabel
    slider: QSlider

    def set_visible(self, visible: bool) -> None:
        """表示・非表示を切り替える。"""
        self.label.setVisible(visible)
        self.slider.setVisible(visible)

    def set_enabled(self, enabled: bool) -> None:
        """操作の有効・無効を切り替える（ラベルも一緒に薄くする）。"""
        self.label.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def set_value(self, value: float) -> None:
        """実値をスライダーに反映する（シグナルは発火させない）。"""
        was_blocked = self.slider.blockSignals(True)
        self.slider.setValue(self.spec.to_slider(value))
        self.slider.blockSignals(was_blocked)

    def value(self) -> float | int:
        """スライダーの現在値を実値で返す。"""
        return self.spec.to_value(self.slider.value())


class ControlPanel(QFrame):
    """左の操作用パネル。

    Signals:
        start_requested: 「検査開始／検査終了」ボタンが押された。
        model_changed: モデルの選択が変わった（新しいモデル名）。
        mode_changed: 生成モードの選択が変わった（新しいモード）。
        parameters_changed: スライダーが操作された（``(モード, パラメータ)``）。
    """

    start_requested = Signal()
    model_changed = Signal(str)
    mode_changed = Signal(object)
    parameters_changed = Signal(object, dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setFixedWidth(config.CONTROL_PANEL_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        #: モードごとの現在のパラメータ（実値）
        self._parameters: dict[GenerationMode, dict[str, float | int]] = all_default_parameters()
        self._mode = GenerationMode.CLEAR
        self._sliders: dict[str, _SliderRow] = {}

        self._build_ui()
        self._update_slider_visibility()

    # -- 構築 ---------------------------------------------------------------

    def _build_ui(self) -> None:
        """ウィジェットを組み立てる。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.btn_start = QPushButton("検査開始")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self.start_requested.emit)
        layout.addWidget(self.btn_start)

        layout.addWidget(QLabel("モデル"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(list(config.AVAILABLE_MODELS))
        self.combo_model.setCurrentText(config.DEFAULT_MODEL)
        self.combo_model.currentTextChanged.connect(self.model_changed.emit)
        layout.addWidget(self.combo_model)

        layout.addWidget(QLabel("生成モード"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([mode.value for mode in GenerationMode])
        self.combo_mode.setCurrentText(GenerationMode.CLEAR.value)
        self.combo_mode.currentTextChanged.connect(self._on_mode_selected)
        layout.addWidget(self.combo_mode)

        layout.addWidget(QLabel("サンプル"))
        layout.addWidget(self._build_sample_area())
        layout.addWidget(self._build_settings_area())
        layout.addStretch()

    def _build_sample_area(self) -> QScrollArea:
        """サンプル画像の表示エリアを作る。"""
        self.sample_scroll = QScrollArea()
        self.sample_scroll.setWidgetResizable(True)
        self.sample_scroll.setFrameStyle(QFrame.Shape.StyledPanel)
        self.sample_scroll.setMinimumHeight(200)

        self.sample_label = QLabel()
        self.sample_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_label.setStyleSheet(style.SAMPLE_STYLE)
        self.sample_scroll.setWidget(self.sample_label)
        return self.sample_scroll

    def _build_settings_area(self) -> QFrame:
        """設定値調整エリア（全モードのスライダーを保持し、必要なものだけ表示）を作る。"""
        self.settings_frame = QFrame()
        self.settings_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self.settings_frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("設定値")
        title.setStyleSheet(style.BOLD_STYLE)
        header.addWidget(title)
        header.addStretch()
        self.btn_reset_defaults = QPushButton("デフォルトに戻す")
        self.btn_reset_defaults.setFixedHeight(24)
        self.btn_reset_defaults.clicked.connect(self._on_reset_requested)
        header.addWidget(self.btn_reset_defaults)
        layout.addLayout(header)

        # モードごとのスライダーを作り、あとで表示を切り替える
        for mode in GenerationMode:
            for spec in parameter_specs(mode):
                row = self._create_slider(spec)
                self._sliders[spec.key] = row
                layout.addWidget(row.label)
                layout.addWidget(row.slider)

        return self.settings_frame

    def _create_slider(self, spec: ParameterSpec) -> _SliderRow:
        """スライダー 1 つを作る。"""
        label = QLabel(self._label_text(spec, spec.default))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(spec.minimum)
        slider.setMaximum(spec.maximum)
        slider.setValue(spec.default)
        slider.setSingleStep(spec.step)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(spec.step)

        row = _SliderRow(spec=spec, label=label, slider=slider)
        slider.valueChanged.connect(lambda _value, r=row: self._on_slider_changed(r))
        return row

    # -- 公開 API（読み取り） -----------------------------------------------

    def selected_model(self) -> str:
        """選択中のモデル名を返す。"""
        return self.combo_model.currentText()

    def selected_mode(self) -> GenerationMode:
        """選択中の生成モードを返す。"""
        return self._mode

    def parameters(self, mode: GenerationMode | None = None) -> dict[str, float | int]:
        """指定モードの現在のパラメータ（実値）を返す。

        Args:
            mode: 対象のモード。``None`` なら選択中のモード。
        """
        target = self._mode if mode is None else mode
        return dict(self._parameters[target])

    def sample_viewport_size(self) -> QSize:
        """サンプル表示エリアの大きさを返す。

        Returns:
            サンプル画像を収める領域の大きさ（px）。
        """
        return self.sample_scroll.size()

    # -- 公開 API（操作） ---------------------------------------------------

    def set_sample_pixmap(self, pixmap: QPixmap) -> None:
        """サンプル画像を表示する。"""
        self.sample_label.setPixmap(pixmap)

    def show_idle(self) -> None:
        """待機中の見た目にする（検査開始ボタンが有効、他も操作可能）。"""
        self.btn_start.setEnabled(True)
        self.btn_start.setText("検査開始")
        self._set_controls_enabled(True)

    def show_checking(self) -> None:
        """接続確認中の見た目にする。"""
        self.btn_start.setEnabled(False)
        self.btn_start.setText("接続確認中...")

    def show_running(self) -> None:
        """検査実行中の見た目にする（「検査終了」ボタンだけが有効）。"""
        self.btn_start.setEnabled(True)
        self.btn_start.setText("検査終了")
        self._set_controls_enabled(False)

    # -- 内部処理 -----------------------------------------------------------

    @staticmethod
    def _label_text(spec: ParameterSpec, slider_value: int) -> str:
        """スライダーのラベル文字列を組み立てる。"""
        return f"{spec.label}: {slider_value}"

    def _on_slider_changed(self, row: _SliderRow) -> None:
        """スライダーが操作されたときの処理。"""
        row.label.setText(self._label_text(row.spec, row.slider.value()))
        self._parameters[self._mode][row.spec.key] = row.value()
        self.parameters_changed.emit(self._mode, self.parameters())

    def _on_mode_selected(self, text: str) -> None:
        """生成モードの選択が変わったときの処理。"""
        self._mode = GenerationMode(text)
        self._update_slider_visibility()
        self._apply_parameters_to_sliders()
        self.mode_changed.emit(self._mode)

    def _on_reset_requested(self) -> None:
        """「デフォルトに戻す」が押されたときの処理。"""
        self._parameters[self._mode] = all_default_parameters()[self._mode]
        self._apply_parameters_to_sliders()
        self.parameters_changed.emit(self._mode, self.parameters())

    def _update_slider_visibility(self) -> None:
        """選択中のモードで使うスライダーだけを表示する。"""
        visible_keys = {spec.key for spec in parameter_specs(self._mode)}
        for key, row in self._sliders.items():
            row.set_visible(key in visible_keys)
        self.settings_frame.setVisible(bool(visible_keys))

    def _apply_parameters_to_sliders(self) -> None:
        """選択中のモードのパラメータをスライダーに反映する。"""
        for key, value in self._parameters[self._mode].items():
            row = self._sliders.get(key)
            if row is not None:
                row.set_value(value)
                row.label.setText(self._label_text(row.spec, row.slider.value()))

    def _set_controls_enabled(self, enabled: bool) -> None:
        """モード選択とスライダーの有効・無効をまとめて切り替える。"""
        self.combo_model.setEnabled(enabled)
        self.combo_mode.setEnabled(enabled)
        self.btn_reset_defaults.setEnabled(enabled)
        for row in self._sliders.values():
            row.set_enabled(enabled)
