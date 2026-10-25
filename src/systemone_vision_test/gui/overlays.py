"""本体部分に重ねるオーバーレイ（補助情報と検査結果）。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from systemone_vision_test import config
from systemone_vision_test.domain.landolt import GenerationMode
from systemone_vision_test.gui import style
from systemone_vision_test.gui.probability_bar import Probabilities, ProbabilityBarGraph


class TopInfoOverlay(QFrame):
    """本体部分の上辺に重ねる補助情報のオーバーレイ。

    1 行目に使用モデル・生成モード・API 応答時間、2 行目に応答確率の棒グラフを表示する。
    枠は ID セレクタで指定し、子のラベルに枠が波及しないようにしている。

    Args:
        parent: 親ウィジェット。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setObjectName(style.TOP_OVERLAY_OBJECT_NAME)
        self.setStyleSheet(style.TOP_OVERLAY_STYLE)
        self.setFixedHeight(config.TOP_OVERLAY_HEIGHT)
        self.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 8)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(16)

        self.lbl_model = QLabel("")
        self.lbl_model.setStyleSheet(style.MODEL_LABEL_STYLE)
        header.addWidget(self.lbl_model)

        self.lbl_mode = QLabel("")
        self.lbl_mode.setStyleSheet(style.BOLD_STYLE)
        header.addWidget(self.lbl_mode)

        self.lbl_response_time = QLabel("")
        self.lbl_response_time.setStyleSheet(style.MUTED_STYLE)
        header.addWidget(self.lbl_response_time)

        header.addStretch()
        layout.addLayout(header)

        self.graph = ProbabilityBarGraph()
        layout.addWidget(self.graph)

    # -- 公開 API -----------------------------------------------------------

    def set_model(self, model: str) -> None:
        """使用モデルの表示を更新する。"""
        self.lbl_model.setText(f"モデル: {model}")

    def set_mode(self, mode: GenerationMode) -> None:
        """生成モードの表示を更新する。"""
        self.lbl_mode.setText(f"生成モード: {mode.value}")

    def set_response_time_ms(self, elapsed_ms: float | None) -> None:
        """API 応答時間の表示を更新する。

        Args:
            elapsed_ms: 応答時間（ミリ秒）。``None`` なら「計測中」と表示する。
        """
        if elapsed_ms is None:
            self.lbl_response_time.setText("API応答時間: 計測中...")
        else:
            self.lbl_response_time.setText(f"API応答時間: {elapsed_ms:.0f} ms")

    def set_result(
        self,
        probabilities: Probabilities,
        answer: str,
        correct_direction: str,
        is_correct: bool,
    ) -> None:
        """棒グラフに判定結果を反映する。"""
        self.graph.set_result(probabilities, answer, correct_direction, is_correct)

    def clear_result(self) -> None:
        """棒グラフを空にする（次の判定待ちの状態）。"""
        self.graph.clear()


class ResultOverlay(QFrame):
    """本体部分の中央に重ねる検査結果のオーバーレイ。

    枠は外側のひとつだけで、大きさは内容に依存せず固定。
    文字列は枠内に収まる最大の大きさで表示する。

    Args:
        parent: 親ウィジェット。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setObjectName(style.RESULT_OVERLAY_OBJECT_NAME)
        self.setStyleSheet(style.RESULT_OVERLAY_STYLE)
        self.setFixedSize(config.RESULT_OVERLAY_WIDTH, config.RESULT_OVERLAY_HEIGHT)
        self.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

    # -- 公開 API -----------------------------------------------------------

    def show_text(self, text: str) -> None:
        """文字列を枠に沿う最大サイズで表示する。"""
        self.label.setText(text)
        size = self.fit_font_size(text)
        self.label.setStyleSheet(style.RESULT_TEXT_STYLE_TEMPLATE.format(size=size))

    def show_vision(self, vision: float | None) -> None:
        """確定した視力値を表示する。``None`` なら測定不可と表示する。"""
        if vision is None:
            self.show_text("視力：測定不可")
        else:
            self.show_text(f"視力：{vision}")

    def fit_font_size(self, text: str) -> int:
        """固定した枠に収まる最大のフォントサイズ（px）を求める。

        Args:
            text: 表示する文字列。

        Returns:
            枠の幅・高さの両方に収まる最大のフォントサイズ。
        """
        max_width = max(1, self.width() - style.RESULT_TEXT_PADDING)
        max_height = max(1, self.height() - style.RESULT_TEXT_PADDING)
        font = QFont(self.label.font())
        font.setBold(True)

        best = 10
        for size in range(10, 201):
            font.setPixelSize(size)
            bounds = QFontMetrics(font).boundingRect(text)
            if bounds.width() > max_width or bounds.height() > max_height:
                break
            best = size
        return best
