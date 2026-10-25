"""clef の応答確率を棒グラフで表示するウィジェット。"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

#: ``{方向: 確率}``
Probabilities = dict[str, float]


class ProbabilityBarGraph(QWidget):
    """clef の応答確率を棒グラフで表示する。

    左から上・下・左・右の順に、棒・ラベル・値を描く。
    clef が回答した方向の棒は外枠を太くし、本来の正解方向の棒は
    正答なら緑、誤答なら赤で表示する。

    Args:
        parent: 親ウィジェット。
    """

    #: (応答キー, 表示ラベル) を左からの並び順で定義
    LABELS: tuple[tuple[str, str], ...] = (
        ("up", "上"),
        ("down", "下"),
        ("left", "左"),
        ("right", "右"),
    )

    #: 棒の高さ（ラベル部分を除く、px）
    BAR_HEIGHT = 20
    #: 棒同士の間隔（px）
    BAR_GAP = 8
    #: 右端が切れないように確保する余白（px）
    RIGHT_MARGIN = 2

    DEFAULT_FILL = QColor(70, 130, 200)
    TRACK_FILL = QColor(238, 238, 238)
    TRACK_BORDER = QColor(170, 170, 170)
    CORRECT_COLOR = QColor(30, 160, 70)
    WRONG_COLOR = QColor(205, 45, 45)
    ANSWER_BORDER_COLOR = QColor(45, 45, 45)
    ANSWER_BORDER_WIDTH = 3
    TEXT_COLOR = QColor(30, 30, 30)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._probabilities: Probabilities = {key: 0.0 for key, _ in self.LABELS}
        self._answer: str | None = None
        self._correct_direction: str | None = None
        self._is_correct: bool | None = None
        self.setMinimumHeight(self.BAR_HEIGHT + 20)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # -- 公開 API -----------------------------------------------------------

    @property
    def probabilities(self) -> Probabilities:
        """現在表示している確率。"""
        return dict(self._probabilities)

    @property
    def answer(self) -> str | None:
        """強調している回答方向。"""
        return self._answer

    @property
    def correct_direction(self) -> str | None:
        """色分けしている正解方向。"""
        return self._correct_direction

    @property
    def is_correct(self) -> bool | None:
        """判定結果（未設定なら ``None``）。"""
        return self._is_correct

    def clear(self) -> None:
        """棒グラフを空にして、判定結果の注釈も消す。"""
        self.set_probabilities({})

    def set_probabilities(self, probabilities: Probabilities) -> None:
        """確率だけを設定する（判定結果の注釈は消す）。

        Args:
            probabilities: 方向ごとの確率（0.0〜1.0）。
        """
        self._probabilities = self._normalized(probabilities)
        self._answer = None
        self._correct_direction = None
        self._is_correct = None
        self.update()

    def set_result(
        self,
        probabilities: Probabilities,
        answer: str,
        correct_direction: str,
        is_correct: bool,
    ) -> None:
        """確率と判定結果を設定する。

        Args:
            probabilities: 方向ごとの確率（0.0〜1.0）。
            answer: clef が回答した方向（外枠を太くしラベルを太字にする）。
            correct_direction: 本来の正解方向（正答なら緑、誤答なら赤）。
            is_correct: 判定が正答だったか。
        """
        self._probabilities = self._normalized(probabilities)
        self._answer = answer
        self._correct_direction = correct_direction
        self._is_correct = is_correct
        self.update()

    # -- 内部処理 -----------------------------------------------------------

    @staticmethod
    def _normalized(probabilities: Probabilities) -> Probabilities:
        """表示用に 0.0〜1.0 へ丸めた確率を作る。"""
        return {
            key: max(0.0, min(1.0, float(probabilities.get(key, 0.0) or 0.0)))
            for key, _ in ProbabilityBarGraph.LABELS
        }

    def bar_width(self) -> float:
        """棒 1 本の幅（px）。"""
        draw_width = self.width() - self.RIGHT_MARGIN
        return (draw_width - self.BAR_GAP * (len(self.LABELS) - 1)) / len(self.LABELS)

    def paintEvent(self, event) -> None:
        """棒グラフを描画する。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        normal_font = QFont(self.font())
        normal_font.setPointSize(8)
        bold_font = QFont(normal_font)
        bold_font.setBold(True)
        painter.setFont(normal_font)
        text_height = painter.fontMetrics().height()

        width = self.bar_width()
        if width <= 1:
            painter.end()
            return

        top = 1.0
        height = float(self.BAR_HEIGHT)
        for index, (key, label) in enumerate(self.LABELS):
            self._paint_bar(
                painter,
                key,
                label,
                index,
                width,
                top,
                height,
                text_height,
                normal_font,
                bold_font,
            )

        painter.end()

    def _paint_bar(
        self,
        painter: QPainter,
        key: str,
        label: str,
        index: int,
        width: float,
        top: float,
        height: float,
        text_height: int,
        normal_font: QFont,
        bold_font: QFont,
    ) -> None:
        """1 本分の棒・外枠・ラベルを描画する。"""
        x = index * (width + self.BAR_GAP)
        value = self._probabilities.get(key, 0.0)
        is_answer = self._answer == key

        if self._correct_direction == key:
            fill_color = self.CORRECT_COLOR if self._is_correct else self.WRONG_COLOR
            text_color = fill_color
        else:
            fill_color = self.DEFAULT_FILL
            text_color = self.TEXT_COLOR

        # 枠（トラック）
        painter.setPen(QPen(self.TRACK_BORDER))
        painter.setBrush(self.TRACK_FILL)
        painter.drawRect(QRectF(x, top, width, height))

        # 値に応じた塗り
        fill_width = width * value
        if fill_width > 0.0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            painter.drawRect(QRectF(x, top, fill_width, height))

        # 回答した棒は外枠を太くする
        if is_answer:
            inset = self.ANSWER_BORDER_WIDTH / 2.0
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(self.ANSWER_BORDER_COLOR, self.ANSWER_BORDER_WIDTH))
            painter.drawRect(
                QRectF(
                    x + inset,
                    top + inset,
                    width - self.ANSWER_BORDER_WIDTH,
                    height - self.ANSWER_BORDER_WIDTH,
                )
            )

        # ラベルと値（回答した棒は太字）
        painter.setFont(bold_font if is_answer else normal_font)
        painter.setPen(text_color)
        painter.drawText(
            QRectF(x, top + height + 2.0, width, float(text_height)),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
            f"{label} {value * 100:.1f}%",
        )
