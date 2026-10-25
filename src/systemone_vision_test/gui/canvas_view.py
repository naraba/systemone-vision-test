"""本体部分のキャンバス（ランドルト環の表示とオーバーレイの配置）。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QWidget

from systemone_vision_test import config
from systemone_vision_test.gui import style
from systemone_vision_test.gui.images import to_pixmap
from systemone_vision_test.gui.overlays import ResultOverlay, TopInfoOverlay


class CanvasView(QScrollArea):
    """ランドルト環を表示するキャンバス。

    中央のラベルに画像を表示し、その子として上辺の補助情報
    （:class:`TopInfoOverlay`）と中央の検査結果（:class:`ResultOverlay`）を重ねる。
    自分の大きさが変わったらオーバーレイの位置と幅を整える。

    既定のウィンドウサイズでは表示領域とキャンバス（:data:`config.CANVAS_SIZE`）の
    大きさが一致するので、画像は原寸（1:1）で表示される。ウィンドウを広げれば拡大、
    狭めれば縮小される（アスペクト比は維持）。スクロールバーは、出るとオーバーレイが
    表示領域より広くなって右端が切れるため常に隠す。

    Args:
        parent: 親ウィジェット。
    """

    #: 画像を表示するラベルが確保する最小の大きさ（px）
    MINIMUM_SIZE = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        #: 画像を表示するラベル
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(style.CANVAS_STYLE)
        self.image_label.setMinimumSize(self.MINIMUM_SIZE, self.MINIMUM_SIZE)
        self.setWidget(self.image_label)

        #: 上辺の補助情報（モデル・生成モード・応答時間・確率）
        self.top_overlay = TopInfoOverlay(self.image_label)
        self.top_overlay.move(0, 0)
        #: 中央の検査結果
        self.result_overlay = ResultOverlay(self.image_label)

        #: 表示中の画像。``QImage`` がこのバッファを参照するため保持しておく
        self._image: np.ndarray | None = None

    # -- 公開 API -----------------------------------------------------------

    def set_image(self, image: np.ndarray) -> None:
        """キャンバスに画像を表示する。

        表示領域に合わせて拡大・縮小する（縦横比は維持）。

        Args:
            image: 表示するグレースケール画像。
        """
        pixmap = to_pixmap(image)
        area = self.viewport().size()
        if area.width() > 0 and area.height() > 0:
            pixmap = pixmap.scaled(
                area,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._image = image
        self.image_label.setPixmap(pixmap)

    def show_blank(self) -> None:
        """キャンバスを背景色一色にする（検査の区切りに使う）。"""
        self.set_image(
            np.full(
                (config.CANVAS_SIZE, config.CANVAS_SIZE),
                config.BACKGROUND_GRAY,
                dtype=np.uint8,
            )
        )

    def show_result(self, vision: float | None) -> None:
        """中央のオーバーレイに確定した視力値を表示する。

        Args:
            vision: 確定した視力値。``None`` なら測定不可と表示する。
        """
        self.result_overlay.show_vision(vision)
        self.result_overlay.show()
        self.result_overlay.raise_()
        self._place_overlays()

    def hide_result(self) -> None:
        """中央のオーバーレイを隠す。"""
        self.result_overlay.hide()

    def displayed_pixmap(self) -> QPixmap | None:
        """表示中の画像（テスト・診断用）。

        Returns:
            表示中の ``QPixmap``。まだ何も表示していなければ ``None``。
        """
        return self.image_label.pixmap()

    # -- 内部処理 -----------------------------------------------------------

    def resizeEvent(self, event: QResizeEvent) -> None:
        """自分の大きさが変わったらオーバーレイの位置を合わせる。"""
        super().resizeEvent(event)
        self._place_overlays()

    def _place_overlays(self) -> None:
        """中央のオーバーレイを画像の中央へ、上辺のオーバーレイを幅いっぱいへ置く。"""
        pixmap = self.image_label.pixmap()
        label_rect = self.image_label.rect()

        if pixmap is not None and not pixmap.isNull():
            image_rect = pixmap.rect()
            offset_x = (label_rect.width() - image_rect.width()) // 2
            offset_y = (label_rect.height() - image_rect.height()) // 2
            self.result_overlay.move(
                offset_x + (image_rect.width() - self.result_overlay.width()) // 2,
                offset_y + (image_rect.height() - self.result_overlay.height()) // 2,
            )

        # 上辺のオーバーレイは表示領域の幅に合わせる（右端が切れないように）
        self.top_overlay.setFixedWidth(self.viewport().width())
        self.top_overlay.move(0, 0)
