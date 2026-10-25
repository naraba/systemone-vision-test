"""画像（``numpy`` ↔ ``QPixmap``）の変換と拡大・縮小のヘルパ。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap


def to_pixmap(image: np.ndarray) -> QPixmap:
    """グレースケール画像を ``QPixmap`` に変換する。

    Args:
        image: グレースケール（``uint8``）の画像。

    Returns:
        変換した画像。画素の並び（ストライド）も引き継ぐ。
    """
    height, width = image.shape[:2]
    qimage = QImage(image.data, width, height, image.strides[0], QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimage)


def fit_to_box(pixmap: QPixmap, box: QSize) -> QPixmap:
    """縦横比を維持して、指定した箱に収まる大きさへ拡大・縮小する。

    箱より大きければ縮小し、小さければ拡大する。

    Args:
        pixmap: 元の画像。
        box: 収めたい大きさ（幅・高さ）。

    Returns:
        収めた画像。``box`` が不正な大きさなら元の画像をそのまま返す。
    """
    if box.width() <= 0 or box.height() <= 0:
        return pixmap
    return pixmap.scaled(
        box,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
