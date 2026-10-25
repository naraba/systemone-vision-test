"""GUI の表示部品（gui.images / gui.overlays / gui.probability_bar / gui.style）のテスト。

描画結果をピクセル単位で検証する。PNG は ``artifacts/tests`` に保存するので
見た目も目視確認できる。
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QFont, QFontMetrics, QImage, QPixmap
from PySide6.QtWidgets import QFrame

from systemone_vision_test import config
from systemone_vision_test.domain.landolt import GenerationMode
from systemone_vision_test.gui import style
from systemone_vision_test.gui.images import fit_to_box, to_pixmap
from systemone_vision_test.gui.overlays import ResultOverlay, TopInfoOverlay
from systemone_vision_test.gui.probability_bar import ProbabilityBarGraph

#: 検証に使う確率（右だけ高い）
PROBABILITIES = {"up": 0.02, "down": 0.01, "left": 0.03, "right": 0.94}
#: 塗り幅の検証に使う確率
UNIFORM_KEYS_PROBABILITIES = {"up": 0.0, "down": 0.5, "left": 0.25, "right": 1.0}


def is_blue(rgb: tuple[int, int, int]) -> bool:
    """既定の棒の色（青）かどうか。"""
    red, green, blue = rgb
    return red < 130 and 90 < green < 170 and blue > 170


def is_greenish(rgb: tuple[int, int, int]) -> bool:
    """緑系かどうか。"""
    red, green, blue = rgb
    return green > 110 and green > red + 40 and green > blue + 40


def is_reddish(rgb: tuple[int, int, int]) -> bool:
    """赤系かどうか。"""
    red, green, blue = rgb
    return red > 150 and red > green + 60 and red > blue + 60


def render(widget) -> QImage:
    """ウィジェットを描画して ``QImage`` を返す。"""
    return widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)


def pixel(image: QImage, x: int, y: int) -> tuple[int, int, int]:
    """指定位置のピクセルを ``(r, g, b)`` で返す。"""
    color = image.pixelColor(x, y)
    return (color.red(), color.green(), color.blue())


def bar_range(graph: ProbabilityBarGraph, key: str) -> tuple[int, int]:
    """指定した棒の x 範囲を返す。"""
    keys = [name for name, _ in ProbabilityBarGraph.LABELS]
    index = keys.index(key)
    width = graph.bar_width()
    start = int(index * (width + ProbabilityBarGraph.BAR_GAP))
    return start, int(start + width)


def fill_color(graph: ProbabilityBarGraph, key: str) -> tuple[int, int, int]:
    """指定した棒の塗り部分の色を返す（灰色のトラックは除く）。"""
    image = render(graph)
    start, end = bar_range(graph, key)
    y = ProbabilityBarGraph.BAR_HEIGHT // 2
    for x in range(start, min(end, image.width())):
        rgb = pixel(image, x, y)
        if max(rgb) - min(rgb) >= 30:  # 彩度がある＝塗り
            return rgb
    return (0, 0, 0)


def fill_ratio(graph: ProbabilityBarGraph, key: str) -> float:
    """指定した棒の充填率（塗りの幅 ÷ 棒の幅）を返す。"""
    image = render(graph)
    start, end = bar_range(graph, key)
    y = ProbabilityBarGraph.BAR_HEIGHT // 2
    filled = total = 0
    for x in range(start + 1, min(end, image.width())):
        total += 1
        filled += int(is_blue(pixel(image, x, y)))
    return filled / total if total else 0.0


# ---------------------------------------------------------------------------
# 棒グラフ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [("up", 0.0), ("down", 0.5), ("left", 0.25), ("right", 1.0)],
)
def test_bar_fill_is_proportional_to_value(key, expected, qapp):
    """棒の塗り幅が確率に比例する。"""
    graph = ProbabilityBarGraph()
    graph.resize(800, 40)
    graph.set_probabilities(UNIFORM_KEYS_PROBABILITIES)

    assert fill_ratio(graph, key) == pytest.approx(expected, abs=0.08)


def test_correct_bar_is_green(qapp):
    """正答時は正解バーが緑になる。"""
    graph = ProbabilityBarGraph()
    graph.resize(800, 40)
    graph.set_result(PROBABILITIES, answer="right", correct_direction="right", is_correct=True)

    assert is_greenish(fill_color(graph, "right"))


def test_wrong_bar_is_red_and_answer_bar_keeps_default_color(qapp):
    """誤答時は正解バーが赤、回答バーは既定色のまま。"""
    graph = ProbabilityBarGraph()
    graph.resize(800, 40)
    graph.set_result(PROBABILITIES, answer="right", correct_direction="down", is_correct=False)

    assert is_reddish(fill_color(graph, "down"))
    assert is_blue(fill_color(graph, "right"))


def test_answer_bar_has_thick_border(qapp):
    """回答した棒は外枠が太くなる。"""
    graph = ProbabilityBarGraph()
    graph.resize(800, 40)
    graph.set_result(PROBABILITIES, answer="right", correct_direction="down", is_correct=False)

    image = render(graph)
    start, end = bar_range(graph, "right")
    dark = sum(
        1
        for x in range(start + 2, end - 2)
        if pixel(image, x, 1)[0] < 80 and pixel(image, x, 1)[1] < 80
    )
    assert dark > int(graph.bar_width() * 0.8)


def test_rightmost_bar_is_not_clipped(qapp):
    """端数が出る幅でも右端の棒が切れない。"""
    graph = ProbabilityBarGraph()
    graph.resize(801, 40)
    graph.set_probabilities({key: 0.0 for key, _ in ProbabilityBarGraph.LABELS} | {"right": 1.0})

    image = render(graph)
    y = ProbabilityBarGraph.BAR_HEIGHT // 2
    last_filled = max(x for x in range(image.width()) if is_blue(pixel(image, x, y)))
    assert last_filled >= image.width() - ProbabilityBarGraph.RIGHT_MARGIN - 2


def test_clear_resets_annotations(qapp):
    """``clear`` で注釈も確率も消える。"""
    graph = ProbabilityBarGraph()
    graph.set_result(PROBABILITIES, answer="right", correct_direction="right", is_correct=True)
    graph.clear()

    assert graph.answer is None
    assert graph.correct_direction is None
    assert graph.is_correct is None
    assert set(graph.probabilities.values()) == {0.0}


# ---------------------------------------------------------------------------
# 上辺のオーバーレイ
# ---------------------------------------------------------------------------


def test_top_overlay_texts(qapp):
    """1 行目にモデル・モード・応答時間が並ぶ。"""
    overlay = TopInfoOverlay()
    overlay.set_model("clef-flash")
    overlay.set_mode(GenerationMode.MYOPIA)
    overlay.set_response_time_ms(None)
    assert overlay.lbl_response_time.text() == "API応答時間: 計測中..."

    overlay.set_response_time_ms(312.4)
    assert overlay.lbl_model.text() == "モデル: clef-flash"
    assert overlay.lbl_mode.text() == "生成モード: 近視"
    assert overlay.lbl_response_time.text() == "API応答時間: 312 ms"


def test_top_overlay_model_is_left_of_mode(qapp):
    """モデル表示は生成モードの左隣に置かれる。"""
    overlay = TopInfoOverlay()
    overlay.set_model("clef")
    overlay.set_mode(GenerationMode.CLEAR)
    overlay.setFixedWidth(600)
    overlay.show()
    qapp.processEvents()

    assert overlay.lbl_model.x() < overlay.lbl_mode.x()
    assert overlay.lbl_model.y() == overlay.lbl_mode.y()


def test_top_overlay_uses_id_selector_and_children_have_no_border(qapp):
    """枠は ID セレクタで指定し、子のラベルに枠が波及しない。"""
    overlay = TopInfoOverlay()

    assert overlay.styleSheet().strip().startswith(f"#{style.TOP_OVERLAY_OBJECT_NAME}")
    assert overlay.frameShape() is QFrame.Shape.NoFrame
    for label in (overlay.lbl_model, overlay.lbl_mode, overlay.lbl_response_time):
        assert "border" not in label.styleSheet()


def test_top_overlay_graph_fills_content_width(qapp):
    """棒グラフの幅がオーバーレイの内側の幅にほぼ一致する。"""
    overlay = TopInfoOverlay()
    overlay.setFixedWidth(500)
    overlay.show()
    qapp.processEvents()

    content_width = overlay.width() - 24  # 左右のマージン 12px
    assert overlay.graph.width() == pytest.approx(content_width, abs=2)


# ---------------------------------------------------------------------------
# 中央のオーバーレイ（検査結果）
# ---------------------------------------------------------------------------


def test_result_overlay_has_fixed_size(qapp):
    """大きさは内容に依存せず固定。"""
    overlay = ResultOverlay()

    assert overlay.width() == config.RESULT_OVERLAY_WIDTH
    assert overlay.height() == config.RESULT_OVERLAY_HEIGHT
    overlay.resize(999, 999)  # setFixedSize なので変わらない
    assert overlay.width() == config.RESULT_OVERLAY_WIDTH


def test_result_overlay_has_single_outer_frame(qapp):
    """枠は外側のひとつだけ（二重枠にならない）。"""
    overlay = ResultOverlay()
    overlay.show_text("視力：1.0")
    overlay.show()
    qapp.processEvents()

    image = render(overlay)
    middle_row = image.height() // 2

    def is_border(rgb: tuple[int, int, int]) -> bool:
        return all(abs(channel - 153) < 25 for channel in rgb)

    # 中央行を横断したときの枠線の本数（左右で 2 本＝単一の枠）
    runs = 0
    previous = False
    for x in range(image.width()):
        current = is_border(pixel(image, x, middle_row))
        runs += int(current and not previous)
        previous = current

    assert runs == 2
    assert overlay.styleSheet().strip().startswith(f"#{style.RESULT_OVERLAY_OBJECT_NAME}")
    assert overlay.frameShape() is QFrame.Shape.NoFrame
    assert "border" not in overlay.label.styleSheet()


@pytest.mark.parametrize(("vision", "expected"), [(1.0, "視力：1.0"), (None, "視力：測定不可")])
def test_result_overlay_text_has_no_brackets_and_fits(vision, expected, qapp):
    """角カッコなしで、枠に沿う大きさ（枠内に収まる最大サイズ）で表示する。"""
    overlay = ResultOverlay()
    overlay.show_vision(vision)
    overlay.show()  # レイアウトを確定させてから測る
    qapp.processEvents()

    assert overlay.label.text() == expected
    assert "[" not in overlay.label.text()
    assert "]" not in overlay.label.text()

    size = overlay.fit_font_size(expected)
    font = QFont(overlay.label.font())
    font.setBold(True)
    font.setPixelSize(size)
    bounds = QFontMetrics(font).boundingRect(expected)

    # 固定サイズではなく、収まる範囲で十分大きい文字になる
    assert size >= 30
    assert bounds.width() <= overlay.width() - style.RESULT_TEXT_PADDING
    assert bounds.height() <= overlay.height() - style.RESULT_TEXT_PADDING
    assert overlay.label.width() <= overlay.width()
    assert overlay.label.height() <= overlay.height()


def test_result_overlay_saves_renders(artifacts_dir, qapp):
    """目視確認用に描画結果を保存する。"""
    overlay = ResultOverlay()
    for vision, name in ((1.0, "result_ok.png"), (None, "result_na.png")):
        overlay.show_vision(vision)
        overlay.grab().save(str(artifacts_dir / name))
        assert (artifacts_dir / name).exists()


# ---------------------------------------------------------------------------
# 表示エリアへのフィット（拡大・縮小）
# ---------------------------------------------------------------------------


def test_fit_to_box_scales_down(qapp):
    """箱より大きい画像は縮小される。"""
    pixmap = QPixmap(640, 640)
    fitted = fit_to_box(pixmap, QSize(320, 320))

    assert (fitted.width(), fitted.height()) == (320, 320)


def test_fit_to_box_scales_up(qapp):
    """箱より小さい画像は拡大される。"""
    pixmap = QPixmap(640, 640)
    fitted = fit_to_box(pixmap, QSize(1280, 900))

    assert (fitted.width(), fitted.height()) == (900, 900)


def test_fit_to_box_keeps_aspect_ratio(qapp):
    """縦横比を維持して箱に収める。"""
    pixmap = QPixmap(640, 640)
    fitted = fit_to_box(pixmap, QSize(300, 200))

    assert (fitted.width(), fitted.height()) == (200, 200)


def test_fit_to_box_returns_original_for_invalid_box(qapp):
    """箱の大きさが不正なら元の画像をそのまま返す。"""
    pixmap = QPixmap(640, 640)

    assert fit_to_box(pixmap, QSize(0, 100)) is pixmap
    assert fit_to_box(pixmap, QSize(100, -1)) is pixmap


def test_to_pixmap_keeps_size_and_gray_values(qapp):
    """グレースケール画像を、同じ大きさで同じ階調の ``QPixmap`` に変換する。"""
    image = np.full((16, 8), config.RING_GRAY, dtype=np.uint8)
    image[:, 4:] = config.BACKGROUND_GRAY

    pixmap = to_pixmap(image)
    rendered = pixmap.toImage()

    assert (pixmap.width(), pixmap.height()) == (8, 16)
    assert rendered.pixelColor(1, 1).red() == config.RING_GRAY
    assert rendered.pixelColor(6, 1).red() == config.BACKGROUND_GRAY
