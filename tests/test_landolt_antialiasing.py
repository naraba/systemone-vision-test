"""ランドルト環の輪郭（アンチエイリアス）のテスト。

輪郭が滑らかに描かれているか（ギザつきがないか）と、被覆率が幾何学的な面積に
一致するかを、画素値をサブピクセル補間して数値で検証する。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from systemone_vision_test import config
from systemone_vision_test.domain.landolt import (
    LandoltCParams,
    generate_landolt_c_image,
    outer_diameter_px,
)

CENTER = (config.CANVAS_SIZE // 2, config.CANVAS_SIZE // 2)
#: 背景と環のちょうど中間（輪郭の 0.5 等高線を拾うためのしきい値）
EDGE_THRESHOLD = (config.BACKGROUND_GRAY + config.RING_GRAY) / 2
#: 切れ目が外円上で占める角度の半幅（度）
SLIT_ANGLE_DEG = math.degrees(math.asin(config.ISO_SLIT_WIDTH_RATIO))
#: 輪郭を探すときに切れ目の方向を避ける余裕（度）
SLIT_ANGLE_MARGIN_DEG = 5.0


def sample_bilinear(image: np.ndarray, x: float, y: float) -> float:
    """サブピクセル位置の画素値を双線形補間で返す。"""
    x0, y0 = math.floor(x), math.floor(y)
    dx, dy = x - x0, y - y0
    return float(
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x0 + 1] * dx * (1 - dy)
        + image[y0 + 1, x0] * (1 - dx) * dy
        + image[y0 + 1, x0 + 1] * dx * dy
    )


def outer_edge_radius(image: np.ndarray, angle_deg: float, expected_radius: float) -> float:
    """背景から環へ入る輪郭（0.5 等高線）の半径を線形補間で求める。"""
    rad = math.radians(angle_deg)
    previous_radius: float | None = None
    previous_value: float | None = None
    for step in np.arange(2.0, -2.0, -0.02):
        radius = expected_radius + step
        value = sample_bilinear(
            image, CENTER[0] + radius * math.cos(rad), CENTER[1] + radius * math.sin(rad)
        )
        if previous_value is not None and previous_value >= EDGE_THRESHOLD > value:
            fraction = (previous_value - EDGE_THRESHOLD) / (previous_value - value)
            return previous_radius + (radius - previous_radius) * fraction
        previous_radius, previous_value = radius, value
    return float("nan")


def outline_distance(vision: float, direction: str) -> np.ndarray:
    """画素中心から環の輪郭（外円・内円・切れ目の辺）までの距離（px）を返す。"""
    outer_radius = outer_diameter_px(vision) / 2
    inner_radius = outer_radius * config.ISO_INNER_DIAMETER_RATIO
    slit_half_width = outer_radius * config.ISO_SLIT_WIDTH_RATIO

    ys, xs = np.mgrid[0 : config.CANVAS_SIZE, 0 : config.CANVAS_SIZE]
    radius = np.hypot(xs - CENTER[0], ys - CENTER[1])
    across = (xs - CENTER[0]) if direction in ("up", "down") else (ys - CENTER[1])
    return np.minimum.reduce(
        [
            np.abs(radius - outer_radius),
            np.abs(radius - inner_radius),
            np.abs(np.abs(across) - slit_half_width),
        ]
    )


def ink_area(image: np.ndarray) -> float:
    """画像から求めた環の面積（画素の被覆率の合計）。"""
    span = config.BACKGROUND_GRAY - config.RING_GRAY
    return float((config.BACKGROUND_GRAY - image.astype(np.int32)).sum()) / span


def geometric_ink_area(vision: float, direction: str, factor: int = 12) -> float:
    """超高解像度で数えた環の面積（被覆率の検算用）。

    切れ目は「中心から指定方向へ伸びる幅 2 * slit の長方形」なので、片側だけを
    くり抜く。
    """
    outer_radius = outer_diameter_px(vision) / 2
    inner_radius = outer_radius * config.ISO_INNER_DIAMETER_RATIO
    slit_half_width = outer_radius * config.ISO_SLIT_WIDTH_RATIO

    reach = int(np.ceil(outer_radius)) + 1
    offsets = (np.arange(factor) + 0.5) / factor - 0.5
    axis = (np.arange(-reach, reach + 1)[:, None] + offsets[None, :]).reshape(-1)
    ys, xs = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xs, ys)
    inside = (radius <= outer_radius) & (radius >= inner_radius)

    if direction == "up":
        along, across = -ys, xs
    elif direction == "down":
        along, across = ys, xs
    elif direction == "left":
        along, across = -xs, ys
    else:
        along, across = xs, ys
    inside &= ~((np.abs(across) <= slit_half_width) & (along >= 0.0))
    return float(inside.sum()) / (factor * factor)


# ---------------------------------------------------------------------------
# 輪郭の滑らかさ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vision", [2.0, 1.5, 1.0, 0.5, 0.1])
def test_circle_outline_is_smooth_at_any_size(vision):
    """円の輪郭が滑らかで、半径も正確（小さい環でもギザつかない）。

    輪郭を背景／環の 2 値でラスタライズしていた頃は、半径が 1px 刻みに量子化されて
    0.8〜0.95px ばらつき、平均も -0.5px ほど小さくなっていた。
    """
    image = generate_landolt_c_image(LandoltCParams(vision=vision, direction="right"))
    expected_radius = outer_diameter_px(vision) / 2

    radii = np.array(
        [
            outer_edge_radius(image, angle, expected_radius)
            for angle in range(0, 360, 5)
            if abs((angle + 180) % 360 - 180) > SLIT_ANGLE_DEG + SLIT_ANGLE_MARGIN_DEG
        ]
    )

    assert not np.isnan(radii).any()
    assert radii.max() - radii.min() < 0.4  # 輪郭のギザつき（ばらつき）
    assert abs(float(radii.mean()) - expected_radius) < 0.1  # 半径の正確さ


@pytest.mark.parametrize("direction", list(config.DIRECTIONS))
def test_anti_aliased_pixels_are_only_at_the_outline(direction):
    """アンチエイリアスの中間調は輪郭の 1px 以内にだけ現れる。"""
    vision = 0.5
    image = generate_landolt_c_image(LandoltCParams(vision=vision, direction=direction))
    intermediate = (image > config.RING_GRAY) & (image < config.BACKGROUND_GRAY)

    assert intermediate.any()  # アンチエイリアスが効いている
    assert outline_distance(vision, direction)[intermediate].max() <= 1.0


@pytest.mark.parametrize("vision", [2.0, 0.5])
def test_coverage_matches_geometric_area(vision):
    """輪郭の被覆率の合計が幾何学的な面積と一致する。"""
    direction = "right"
    image = generate_landolt_c_image(LandoltCParams(vision=vision, direction=direction))
    expected = geometric_ink_area(vision, direction)

    assert abs(ink_area(image) - expected) / expected < 0.015
