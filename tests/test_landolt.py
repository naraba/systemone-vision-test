"""ランドルト環画像生成（domain.landolt）のテスト。"""

from __future__ import annotations

import base64
import math

import numpy as np
import pytest

from systemone_vision_test import config
from systemone_vision_test.domain import landolt
from systemone_vision_test.domain.landolt import (
    GenerationMode,
    LandoltCParams,
    default_parameters,
    encode_png_base64,
    generate_landolt_c_image,
    generate_sample_image,
    outer_diameter_px,
    parameter_specs,
)

CENTER = (config.CANVAS_SIZE // 2, config.CANVAS_SIZE // 2)

#: 画面座標（y は下向き）での方向ごとの角度（度）
DIRECTION_ANGLES = {"right": 0, "down": 90, "left": 180, "up": 270}


def ring_radii(vision: float) -> tuple[int, int]:
    """``(内半径, 外半径)`` を返す。"""
    outer = outer_diameter_px(vision)
    outer_radius = outer // 2
    inner_radius = round(outer * config.ISO_INNER_DIAMETER_RATIO) // 2
    return inner_radius, outer_radius


def pixel_at(image: np.ndarray, radius: float, angle_deg: float) -> int:
    """中心から指定半径・角度の位置のピクセル値を返す。"""
    rad = math.radians(angle_deg)
    x = int(CENTER[0] + radius * math.cos(rad))
    y = int(CENTER[1] + radius * math.sin(rad))
    return int(image[y, x])


# ---------------------------------------------------------------------------
# キャンバスと ISO 8596 の形状
# ---------------------------------------------------------------------------


def test_canvas_is_fixed_size_white_background():
    """キャンバスは固定サイズで、外側は白である。"""
    image = generate_landolt_c_image(LandoltCParams(vision=1.0, direction="right"))

    assert image.shape == (config.CANVAS_SIZE, config.CANVAS_SIZE)
    assert image.dtype == np.uint8
    assert image[0, 0] == config.BACKGROUND_GRAY


@pytest.mark.parametrize("vision", [2.0, 1.0, 0.1])
def test_ring_is_centered_on_canvas(vision):
    """環はどの視力でもキャンバスの中央に置かれる。"""
    image = generate_landolt_c_image(LandoltCParams(vision=vision))
    inner_radius, _ = ring_radii(vision)

    assert image[CENTER[1], CENTER[0]] == config.BACKGROUND_GRAY
    # 内側のくり抜きの中心付近は白（小さい環でも破綻しない半径で見る）
    assert pixel_at(image, max(1.0, inner_radius / 2), 45) == config.BACKGROUND_GRAY


@pytest.mark.parametrize("direction", list(config.DIRECTIONS))
def test_ring_shape_has_single_slit(direction):
    """環状で、切れ目が指定方向に 1 箇所だけ開いている。"""
    vision = 0.1
    image = generate_landolt_c_image(LandoltCParams(vision=vision, direction=direction))
    inner_radius, outer_radius = ring_radii(vision)
    mid_radius = (inner_radius + outer_radius) / 2
    angle = DIRECTION_ANGLES[direction]

    # 内側はくり抜かれている（白）
    assert pixel_at(image, inner_radius - 20, angle) == config.BACKGROUND_GRAY
    # 切れ目の方向は白（開いている）
    assert pixel_at(image, mid_radius, angle) == config.BACKGROUND_GRAY
    # 直角方向と反対方向は黒（環がつながっている）
    assert pixel_at(image, mid_radius, angle + 90) == config.RING_GRAY
    assert pixel_at(image, mid_radius, angle - 90) == config.RING_GRAY
    assert pixel_at(image, mid_radius, angle + 180) == config.RING_GRAY


def test_slit_width_is_one_fifth_of_outer_diameter():
    """切れ目の幅が ISO 8596 の比率（外径の 20%）になっている。"""
    vision = 0.1
    image = generate_landolt_c_image(LandoltCParams(vision=vision, direction="right"))
    inner_radius, outer_radius = ring_radii(vision)
    outer_size = outer_diameter_px(vision)
    half_slit = round(outer_size * config.ISO_SLIT_WIDTH_RATIO) // 2

    # 環の上下が確実に残る x 位置で、切れ目の境界の内外を見る
    x_probe = CENTER[0] + (inner_radius + outer_radius) // 2
    margin = 4
    top_edge = CENTER[1] - half_slit
    bottom_edge = CENTER[1] + half_slit

    assert image[top_edge - margin, x_probe] == config.RING_GRAY
    assert image[top_edge + margin, x_probe] == config.BACKGROUND_GRAY
    assert image[bottom_edge - margin, x_probe] == config.BACKGROUND_GRAY
    assert image[bottom_edge + margin, x_probe] == config.RING_GRAY


@pytest.mark.parametrize(
    ("vision", "expected"),
    [(2.0, 26), (1.0, 53), (0.5, 105), (0.1, 527)],
)
def test_outer_diameter_follows_scale(vision, expected):
    """外径 = 52.7 / 視力（四捨五入）になる。"""
    assert outer_diameter_px(vision) == expected


# ---------------------------------------------------------------------------
# 生成モード
# ---------------------------------------------------------------------------


def test_clear_mode_keeps_pure_black_and_white_away_from_edges():
    """清明モードは輪郭以外をぼかさない（環の内部は純黒、背景は純白）。"""
    vision = 0.5
    direction = "up"
    image = generate_landolt_c_image(
        LandoltCParams(vision=vision, direction=direction, mode=GenerationMode.CLEAR)
    )

    black = int((image == config.RING_GRAY).sum())
    white = int((image == config.BACKGROUND_GRAY).sum())

    intermediate = int(((image > config.RING_GRAY) & (image < config.BACKGROUND_GRAY)).sum())
    assert black >= 5 * intermediate  # 環の内部は純黒（輪郭だけが中間調）
    assert white >= 0.7 * image.size  # 背景は純白


@pytest.mark.parametrize(
    "mode",
    [GenerationMode.MYOPIA, GenerationMode.CATARACT, GenerationMode.ASTIGMATISM],
)
def test_non_clear_modes_blur_the_image(mode):
    """清明以外のモードは中間調が生じる（ぼけている）。"""
    vision = 1.0
    clear = generate_landolt_c_image(
        LandoltCParams(vision=vision, direction="up", mode=GenerationMode.CLEAR)
    )
    blurred = generate_landolt_c_image(LandoltCParams(vision=vision, direction="up", mode=mode))

    assert not np.array_equal(clear, blurred)
    assert np.count_nonzero((blurred > 0) & (blurred < 255)) > 0


def test_cataract_mode_lowers_contrast():
    """混濁モードはコントラストが下がる（環が灰色寄りになる）。"""
    vision = 0.5
    image = generate_landolt_c_image(
        LandoltCParams(
            vision=vision,
            direction="up",
            mode=GenerationMode.CATARACT,
            box_ksize=41,
            contrast=0.15,
        )
    )
    inner_radius, outer_radius = ring_radii(vision)
    ring_value = pixel_at(image, (inner_radius + outer_radius) / 2, 180)

    assert ring_value > 60  # 清明なら 0


def test_astigmatism_direction_changes_result():
    """乱視はぼかす方向によって結果が変わる。"""
    base = {"vision": 1.0, "direction": "right", "mode": GenerationMode.ASTIGMATISM}
    horizontal = generate_landolt_c_image(LandoltCParams(**base, astig_angle_deg=0.0))
    vertical = generate_landolt_c_image(LandoltCParams(**base, astig_angle_deg=90.0))

    assert not np.array_equal(horizontal, vertical)


def test_filter_registry_covers_all_modes():
    """清明以外のすべてのモードにフィルタが定義されている。"""
    assert set(landolt._FILTERS) == set(GenerationMode) - {GenerationMode.CLEAR}


# ---------------------------------------------------------------------------
# パラメータとその仕様
# ---------------------------------------------------------------------------


def test_parameter_specs_match_dataclass_defaults():
    """モード既定値（仕様）が ``LandoltCParams`` の既定値と一致する。"""
    defaults = LandoltCParams(vision=1.0)
    for mode in GenerationMode:
        for spec in parameter_specs(mode):
            assert getattr(defaults, spec.key) == spec.to_value(spec.default)
        assert default_parameters(mode) == {
            spec.key: spec.to_value(spec.default) for spec in parameter_specs(mode)
        }


def test_slider_value_round_trip():
    """スライダー値と実値の変換が往復する。"""
    for mode in GenerationMode:
        for spec in parameter_specs(mode):
            assert spec.to_slider(spec.to_value(spec.default)) == spec.default


def test_with_parameters_overrides_and_rejects_unknown():
    """パラメータの上書きと、未知キーの拒否。"""
    spec = LandoltCParams(vision=1.0, mode=GenerationMode.MYOPIA)
    updated = spec.with_parameters({"gaussian_ksize": 5})

    assert updated.gaussian_ksize == 5
    assert spec.gaussian_ksize == config.MYOPIA_GAUSSIAN_KSIZE_DEFAULT  # 元は不変
    with pytest.raises(KeyError):
        spec.with_parameters({"unknown_key": 1})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vision": 0.0},
        {"vision": -1.0},
        {"direction": "diagonal"},
        {"gaussian_ksize": 4},
        {"box_ksize": 0},
        {"contrast": 0.0},
        {"contrast": 1.5},
        {"astig_strength": 0.0},
    ],
)
def test_invalid_parameters_raise(kwargs):
    """許容範囲外のパラメータは ``ValueError`` になる。"""
    with pytest.raises(ValueError):
        LandoltCParams(**{"vision": 1.0, **kwargs})


# ---------------------------------------------------------------------------
# サンプル画像とエンコード
# ---------------------------------------------------------------------------


def test_sample_image_uses_large_ring():
    """サンプル画像は表示側で縮小する前提の大きな環になる。"""
    image = generate_sample_image(GenerationMode.CLEAR, direction="right")
    inner_radius, outer_radius = ring_radii(config.SAMPLE_VISION)
    mid_radius = (inner_radius + outer_radius) / 2

    assert image.shape == (config.CANVAS_SIZE, config.CANVAS_SIZE)
    assert pixel_at(image, mid_radius, 0) == config.BACKGROUND_GRAY  # 右が開いている
    assert pixel_at(image, mid_radius, 180) == config.RING_GRAY


def test_sample_image_accepts_parameter_overrides():
    """サンプル画像にもモード固有パラメータを渡せる。"""
    sharp = generate_sample_image(GenerationMode.MYOPIA, gaussian_ksize=3)
    blurred = generate_sample_image(GenerationMode.MYOPIA, gaussian_ksize=61)

    assert not np.array_equal(sharp, blurred)


@pytest.mark.parametrize("mode", list(GenerationMode))
def test_sample_image_fits_in_canvas(mode):
    """サンプル画像の環がキャンバスの縁に接しない（はみ出さない）。"""
    image = generate_sample_image(mode, direction="right", **default_parameters(mode))

    background = int(image[0, 0])
    difference = np.abs(border_pixels(image).astype(np.int16) - background)
    assert difference.max() <= BORDER_TOLERANCE


def test_sample_vision_is_the_largest_ring_that_fits():
    """サンプルは「キャンバスに収まる限界の大きさの環」を使う。"""
    outer_radius = outer_diameter_px(config.SAMPLE_VISION) / 2

    # 環がキャンバスに収まっている
    assert outer_radius < config.CANVAS_SIZE / 2
    # これ以上大きい環（1 段階低い視力）は収まらない
    lower_visions = [vision for vision in config.VISION_LEVELS if vision < config.SAMPLE_VISION]
    if lower_visions:
        too_big = outer_diameter_px(max(lower_visions)) / 2
        assert too_big >= config.CANVAS_SIZE / 2


def test_encode_png_base64_has_no_data_uri():
    """Base64 はデータ URI なしの生文字列で、キャンバス全体の PNG になる。"""
    image = generate_landolt_c_image(LandoltCParams(vision=1.0, direction="left"))
    encoded = encode_png_base64(image)

    assert not encoded.startswith("data:")
    raw = base64.b64decode(encoded)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    assert (width, height) == (config.CANVAS_SIZE, config.CANVAS_SIZE)


def test_mode_enum_str_is_display_name():
    """``GenerationMode`` の文字列化は表示名になる。"""
    assert str(GenerationMode.CATARACT) == "混濁"
    assert GenerationMode("近視") is GenerationMode.MYOPIA


# ---------------------------------------------------------------------------
# キャンバスサイズに対する余白（縁に接しないこと）
# ---------------------------------------------------------------------------

#: 「背景と同じ」とみなす許容差（ぼけの裾を考慮したわずかな差）
BORDER_TOLERANCE = 4


def border_pixels(image: np.ndarray) -> np.ndarray:
    """画像の外周 1 ピクセル分をまとめて返す。"""
    return np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])


@pytest.mark.parametrize("mode", list(GenerationMode))
def test_largest_ring_never_touches_canvas_border(mode):
    """最大の環（視力 0.1）でも、ぼけを含めてキャンバスの縁に接しない。"""
    spec = LandoltCParams(
        vision=min(config.VISION_LEVELS), direction="right", mode=mode
    ).with_parameters(default_parameters(mode))
    image = generate_landolt_c_image(spec)

    background = int(image[0, 0])
    difference = np.abs(border_pixels(image).astype(np.int16) - background)
    assert difference.max() <= BORDER_TOLERANCE


@pytest.mark.parametrize(
    "mode",
    [GenerationMode.MYOPIA, GenerationMode.CATARACT, GenerationMode.ASTIGMATISM],
)
def test_largest_ring_fits_even_at_parameter_maximums(mode):
    """パラメータを最大にしても、最大の環がキャンバスに収まる。"""
    maximums = {spec.key: spec.to_value(spec.maximum) for spec in parameter_specs(mode)}
    spec = LandoltCParams(
        vision=min(config.VISION_LEVELS), direction="right", mode=mode
    ).with_parameters(maximums)
    image = generate_landolt_c_image(spec)

    background = int(image[0, 0])
    difference = np.abs(border_pixels(image).astype(np.int16) - background)
    assert difference.max() <= BORDER_TOLERANCE
