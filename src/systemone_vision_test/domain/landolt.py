"""ランドルト環画像の生成（ISO 8596 準拠）。

清明・近視・混濁・乱視の 4 つの生成モードをサポートする。GUI や通信には依存しない。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Literal

import cv2
import numpy as np

from systemone_vision_test import config

#: 切れ目の開く方向
Direction = Literal["up", "down", "left", "right"]


class GenerationMode(str, Enum):
    """画像の生成モード（見え方のシミュレーション種別）。"""

    CLEAR = "清明"
    MYOPIA = "近視"
    CATARACT = "混濁"
    ASTIGMATISM = "乱視"

    def __str__(self) -> str:
        """文字列化では日本語の表示名を返す。"""
        return self.value


@dataclass(frozen=True)
class ParameterSpec:
    """生成モードのパラメータ 1 つ分の仕様（スライダー表示に必要な情報）。

    Attributes:
        key: :class:`LandoltCParams` のフィールド名。
        label: スライダーに表示するラベル。
        minimum: スライダーの最小値（整数）。
        maximum: スライダーの最大値（整数）。
        step: スライダーの刻み。
        default: 既定のスライダー値（整数）。
        scale: 実値 = スライダー値 × scale。
        is_integer: 実値が整数かどうか。
    """

    key: str
    label: str
    minimum: int
    maximum: int
    step: int
    default: int
    scale: float = 1.0
    is_integer: bool = True

    def to_value(self, slider_value: int) -> float | int:
        """スライダー値から実値へ変換する。

        小数の場合は桁を丸めて、既定値との比較で誤差が出ないようにする。
        """
        if self.is_integer:
            return int(slider_value)
        return round(slider_value * self.scale, 10)

    def to_slider(self, value: float) -> int:
        """実値からスライダー値へ変換する。"""
        if self.is_integer:
            return round(value)
        return round(value / self.scale)


#: 生成モードごとのパラメータ仕様（表示順）。既定値の一元管理も兼ねる。
PARAMETER_SPECS: dict[GenerationMode, tuple[ParameterSpec, ...]] = {
    GenerationMode.CLEAR: (),
    GenerationMode.MYOPIA: (
        ParameterSpec(
            key="gaussian_ksize",
            label="ガウシアンブラー カーネルサイズ",
            minimum=config.MYOPIA_GAUSSIAN_KSIZE_RANGE[0],
            maximum=config.MYOPIA_GAUSSIAN_KSIZE_RANGE[1],
            step=config.MYOPIA_GAUSSIAN_KSIZE_RANGE[2],
            default=config.MYOPIA_GAUSSIAN_KSIZE_DEFAULT,
        ),
    ),
    GenerationMode.CATARACT: (
        ParameterSpec(
            key="box_ksize",
            label="boxFilter カーネルサイズ",
            minimum=config.CATARACT_BOX_KSIZE_RANGE[0],
            maximum=config.CATARACT_BOX_KSIZE_RANGE[1],
            step=config.CATARACT_BOX_KSIZE_RANGE[2],
            default=config.CATARACT_BOX_KSIZE_DEFAULT,
        ),
        ParameterSpec(
            key="contrast",
            label="コントラスト係数 (x100)",
            minimum=config.CATARACT_CONTRAST_SLIDER_RANGE[0],
            maximum=config.CATARACT_CONTRAST_SLIDER_RANGE[1],
            step=config.CATARACT_CONTRAST_SLIDER_RANGE[2],
            default=round(config.CATARACT_CONTRAST_DEFAULT / config.CATARACT_CONTRAST_SCALE),
            scale=config.CATARACT_CONTRAST_SCALE,
            is_integer=False,
        ),
    ),
    GenerationMode.ASTIGMATISM: (
        ParameterSpec(
            key="astig_ksize",
            label="乱視 カーネルサイズ",
            minimum=config.ASTIGMATISM_KSIZE_RANGE[0],
            maximum=config.ASTIGMATISM_KSIZE_RANGE[1],
            step=config.ASTIGMATISM_KSIZE_RANGE[2],
            default=config.ASTIGMATISM_KSIZE_DEFAULT,
        ),
        ParameterSpec(
            key="astig_angle_deg",
            label="ぼかし方向 (0〜180度)",
            minimum=config.ASTIGMATISM_ANGLE_RANGE[0],
            maximum=config.ASTIGMATISM_ANGLE_RANGE[1],
            step=config.ASTIGMATISM_ANGLE_RANGE[2],
            default=int(config.ASTIGMATISM_ANGLE_DEFAULT),
            is_integer=False,
        ),
        ParameterSpec(
            key="astig_strength",
            label="ぼかし強度 (x100)",
            minimum=config.ASTIGMATISM_STRENGTH_SLIDER_RANGE[0],
            maximum=config.ASTIGMATISM_STRENGTH_SLIDER_RANGE[1],
            step=config.ASTIGMATISM_STRENGTH_SLIDER_RANGE[2],
            default=round(config.ASTIGMATISM_STRENGTH_DEFAULT / config.ASTIGMATISM_STRENGTH_SCALE),
            scale=config.ASTIGMATISM_STRENGTH_SCALE,
            is_integer=False,
        ),
    ),
}


def parameter_specs(mode: GenerationMode) -> tuple[ParameterSpec, ...]:
    """指定モードで使うパラメータ仕様を返す。"""
    return PARAMETER_SPECS[mode]


def default_parameters(mode: GenerationMode) -> dict[str, float | int]:
    """指定モードの既定パラメータ（実値）を返す。"""
    return {spec.key: spec.to_value(spec.default) for spec in PARAMETER_SPECS[mode]}


def all_default_parameters() -> dict[GenerationMode, dict[str, float | int]]:
    """全モードの既定パラメータ（実値）を返す。"""
    return {mode: default_parameters(mode) for mode in GenerationMode}


def _validate_odd_positive(name: str, value: int) -> None:
    """1 以上の奇数であることを検証する。"""
    if value < 1 or value % 2 == 0:
        raise ValueError(f"{name} は 1 以上の奇数である必要があります: {value}")


@dataclass(frozen=True)
class LandoltCParams:
    """ランドルト環 1 枚分の生成パラメータ。

    Attributes:
        vision: 視力値（0 より大きい）。
        direction: 切れ目の開く方向。
        mode: 生成モード。
        gaussian_ksize: 近視のガウシアンブラーのカーネルサイズ（奇数）。
        gaussian_sigma: 近視のガウシアンブラーのシグマ（0 なら自動算出）。
        box_ksize: 混濁の boxFilter のカーネルサイズ（奇数）。
        contrast: 混濁のコントラスト係数（0 より大きく 1 以下）。
        astig_ksize: 乱視のカスタムカーネルのサイズ（奇数）。
        astig_angle_deg: 乱視でぼかす方向（度）。
        astig_strength: 乱視のぼかしの強度（0 より大きく 1 以下）。
    """

    vision: float
    direction: Direction = "right"
    mode: GenerationMode = GenerationMode.CLEAR
    gaussian_ksize: int = config.MYOPIA_GAUSSIAN_KSIZE_DEFAULT
    gaussian_sigma: float = 0.0
    box_ksize: int = config.CATARACT_BOX_KSIZE_DEFAULT
    contrast: float = config.CATARACT_CONTRAST_DEFAULT
    astig_ksize: int = config.ASTIGMATISM_KSIZE_DEFAULT
    astig_angle_deg: float = config.ASTIGMATISM_ANGLE_DEFAULT
    astig_strength: float = config.ASTIGMATISM_STRENGTH_DEFAULT

    def __post_init__(self) -> None:
        """パラメータの妥当性を検証する。

        Raises:
            ValueError: 値が許容範囲外の場合。
        """
        if self.vision <= 0:
            raise ValueError(f"vision は 0 より大きい必要があります: {self.vision}")
        if self.direction not in config.DIRECTIONS:
            raise ValueError(f"direction が不正です: {self.direction!r}")
        _validate_odd_positive("gaussian_ksize", self.gaussian_ksize)
        _validate_odd_positive("box_ksize", self.box_ksize)
        _validate_odd_positive("astig_ksize", self.astig_ksize)
        if not 0.0 < self.contrast <= 1.0:
            raise ValueError(f"contrast は 0 より大きく 1 以下: {self.contrast}")
        if not 0.0 < self.astig_strength <= 1.0:
            raise ValueError(f"astig_strength は 0 より大きく 1 以下: {self.astig_strength}")

    def with_parameters(self, parameters: dict[str, float | int]) -> LandoltCParams:
        """モード固有のパラメータを上書きした新しいインスタンスを返す。

        Args:
            parameters: フィールド名をキーにした実値。

        Returns:
            上書き後の新しいインスタンス（元のインスタンスは変更しない）。

        Raises:
            KeyError: 存在しないフィールド名が指定された場合。
        """
        known = {f.name for f in fields(self)}
        unknown = set(parameters) - known
        if unknown:
            raise KeyError(f"未知のパラメータです: {sorted(unknown)}")
        return LandoltCParams(**{**self.as_dict(), **parameters})

    def as_dict(self) -> dict[str, Any]:
        """フィールド名をキーにした辞書を返す。"""
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------------
# 内部処理
# ---------------------------------------------------------------------------


def outer_diameter_px(vision: float) -> int:
    """視力値からランドルト環の外径（px）を求める。

    外径 = ``OUTER_DIAMETER_SCALE / 視力``。

    Args:
        vision: 視力値（0 より大きい）。

    Returns:
        外径（px、四捨五入した整数）。
    """
    return round(config.OUTER_DIAMETER_SCALE / vision)


#: 輪郭を均す幅（px）。境界からの距離をこの幅で 0..1 の被覆率に変換する
EDGE_SOFTNESS = 1.0


def _edge_coverage(distance_inside: np.ndarray) -> np.ndarray:
    """境界からの距離（px、内側が正）を被覆率（0..1）に変換する。

    画素は 1px 幅の正方形なので、境界が画素の中心を通るときの被覆率は 1/2、
    半画素ずれると 1 または 0 になる。境界のごく近くでは半平面と同じ形なので、
    この線形ランプが画素の面積被覆率に一致する（アンチエイリアスの基本）。

    Args:
        distance_inside: 境界から内側への距離（px）。

    Returns:
        0（完全に外側）〜1（完全に内側）の被覆率。
    """
    return np.clip(distance_inside / EDGE_SOFTNESS + 0.5, 0.0, 1.0)


def _slit_coverage(
    xs: np.ndarray,
    ys: np.ndarray,
    center: tuple[int, int],
    slit_half_width: float,
    direction: Direction,
) -> np.ndarray:
    """切れ目の被覆率を返す。

    切れ目は「環の中心から外側へ放射状に伸びる、幅 ``2 * slit_half_width`` の
    長方形」とする。中心側の辺は背景の中なので形には効かない。

    Args:
        xs: 評価する x 座標（画素中心）。
        ys: 評価する y 座標（画素中心）。
        center: 環の中心座標 (x, y)。
        slit_half_width: 切れ目の幅の半分（px）。
        direction: 切れ目の開く方向。

    Returns:
        形状 ``(len(ys), len(xs))`` の被覆率。
    """
    cx, cy = center
    # 中心から切れ目の方向へ測った距離（along）と、直角方向のずれ（across）
    if direction == "up":
        along, across = cy - ys[:, None], xs[None, :] - cx
    elif direction == "down":
        along, across = ys[:, None] - cy, xs[None, :] - cx
    elif direction == "left":
        along, across = cx - xs[None, :], ys[:, None] - cy
    else:
        along, across = xs[None, :] - cx, ys[:, None] - cy

    return _edge_coverage(along) * _edge_coverage(slit_half_width - np.abs(across))


def _ring_coverage(
    shape: tuple[int, int],
    center: tuple[int, int],
    outer_diameter: float,
    direction: Direction,
) -> tuple[tuple[slice, slice], np.ndarray]:
    """ランドルト環の被覆率（0..1）と、書き込む画素の範囲を返す。

    円の輪郭を「境界からの距離」で評価するため、1 画素の精度で滑らかな輪郭に
    なる（スーパーサンプリングは不要）。

    Args:
        shape: キャンバスの形状 ``(高さ, 幅)``。
        center: 環の中心座標 (x, y)。
        outer_diameter: 外径（px）。
        direction: 切れ目の開く方向。

    Returns:
        ``((行スライス, 列スライス), 被覆率)``。環がキャンバスに乗らない場合は
        被覆率の要素数が 0 になる。
    """
    outer_radius = outer_diameter / 2.0
    inner_radius = outer_radius * config.ISO_INNER_DIAMETER_RATIO
    slit_half_width = outer_radius * config.ISO_SLIT_WIDTH_RATIO

    cx, cy = center
    reach = int(np.ceil(outer_radius)) + 1
    x0, x1 = max(cx - reach, 0), min(cx + reach + 1, shape[1])
    y0, y1 = max(cy - reach, 0), min(cy + reach + 1, shape[0])

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    distance = np.hypot(xs[None, :] - cx, ys[:, None] - cy)

    # 円環 = 外円の内側 かつ 内円の外側
    coverage = _edge_coverage(outer_radius - distance)
    coverage *= _edge_coverage(distance - inner_radius)
    # 切れ目をくり抜く
    coverage *= 1.0 - _slit_coverage(xs, ys, center, slit_half_width, direction)
    return (slice(y0, y1), slice(x0, x1)), coverage


def _draw_clear_landolt_c(
    canvas: np.ndarray,
    center: tuple[int, int],
    outer_diameter: int,
    direction: Direction,
) -> None:
    """清明なランドルト環をキャンバスに直接描画する（ISO 8596 準拠）。

    外径 : 内径 : 切れ目の幅 = 5 : 3 : 1。同心かつ直径 0.6D の内円をくり抜いた
    円環に対し、外径の 20% 幅の切れ目を中心から放射状に 1 箇所設ける。

    輪郭は画素の被覆率で描く（アンチエイリアス）。``cv2.circle`` の 2 値の
    ラスタライズだと、環が小さいほど円のギザつきが目立つため使わない。

    Args:
        canvas: 描画先（グレースケール画像、この関数で書き換える）。
        center: 環の中心座標 (x, y)。
        outer_diameter: 外径（px、0 より大きければ描く）。
        direction: 切れ目の開く方向。
    """
    if outer_diameter <= 0:
        return

    (rows, cols), coverage = _ring_coverage(canvas.shape, center, outer_diameter, direction)
    if coverage.size == 0:
        return

    # 被覆率を不透明度として環の色を重ねる（縁は背景と混ざって滑らかになる）
    region = canvas[rows, cols].astype(np.float32)
    ink = float(config.RING_GRAY)
    canvas[rows, cols] = np.rint(region + (ink - region) * coverage).clip(0, 255).astype(np.uint8)


def _apply_gaussian_blur(image: np.ndarray, spec: LandoltCParams) -> np.ndarray:
    """ガウシアンブラーを適用する（近視の模擬）。"""
    return cv2.GaussianBlur(
        image,
        (spec.gaussian_ksize, spec.gaussian_ksize),
        spec.gaussian_sigma,
    )


def _apply_box_filter_and_contrast(image: np.ndarray, spec: LandoltCParams) -> np.ndarray:
    """平滑化（boxFilter）とコントラスト低下を適用する（混濁の模擬）。"""
    smoothed = cv2.boxFilter(image, -1, (spec.box_ksize, spec.box_ksize))
    # gray = 128 + (gray - 128) * contrast
    centered = smoothed.astype(np.float32) - 128.0
    result = np.clip(128.0 + centered * spec.contrast, 0.0, 255.0)
    return result.astype(np.uint8)


def _astigmatism_kernel(spec: LandoltCParams) -> np.ndarray:
    """乱視用の楕円カーネル（一方向にだけぼかす）を作る。"""
    ksize = spec.astig_ksize
    center = ksize // 2
    theta = np.deg2rad(spec.astig_angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    semi_major = max(1, int(ksize * spec.astig_strength / 2))
    semi_minor = max(1, int(ksize * (1.0 - spec.astig_strength * 0.7) / 2))

    ys, xs = np.mgrid[0:ksize, 0:ksize]
    dx = (xs - center).astype(np.float64)
    dy = (ys - center).astype(np.float64)
    rx = dx * cos_t + dy * sin_t
    ry = -dx * sin_t + dy * cos_t

    inside = (rx / semi_major) ** 2 + (ry / semi_minor) ** 2 <= 1.0
    kernel = inside.astype(np.float32)
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def _apply_astigmatism(image: np.ndarray, spec: LandoltCParams) -> np.ndarray:
    """カスタムカーネルによる filter2D を適用する（乱視の模擬）。"""
    return cv2.filter2D(image, -1, _astigmatism_kernel(spec))


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

#: 生成モード -> フィルタ関数
_FILTERS = {
    GenerationMode.MYOPIA: _apply_gaussian_blur,
    GenerationMode.CATARACT: _apply_box_filter_and_contrast,
    GenerationMode.ASTIGMATISM: _apply_astigmatism,
}


def generate_landolt_c_image(spec: LandoltCParams) -> np.ndarray:
    """ランドルト環画像を 1 枚生成する。

    キャンバスは ``CANVAS_SIZE`` の正方形で、背景は白、環は黒。
    環の中心は常にキャンバスの中心と一致する。

    Args:
        spec: 生成パラメータ。

    Returns:
        グレースケール（``uint8``）の画像。
    """
    canvas = np.full(
        (config.CANVAS_SIZE, config.CANVAS_SIZE),
        config.BACKGROUND_GRAY,
        dtype=np.uint8,
    )
    center = (config.CANVAS_SIZE // 2, config.CANVAS_SIZE // 2)
    _draw_clear_landolt_c(canvas, center, outer_diameter_px(spec.vision), spec.direction)

    filter_func = _FILTERS.get(spec.mode)
    return filter_func(canvas, spec) if filter_func is not None else canvas


def generate_sample_image(
    mode: GenerationMode,
    direction: Direction = "right",
    **parameters: float | int,
) -> np.ndarray:
    """左パネルのサンプル表示用のランドルト環画像を生成する。

    視力値に依存しない大きな環（``SAMPLE_VISION``）を生成し、表示側で縮小する。

    Args:
        mode: 生成モード。
        direction: 切れ目の開く方向（サンプルでは右固定で使う）。
        **parameters: モード固有パラメータの上書き（フィールド名＝実値）。

    Returns:
        グレースケール（``uint8``）の画像。
    """
    spec = LandoltCParams(vision=config.SAMPLE_VISION, direction=direction, mode=mode)
    return generate_landolt_c_image(spec.with_parameters(parameters))


def encode_png_base64(image: np.ndarray) -> str:
    """画像を PNG にエンコードし、Base64 文字列（データ URI なし）で返す。

    Args:
        image: グレースケール画像。

    Returns:
        改行を含まない Base64 文字列。

    Raises:
        RuntimeError: PNG エンコードに失敗した場合。
    """
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG エンコードに失敗しました")
    return base64.b64encode(encoded).decode("ascii")
