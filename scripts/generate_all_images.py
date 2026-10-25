"""全モード・全視力値のランドルト環画像を一括生成する。

出力先は ``<プロジェクト>/image/<モード名>/landolt_<モード名>_<視力>.png``。
切れ目の方向は上下左右からランダムに選ぶ。

使い方::

    python scripts/generate_all_images.py
    python scripts/generate_all_images.py --seed 42 --output image
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# src レイアウトのため、未インストールでも実行できるようにする簡易ブートストラップ
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import cv2

from systemone_vision_test import config
from systemone_vision_test.domain.landolt import (
    GenerationMode,
    LandoltCParams,
    generate_landolt_c_image,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解釈する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.IMAGE_OUTPUT_DIR,
        help="出力先ディレクトリ（既定: %(default)s）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="切れ目の方向を決める乱数のシード（再現したいときに指定）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """画像を一括生成する。

    Args:
        argv: コマンドライン引数（``None`` なら ``sys.argv``）。

    Returns:
        終了コード。
    """
    args = parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for mode in GenerationMode:
        mode_dir = output_dir / mode.value
        mode_dir.mkdir(parents=True, exist_ok=True)
        for vision in config.VISION_LEVELS:
            direction = random.choice(config.DIRECTIONS)
            spec = LandoltCParams(vision=vision, direction=direction, mode=mode)
            image = generate_landolt_c_image(spec)

            path = mode_dir / f"landolt_{mode.value}_{vision}.png"
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError(f"PNG エンコードに失敗しました: {path}")
            path.write_bytes(encoded.tobytes())
            print(f"生成: {path} (direction={direction})")
            total += 1

    print(f"\n完了: {total} 枚 / 出力先: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
