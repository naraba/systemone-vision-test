"""統計を取る組み合わせ（実行計画）の定義。

GUI 版の検査は正否に応じて視力値を段階的に変えるが、統計では**全視力値 × 全方向を
同じ回数ずつ**呈示して、条件ごとの正答率をそのまま比較できるようにする。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from systemone_vision_test import config
from systemone_vision_test.domain.landolt import (
    Direction,
    GenerationMode,
    default_parameters,
    parameter_specs,
)

#: 各組み合わせを何回呈示するか（既定）
DEFAULT_REPEATS = 4

#: :data:`config.DIRECTIONS` を :data:`Direction` として扱う
DIRECTIONS: tuple[Direction, ...] = cast("tuple[Direction, ...]", config.DIRECTIONS)


@dataclass(frozen=True)
class Presentation:
    """1 回分の呈示条件。

    Attributes:
        model: 使用するモデル名。
        mode: 生成モード。
        vision: 視力値。
        direction: 切れ目の開く方向。
        repeat: 同じ条件の何回目の呈示か（0 始まり）。
    """

    model: str
    mode: GenerationMode
    vision: float
    direction: Direction
    repeat: int

    def __str__(self) -> str:
        """ログや進捗表示に使う表記。"""
        return (
            f"{self.model} / {self.mode.value} / 視力 {self.vision}"
            f" / {self.direction} / {self.repeat + 1} 回目"
        )


@dataclass(frozen=True)
class BenchmarkPlan:
    """呈示する組み合わせの集合（モデル × モード × 視力値 × 方向 × 反復）。

    Attributes:
        models: 対象モデル。
        modes: 対象モード。
        visions: 対象の視力値。
        directions: 対象の切れ目の方向。
        repeats: 各組み合わせの呈示回数。
    """

    models: tuple[str, ...] = config.BENCHMARK_MODELS
    modes: tuple[GenerationMode, ...] = tuple(GenerationMode)
    visions: tuple[float, ...] = config.VISION_LEVELS
    directions: tuple[Direction, ...] = DIRECTIONS
    repeats: int = DEFAULT_REPEATS

    # -- 計画の中身 ---------------------------------------------------------

    @property
    def total(self) -> int:
        """呈示の総数。"""
        return (
            len(self.models)
            * len(self.modes)
            * len(self.visions)
            * len(self.directions)
            * self.repeats
        )

    def __len__(self) -> int:
        """呈示の総数。"""
        return self.total

    def groups(self) -> Iterator[tuple[str, GenerationMode, tuple[Presentation, ...]]]:
        """モデルとモードごとの呈示のまとまりを返す。

        実行・進捗表示・レポートの単位。同じモデルを続けて使うので、モデルの
        切り替え（Ollama のモデル読み込み）が最小になる。

        Yields:
            ``(モデル, モード, そのまとまりの呈示)``。
        """
        for model in self.models:
            for mode in self.modes:
                yield (
                    model,
                    mode,
                    tuple(
                        Presentation(
                            model=model,
                            mode=mode,
                            vision=vision,
                            direction=direction,
                            repeat=repeat,
                        )
                        for vision in self.visions
                        for direction in self.directions
                        for repeat in range(self.repeats)
                    ),
                )

    def presentations(self) -> Iterator[Presentation]:
        """すべての呈示を、実行する順（モデル → モード → 視力値 → 方向 → 反復）で返す。

        Yields:
            呈示条件。
        """
        for _, _, group in self.groups():
            yield from group

    # -- パラメータ ---------------------------------------------------------

    def parameters(self, mode: GenerationMode) -> dict[str, float | int]:
        """モードで使うパラメータ（実値）を返す。

        GUI 版と同じく、各モードの既定値（`domain/landolt.py` の ``default_parameters``）を使う。

        Args:
            mode: 対象のモード。

        Returns:
            パラメータ名をキーにした実値（清明は空）。
        """
        return default_parameters(mode)

    def parameter_rows(self) -> list[tuple[str, str, str]]:
        """レポートの前提条件に載せるパラメータ表を作る。

        Returns:
            ``(モード名, パラメータのラベル, 値)`` の並び。同じモードの 2 行目以降は
            モード名を空にして、表でまとまって見えるようにする。
        """
        rows: list[tuple[str, str, str]] = []
        for mode in self.modes:
            values = self.parameters(mode)
            specs = parameter_specs(mode)
            if not specs:
                rows.append((mode.value, "—", "—"))
                continue
            for index, spec in enumerate(specs):
                name = mode.value if index == 0 else ""
                rows.append((name, spec.label, str(values[spec.key])))
        return rows
