"""視覚検査の進行（判定フロー）を司るステートマシン。

docs/spec.md の検査フロー（ステップ 1〜8）を、GUI に依存しない形で実装する。

各視力値について、以下の優先順で判定する:

1. 初回で正答 → 次の視力値へ（ただし上限値なら検査終了）
2. 初回で誤答 → 同じ視力値でもう一度（累計誤答 1 回）
3. 累計誤答が 2 回 → 検査終了（確定視力は最後に正答した値）
4. 累計正答が 2 回（誤答→正答→正答 を含む）→ 次の視力値へ（上限値なら検査終了）
5. 上記のいずれにも当てはまらない（誤答→正答 の直後）→ 同じ視力値でもう一度
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from systemone_vision_test import config


class InspectionStatus(str, Enum):
    """検査の進行状態。"""

    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class InspectionState:
    """検査のスナップショット。

    Attributes:
        status: 進行状態。
        vision_levels: 検査で使う視力値（高い順）。
        vision_index: 現在の視力値のインデックス。
        error_counts: 視力値ごとの累計誤答回数。
        success_counts: 視力値ごとの累計正答回数。
        last_correct_vision: 最後に正答した視力値（未正答なら ``None``）。
        final_vision: 検査終了時に確定した視力値（未確定なら ``None``）。
        last_answer: clef が最後に回答した方向。
        last_correct: 最後の判定が正答だったか。
        last_probabilities: 最後の応答の確率。
    """

    vision_levels: tuple[float, ...] = config.VISION_LEVELS
    status: InspectionStatus = InspectionStatus.IDLE
    vision_index: int = 0
    error_counts: dict[float, int] = field(default_factory=dict)
    success_counts: dict[float, int] = field(default_factory=dict)
    last_correct_vision: float | None = None
    final_vision: float | None = None
    last_answer: str | None = None
    last_correct: bool | None = None
    last_probabilities: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """開始時の視力値（下限）にインデックスを合わせる。"""
        if not self.vision_levels:
            raise ValueError("vision_levels が空です")
        self.vision_index = len(self.vision_levels) - 1

    @property
    def current_vision(self) -> float:
        """現在の視力値。"""
        return self.vision_levels[self.vision_index]

    @property
    def is_lowest_vision(self) -> bool:
        """現在の視力値が下限かどうか。"""
        return self.vision_index == len(self.vision_levels) - 1


class InspectionEngine:
    """検査フローを進めるエンジン。

    このクラスは Qt に依存しないため、単体テストでは実際の GUI なしで検証できる。

    Example:
        >>> engine = InspectionEngine()
        >>> engine.start()
        >>> engine.current_vision
        0.1
    """

    def __init__(self, vision_levels: tuple[float, ...] = config.VISION_LEVELS) -> None:
        """エンジンを初期化する。

        Args:
            vision_levels: 検査で使う視力値（高い順）。既定は docs/spec.md の値。
        """
        self._vision_levels = tuple(vision_levels)
        self._state = InspectionState(vision_levels=self._vision_levels)

    # -- 状態 ---------------------------------------------------------------

    @property
    def state(self) -> InspectionState:
        """現在の状態（読み取り用）。"""
        return self._state

    @property
    def vision_levels(self) -> tuple[float, ...]:
        """検査で使う視力値（高い順）。"""
        return self._vision_levels

    @property
    def current_vision(self) -> float:
        """現在の視力値。"""
        return self._state.current_vision

    @property
    def final_vision(self) -> float | None:
        """検査終了時に確定した視力値（未確定なら ``None``）。"""
        return self._state.final_vision

    @property
    def last_correct_vision(self) -> float | None:
        """最後に正答した視力値（未正答なら ``None``）。"""
        return self._state.last_correct_vision

    @property
    def is_running(self) -> bool:
        """検査が実行中かどうか。"""
        return self._state.status is InspectionStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        """検査が終了したかどうか。"""
        return self._state.status is InspectionStatus.FINISHED

    @property
    def current_error_count(self) -> int:
        """現在の視力値での累計誤答回数。"""
        return self._state.error_counts.get(self._state.current_vision, 0)

    # -- 操作 ---------------------------------------------------------------

    def reset(self) -> None:
        """検査を未開始の状態に戻す（記録は保持しない）。"""
        self._state = InspectionState(vision_levels=self._vision_levels)

    def start(self) -> None:
        """検査を開始する（視力値を下限にリセットする）。"""
        self._state = InspectionState(
            vision_levels=self._vision_levels, status=InspectionStatus.RUNNING
        )

    def record_result(
        self,
        answer: str,
        is_correct: bool,
        probabilities: dict[str, float] | None = None,
    ) -> None:
        """1 回分の判定結果を取り込み、状態を遷移させる。

        Args:
            answer: clef が回答した方向。
            is_correct: 判定が正答だったか。
            probabilities: clef が返した方向ごとの確率（表示用。省略可）。
        """
        state = self._state
        if state.status is not InspectionStatus.RUNNING:
            raise RuntimeError("検査が実行中ではありません")

        vision = state.current_vision
        state.last_answer = answer
        state.last_correct = is_correct
        state.last_probabilities = dict(probabilities or {})

        if is_correct:
            state.success_counts[vision] = state.success_counts.get(vision, 0) + 1
            state.last_correct_vision = vision
            if self._should_advance(vision):
                self._advance()
        else:
            state.error_counts[vision] = state.error_counts.get(vision, 0) + 1
            if state.error_counts[vision] >= 2:
                self._finish()

    # -- 内部処理 -----------------------------------------------------------

    def _should_advance(self, vision: float) -> bool:
        """現在の視力値で次の段階に進めるかどうか。"""
        errors = self._state.error_counts.get(vision, 0)
        successes = self._state.success_counts.get(vision, 0)
        return (errors == 0 and successes == 1) or successes >= 2

    def _advance(self) -> None:
        """視力値を一段階上げる。上限値なら検査を終了する。"""
        if self._state.vision_index == 0:
            self._finish()
            return
        self._state.vision_index -= 1

    def _finish(self) -> None:
        """検査を終了し、最後に正答した視力値を確定値とする。"""
        self._state.status = InspectionStatus.FINISHED
        self._state.final_vision = self._state.last_correct_vision
