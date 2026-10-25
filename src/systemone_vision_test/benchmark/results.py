"""呈示ごとの結果と、条件別の集計。

「モデル × モード × 視力値 × 開いている向き」の正答率を求めるための型を定義する。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from systemone_vision_test import config
from systemone_vision_test.benchmark.plan import BenchmarkPlan, Presentation
from systemone_vision_test.domain.landolt import Direction, GenerationMode
from systemone_vision_test.domain.questions import DIRECTION_QUESTION


@dataclass(frozen=True)
class Attempt:
    """1 回の呈示と、その結果。

    Attributes:
        presentation: 呈示した条件。
        answer: API が返した回答キーそのまま（回答が得られなければ ``None``）。
        is_correct: 正答だったか（回答が得られなければ ``None``）。
        elapsed_sec: 判定にかかった時間（秒）。
        probabilities: 回答の確率（記録用。キーは質問の回答キー）。
        answered_direction: 回答キーを方向に変換した値（未知のキーなら ``None``）。
    """

    presentation: Presentation
    answer: str | None
    is_correct: bool | None
    elapsed_sec: float
    probabilities: dict[str, float] = field(default_factory=dict)
    answered_direction: str | None = None

    @property
    def failed(self) -> bool:
        """回答が得られなかったかどうか（通信エラーなど）。"""
        return self.answer is None

    def as_dict(self) -> dict[str, Any]:
        """生データ（JSON）用の辞書にする。

        Returns:
            呈示条件と結果を平坦にした辞書。
        """
        presentation = self.presentation
        return {
            "model": presentation.model,
            "mode": presentation.mode.value,
            "vision": presentation.vision,
            "direction": presentation.direction,
            "repeat": presentation.repeat + 1,
            "answer": self.answer,
            "answered_direction": self.answered_direction,
            "is_correct": self.is_correct,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "probabilities": dict(self.probabilities),
        }


@dataclass(frozen=True)
class Tally:
    """正答数の集計。

    Attributes:
        correct: 正答の数。
        wrong: 誤答の数。
        failed: 回答が得られなかった数。
    """

    correct: int = 0
    wrong: int = 0
    failed: int = 0

    @property
    def total(self) -> int:
        """呈示の数。"""
        return self.correct + self.wrong + self.failed

    @property
    def answered(self) -> int:
        """回答が得られた数。"""
        return self.correct + self.wrong

    @property
    def accuracy(self) -> float | None:
        """正答率（呈示数に対する正答。0 件なら ``None``）。"""
        return self.correct / self.total if self.total else None

    @property
    def answer_rate(self) -> float | None:
        """回答率（呈示数に対して回答が返った割合。0 件なら ``None``）。"""
        return self.answered / self.total if self.total else None

    def __add__(self, other: Tally) -> Tally:
        """集計を足し合わせる。"""
        return Tally(
            correct=self.correct + other.correct,
            wrong=self.wrong + other.wrong,
            failed=self.failed + other.failed,
        )

    @classmethod
    def of_attempts(cls, attempts: Iterable[Attempt]) -> Tally:
        """呈示の並びから集計する。

        Args:
            attempts: 集計する呈示の結果。

        Returns:
            集計結果。
        """
        tally = cls()
        for attempt in attempts:
            if attempt.failed:
                tally += cls(failed=1)
            elif attempt.is_correct:
                tally += cls(correct=1)
            else:
                tally += cls(wrong=1)
        return tally


def completed_groups(
    plan: BenchmarkPlan, attempts: Iterable[Attempt]
) -> frozenset[tuple[str, GenerationMode]]:
    """呈示がすべて記録済みの（モデル, モード）を返す。

    Args:
        plan: 実行計画（1 まとまりの呈示数の算出に使う）。
        attempts: 記録済みの呈示。

    Returns:
        記録が揃っているまとまりの組。中断した実行の再開に使う。
    """
    expected = len(plan.visions) * len(plan.directions) * plan.repeats
    counts: dict[tuple[str, GenerationMode], int] = {}
    for attempt in attempts:
        presentation = attempt.presentation
        key = (presentation.model, presentation.mode)
        counts[key] = counts.get(key, 0) + 1
    return frozenset(key for key, count in counts.items() if count >= expected)


@dataclass(frozen=True)
class BenchmarkResult:
    """実行結果（すべての呈示と、条件別に集計するための検索）。

    Attributes:
        plan: 実行した計画。
        attempts: 呈示ごとの結果。
        started_at: 実行を始めた時刻。
        finished_at: 実行が終わった時刻。
        connection_checks: モデルごとの接続確認の結果。
        endpoint: 接続先（レポートの前提条件に載せる）。
        question_name: 使った質問の名前（回答キーの種類が分かるようにする）。
        resumed_from: 中断した実行から引き継いだ呈示の数（新規なら 0）。
    """

    plan: BenchmarkPlan
    attempts: tuple[Attempt, ...]
    started_at: datetime
    finished_at: datetime
    connection_checks: dict[str, bool] = field(default_factory=dict)
    endpoint: str = config.CLEF_BASE_URL
    question_name: str = DIRECTION_QUESTION.name
    resumed_from: int = 0

    @property
    def duration_sec(self) -> float:
        """実行にかかった時間（秒）。"""
        return (self.finished_at - self.started_at).total_seconds()

    def tally(
        self,
        *,
        model: str | None = None,
        mode: GenerationMode | None = None,
        vision: float | None = None,
        direction: Direction | None = None,
    ) -> Tally:
        """条件に一致する呈示を集計する。

        Args:
            model: 絞り込むモデル（``None`` なら絞り込まない。以下同様）。
            mode: 絞り込むモード。
            vision: 絞り込む視力値。
            direction: 絞り込む切れ目の方向。

        Returns:
            条件に一致した呈示の集計。
        """
        correct = wrong = failed = 0
        for attempt in self.attempts:
            presentation = attempt.presentation
            if model is not None and presentation.model != model:
                continue
            if mode is not None and presentation.mode is not mode:
                continue
            if vision is not None and presentation.vision != vision:
                continue
            if direction is not None and presentation.direction != direction:
                continue
            if attempt.failed:
                failed += 1
            elif attempt.is_correct:
                correct += 1
            else:
                wrong += 1
        return Tally(correct=correct, wrong=wrong, failed=failed)

    def response_times(self, *, model: str | None = None) -> list[float]:
        """回答が得られた呈示の応答時間（秒）を返す。

        Args:
            model: 絞り込むモデル（``None`` なら全モデル）。

        Returns:
            応答時間の並び（呈示した順）。
        """
        return [
            attempt.elapsed_sec
            for attempt in self.attempts
            if not attempt.failed and (model is None or attempt.presentation.model == model)
        ]

    def failed_attempts(self) -> list[Attempt]:
        """回答が得られなかった呈示を返す。"""
        return [attempt for attempt in self.attempts if attempt.failed]

    # -- 中断からの再開 -----------------------------------------------------

    def completed_group_keys(self) -> frozenset[tuple[str, GenerationMode]]:
        """呈示がすべて記録済みの（モデル, モード）を返す。

        Returns:
            記録が揃っているまとまりの組。
        """
        return completed_groups(self.plan, self.attempts)

    def keep_complete_groups(self) -> BenchmarkResult:
        """呈示が揃っていないまとまりを捨てた結果を返す（中断からの再開用）。

        途中まで書かれたまとまりは、再開時に最初から呈示し直すため、その一部だけの
        記録は使わない。

        Returns:
            揃っているまとまりだけを持つ結果。
        """
        completed = self.completed_group_keys()
        attempts = tuple(
            attempt
            for attempt in self.attempts
            if (attempt.presentation.model, attempt.presentation.mode) in completed
        )
        return replace(self, attempts=attempts)
