"""検査フロー（domain.inspection）のテスト。

docs/spec.md のフロー（ステップ 6・7）の分岐を、実際の GUI なしで検証する。
"""

from __future__ import annotations

import pytest

from systemone_vision_test import config
from systemone_vision_test.domain.inspection import (
    InspectionEngine,
    InspectionStatus,
)

LEVELS = config.VISION_LEVELS
LOWEST = LEVELS[-1]  # 0.1
SECOND = LEVELS[-2]  # 0.2
HIGHEST = LEVELS[0]  # 2.0
ONE_BELOW_HIGHEST = LEVELS[1]  # 1.5


def run(results: list[bool], engine: InspectionEngine | None = None) -> InspectionEngine:
    """判定結果の列を順に取り込み、エンジンを返す。"""
    engine = engine or InspectionEngine()
    engine.start()
    for is_correct in results:
        if engine.is_finished:
            break
        engine.record_result("up", is_correct)
    return engine


# ---------------------------------------------------------------------------
# 初期状態
# ---------------------------------------------------------------------------


def test_start_resets_to_lowest_vision():
    """検査開始時は視力が下限になる。"""
    engine = InspectionEngine()
    engine.start()

    assert engine.current_vision == LOWEST
    assert engine.is_running
    assert not engine.is_finished
    assert engine.final_vision is None


def test_initial_status_is_idle():
    """生成直後は未開始（IDLE）である。"""
    assert InspectionEngine().state.status is InspectionStatus.IDLE


def test_reset_clears_record():
    """``reset`` は記録も状態も消す。"""
    engine = run([True, True])
    engine.reset()

    assert engine.state.status is InspectionStatus.IDLE
    assert engine.current_vision == LOWEST
    assert engine.final_vision is None
    assert engine.current_error_count == 0


# ---------------------------------------------------------------------------
# 正答・誤答の分岐（ステップ 6・7）
# ---------------------------------------------------------------------------


def test_first_correct_advances_vision():
    """初回で正答したら次の視力値へ進む。"""
    engine = run([True])

    assert engine.current_vision == SECOND
    assert engine.last_correct_vision == LOWEST
    assert not engine.is_finished


def test_first_wrong_repeats_same_vision():
    """初回で誤答したら同じ視力値でもう一度。"""
    engine = run([False])

    assert engine.current_vision == LOWEST
    assert engine.current_error_count == 1
    assert not engine.is_finished


def test_wrong_wrong_finishes_without_valid_vision():
    """×→× は検査終了。まだ一度も正答していないので確定値は無し。"""
    engine = run([False, False])

    assert engine.is_finished
    assert engine.final_vision is None


def test_wrong_then_correct_repeats_same_vision():
    """×→〇 は同じ視力値でもう一度（まだ進めない）。"""
    engine = run([False, True])

    assert engine.current_vision == LOWEST
    assert engine.last_correct_vision == LOWEST
    assert not engine.is_finished


def test_wrong_correct_correct_advances():
    """×→〇→〇 で次の視力値へ進む。"""
    engine = run([False, True, True])

    assert engine.current_vision == SECOND
    assert not engine.is_finished


def test_wrong_correct_wrong_finishes_with_previous_vision():
    """×→〇→× は検査終了。確定値は「最後に正答した視力値」。"""
    engine = run([False, True, False])

    assert engine.is_finished
    assert engine.final_vision == LOWEST


def test_error_count_is_per_vision():
    """誤答回数は視力値ごとに数える。"""
    engine = run([False, True, True])  # 0.1 で ×→〇→〇 → 0.2 へ

    assert engine.current_vision == SECOND
    assert engine.current_error_count == 0  # 0.2 ではまだ誤答なし
    assert engine.state.error_counts[LOWEST] == 1


def test_reaches_highest_vision_by_repeated_correct():
    """正答を繰り返すと上限まで上がる（各視力 1 回目で進む）。"""
    engine = run([True] * (len(LEVELS) - 1))

    assert engine.current_vision == HIGHEST
    assert not engine.is_finished


def test_correct_at_highest_vision_finishes():
    """上限値で正答したら検査終了し、確定値は上限値。"""
    engine = run([True] * len(LEVELS))

    assert engine.is_finished
    assert engine.final_vision == HIGHEST


def test_wrong_twice_at_highest_vision_uses_previous_vision():
    """上限値で ×→× なら確定値は一つ下の視力値。"""
    results = [True] * (len(LEVELS) - 1) + [False, False]

    engine = run(results)

    assert engine.is_finished
    assert engine.final_vision == ONE_BELOW_HIGHEST


@pytest.mark.parametrize(
    ("results", "expected_vision", "expected_finished"),
    [
        ([True], SECOND, False),
        ([False, True], LOWEST, False),
        ([False, False], None, True),
        ([False, True, False], LOWEST, True),
        ([False, True, True], SECOND, False),
    ],
)
def test_flow_matrix(results, expected_vision, expected_finished):
    """docs/spec.md のフロー分岐を一覧で確認する。"""
    engine = run(results)
    actual_vision = engine.final_vision if engine.is_finished else engine.current_vision

    assert actual_vision == expected_vision
    assert engine.is_finished is expected_finished


# ---------------------------------------------------------------------------
# 異常系
# ---------------------------------------------------------------------------


def test_record_result_without_start_raises():
    """開始前に結果を取り込むと ``RuntimeError``。"""
    engine = InspectionEngine()
    with pytest.raises(RuntimeError):
        engine.record_result("up", True)


def test_empty_vision_levels_raises():
    """視力値のリストが空なら ``ValueError``。"""
    with pytest.raises(ValueError):
        InspectionEngine(vision_levels=())


def test_custom_vision_levels_are_supported():
    """視力値のリストを差し替えられる（テスト容易性）。"""
    engine = InspectionEngine(vision_levels=(1.0, 0.5))
    engine.start()

    assert engine.current_vision == 0.5
    engine.record_result("up", True)
    assert engine.current_vision == 1.0
    engine.record_result("up", True)
    assert engine.final_vision == 1.0


def test_state_records_last_answer():
    """最後の回答と確率が状態に残る。"""
    engine = InspectionEngine()
    engine.start()
    engine.record_result("right", True, {"right": 0.9})

    assert engine.state.last_answer == "right"
    assert engine.state.last_correct is True
    assert engine.state.last_probabilities == {"right": 0.9}
