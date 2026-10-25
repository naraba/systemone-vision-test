"""統計結果のレポート（Markdown）と生データ（JSON）の書き出し。

GUI に依存しないよう、方向の表示名などレポート用の表現はこの module で定義する。
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from systemone_vision_test import config
from systemone_vision_test.benchmark.plan import BenchmarkPlan, Presentation
from systemone_vision_test.benchmark.results import Attempt, BenchmarkResult, Tally
from systemone_vision_test.domain.landolt import Direction, GenerationMode
from systemone_vision_test.domain.questions import DIRECTION_QUESTION

#: 表で使う切れ目の方向の表示名
DIRECTION_LABELS: dict[str, str] = {"up": "上", "down": "下", "left": "左", "right": "右"}

#: パラメータが無いモードの表記
NO_VALUE = "—"

#: 確信度（回答の確率の最大値）の帯。``(下限, 上限, 表示名)``。
#: 低確信の呈示を「判定不能」として切り分けられるようにする。
CONFIDENCE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.8, 1.01, "0.8 以上"),
    (0.6, 0.8, "0.6〜0.8"),
    (0.4, 0.6, "0.4〜0.6"),
    (0.0, 0.4, "0.4 未満"),
)


def direction_label(direction: str) -> str:
    """切れ目の方向の表示名を返す。

    Args:
        direction: ``up`` / ``down`` / ``left`` / ``right``。

    Returns:
        表示名（``上`` など）。未知の値はそのまま返す。
    """
    return DIRECTION_LABELS.get(direction, direction)


def format_tally(tally: Tally) -> str:
    """集計を ``3/4 (75%)`` の形にする。

    Args:
        tally: 集計結果。

    Returns:
        表示用の文字列。呈示が無ければ ``—``。
    """
    rate = tally.accuracy
    return NO_VALUE if rate is None else f"{tally.correct}/{tally.total} ({rate:.0%})"


def format_duration(seconds: float) -> str:
    """秒を ``17 分 14 秒`` の形にする。

    Args:
        seconds: 秒数。

    Returns:
        表示用の文字列。
    """
    minutes, sec = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} 時間 {minutes} 分 {sec} 秒"
    if minutes:
        return f"{minutes} 分 {sec} 秒"
    return f"{sec} 秒"


def render_markdown(result: BenchmarkResult) -> str:
    """レポート（Markdown）を組み立てる。

    Args:
        result: 実行結果。

    Returns:
        Markdown 形式のレポート。
    """
    lines: list[str] = []
    lines += _render_header(result)
    lines += _render_preconditions(result)
    lines += _render_results(result)
    lines += _render_summary(result)
    lines += _render_diagnostics(result)
    lines += _render_notes()
    return "\n".join(lines).rstrip() + "\n"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Markdown の表を組み立てる。"""
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def _render_header(result: BenchmarkResult) -> list[str]:
    """表題と実行日時を組み立てる。"""
    lines = [
        f"# {' / '.join(result.plan.models)} 全モード統計レポート",
        "",
        f"- 実行日時: {result.started_at:%Y-%m-%d %H:%M:%S} 〜 "
        f"{result.finished_at:%H:%M:%S}（所要 {format_duration(result.duration_sec)}）",
        f"- 呈示の総数: {len(result.attempts)} 件",
    ]
    if result.resumed_from:
        lines.append(
            f"- 中断から再開している（前回の実行から {result.resumed_from} 件を引き継ぎ。"
            "所要時間は今回の実行分のみ）"
        )
    lines.append("")
    return lines


def _render_preconditions(result: BenchmarkResult) -> list[str]:
    """前提条件（実行環境・検査の構成・モードごとのパラメータ）を組み立てる。"""
    lines = ["## 1. 前提条件", ""]
    lines += _render_environment(result)
    lines += _render_composition(result.plan)
    lines += _render_parameter_table(result.plan)
    return lines


def _render_environment(result: BenchmarkResult) -> list[str]:
    """実行環境（接続先・モデル・呈示数など）の表を組み立てる。"""
    plan = result.plan
    per_model = plan.total // len(plan.models) if plan.models else 0
    connectivity = (
        " / ".join(
            f"{model}: {'OK' if ok else 'NG'}" for model, ok in result.connection_checks.items()
        )
        or NO_VALUE
    )
    return [
        "### 実行環境",
        "",
        *_table(
            ["項目", "値"],
            [
                ["接続先（API エンドポイント）", result.endpoint],
                ["モデル", ", ".join(plan.models)],
                ["接続確認", connectivity],
                ["呈示の総数", f"{plan.total} 件（モデルあたり {per_model} 件）"],
                ["反復回数", f"各条件 {plan.repeats} 回（同じ画像を繰り返し呈示）"],
                [
                    "キャンバス",
                    f"{config.CANVAS_SIZE} × {config.CANVAS_SIZE} px"
                    "（ISO 8596・輪郭は画素の被覆率で描画）",
                ],
                ["判定ごとの待ち時間", "なし（画面表示を伴わないため）"],
                ["質問", result.question_name],
                ["所要時間", format_duration(result.duration_sec)],
                [
                    "中断からの再開",
                    f"あり（前回から {result.resumed_from} 件を引き継ぎ）"
                    if result.resumed_from
                    else "なし",
                ],
            ],
        ),
    ]


def _render_composition(plan: BenchmarkPlan) -> list[str]:
    """検査の構成（モード・視力値・方向・繰り返し）の表を組み立てる。"""
    return [
        "",
        "### 検査の構成",
        "",
        *_table(
            ["項目", "内容"],
            [
                ["モード", " / ".join(mode.value for mode in plan.modes)],
                ["視力値", ", ".join(f"{vision:g}" for vision in plan.visions)],
                ["切れ目の方向", " / ".join(direction_label(d) for d in plan.directions)],
                [
                    "組み合わせ",
                    f"モデル × モード × 視力値 × 方向の各条件を {plan.repeats} 回ずつ呈示",
                ],
            ],
        ),
    ]


def _render_parameter_table(plan: BenchmarkPlan) -> list[str]:
    """モードごとのパラメータ（既定値）の表を組み立てる。"""
    return [
        "",
        "### モードごとのパラメータ（各モードの既定値）",
        "",
        *_table(["モード", "パラメータ", "値"], [list(row) for row in plan.parameter_rows()]),
        "",
    ]


def _render_results(result: BenchmarkResult) -> list[str]:
    """モデル × モード × 視力値 × 方向の正答率を組み立てる。"""
    plan = result.plan
    lines = ["## 2. 結果（モデル × モード × 視力値 × 開いている向き）", ""]

    for model in plan.models:
        lines += [f"### {model}", ""]
        for mode in plan.modes:
            header = ["視力", *[direction_label(d) for d in plan.directions], "合計"]
            rows: list[list[str]] = []
            for vision in plan.visions:
                cells = [
                    format_tally(
                        result.tally(model=model, mode=mode, vision=vision, direction=direction)
                    )
                    for direction in plan.directions
                ]
                rows.append(
                    [
                        f"{vision:g}",
                        *cells,
                        format_tally(result.tally(model=model, mode=mode, vision=vision)),
                    ]
                )
            rows.append(
                [
                    "**合計**",
                    *[
                        format_tally(result.tally(model=model, mode=mode, direction=direction))
                        for direction in plan.directions
                    ],
                    format_tally(result.tally(model=model, mode=mode)),
                ]
            )
            lines += [f"#### {mode.value}", ""]
            lines += _table(header, rows)
            lines += [""]
    return lines


def _render_summary(result: BenchmarkResult) -> list[str]:
    """モデル別・モード別・視力値別のまとめを組み立てる。"""
    plan = result.plan
    lines = ["## 3. まとめ", "", "### モデル × モード", ""]
    lines += _table(
        ["モデル", *[mode.value for mode in plan.modes], "合計"],
        [
            [
                model,
                *[format_tally(result.tally(model=model, mode=mode)) for mode in plan.modes],
                format_tally(result.tally(model=model)),
            ]
            for model in plan.models
        ],
    )

    lines += ["", "### モデル × 視力値", ""]
    lines += _table(
        ["視力", *plan.models],
        [
            [
                f"{vision:g}",
                *[format_tally(result.tally(model=model, vision=vision)) for model in plan.models],
            ]
            for vision in plan.visions
        ],
    )

    lines += ["", "### 応答時間（回答が得られた呈示のみ）", ""]
    lines += _table(
        ["モデル", "件数", "平均", "中央値", "最大"],
        [[model, *_response_time_cells(result, model)] for model in plan.models],
    )

    lines += ["", "### 回答が得られなかった呈示", ""]
    failed_rows = _failed_rows(result.failed_attempts())
    lines += (
        _table(["モデル", "モード", "視力", "方向", "件数"], failed_rows)
        if failed_rows
        else ["なし"]
    )
    lines += [""]
    return lines


def _response_time_cells(result: BenchmarkResult, model: str) -> list[str]:
    """応答時間の統計（件数・平均・中央値・最大）を返す。"""
    times = result.response_times(model=model)
    if not times:
        return ["0", NO_VALUE, NO_VALUE, NO_VALUE]
    return [
        str(len(times)),
        f"{statistics.mean(times):.2f} 秒",
        f"{statistics.median(times):.2f} 秒",
        f"{max(times):.2f} 秒",
    ]


def _failed_rows(failed: list[Attempt]) -> list[list[str]]:
    """回答が得られなかった呈示を、条件ごとにまとめる。"""
    counts: dict[tuple[str, str, float, str], int] = {}
    for attempt in failed:
        presentation = attempt.presentation
        key = (
            presentation.model,
            presentation.mode.value,
            presentation.vision,
            presentation.direction,
        )
        counts[key] = counts.get(key, 0) + 1

    return [
        [model, mode, f"{vision:g}", direction_label(direction), str(count)]
        for (model, mode, vision, direction), count in sorted(counts.items())
    ]


def _answered_labels(result: BenchmarkResult) -> list[str]:
    """実際に観測された回答キーを、件数の多い順に返す。"""
    counts: Counter = Counter(
        attempt.answer for attempt in result.attempts if attempt.answer is not None
    )
    return [label for label, _ in counts.most_common()]


def _max_probability(attempt: Attempt) -> float:
    """回答の確信度（API が返した確率の最大値）を返す。

    確率が記録されていない呈示は、どの帯にも入らないよう負の値を返す。

    Args:
        attempt: 呈示の結果。

    Returns:
        確率の最大値（確率が無ければ ``-1.0``）。
    """
    return max(attempt.probabilities.values(), default=-1.0)


def _format_confidence(attempts: list[Attempt], low: float, high: float) -> str:
    """確信度が ``[low, high)`` の呈示だけを集計して ``3/4 (75%)`` の形にする。

    Args:
        attempts: 対象モデルの呈示すべて。
        low: 帯の下限。
        high: 帯の上限。

    Returns:
        表示用の文字列。該当する呈示が無ければ ``—``。
    """
    selected = [a for a in attempts if low <= _max_probability(a) < high]
    if not selected:
        return NO_VALUE
    correct = sum(1 for a in selected if a.is_correct)
    return f"{correct}/{len(selected)} ({correct / len(selected):.0%})"


def _render_diagnostics(result: BenchmarkResult) -> list[str]:
    """回答の偏り（回答ラベルの分布と混同行列）を組み立てる。"""
    labels = _answered_labels(result)
    if not labels:
        return []

    lines = ["## 4. 回答の偏り（診断用）", ""]
    lines += _render_label_distribution(result, labels)
    lines += _render_confusion_matrix(result)
    lines += _render_confidence_table(result)
    return lines


def _render_label_distribution(result: BenchmarkResult, labels: list[str]) -> list[str]:
    """回答ラベルの分布と、回答キーごとの平均確率を組み立てる。

    Args:
        result: 実行結果。
        labels: 回答ラベルの一覧（:func:`_answered_labels`）。

    Returns:
        Markdown の行（見出しを含む）。
    """
    plan = result.plan
    lines = ["### 回答ラベルの分布", ""]
    rows = []
    probability_rows = []
    for model in plan.models:
        attempts = [a for a in result.attempts if a.presentation.model == model]
        counts: Counter = Counter(a.answer for a in attempts if a.answer is not None)
        failed = sum(1 for a in attempts if a.failed)
        total = len(attempts)
        cells = [
            f"{counts[label]} ({counts[label] / total:.0%})" if total else NO_VALUE
            for label in labels
        ]
        rows.append([model, *cells, str(failed)])
        probability_rows.append(
            [
                model,
                *[
                    f"{statistics.mean(a.probabilities.get(label, 0.0) for a in attempts):.3f}"
                    if attempts
                    else NO_VALUE
                    for label in labels
                ],
            ]
        )

    lines += _table(["モデル", *labels, "失敗"], rows)
    lines += ["", "回答キーごとの平均確率（API が返した確率の平均）", ""]
    lines += _table(["モデル", *labels], probability_rows)
    return lines


def _render_confusion_matrix(result: BenchmarkResult) -> list[str]:
    """混同行列（提示した向き × 回答した向き）を組み立てる。"""
    plan = result.plan
    lines = ["", "### 混同行列（提示した向き × 回答した向き）", ""]
    directions = list(plan.directions)
    header = ["モデル", "提示", *[direction_label(d) for d in directions], "未対応", "失敗"]
    confusion_rows = []
    for model in plan.models:
        for presented in directions:
            selected = [
                a
                for a in result.attempts
                if a.presentation.model == model and a.presentation.direction == presented
            ]
            if not selected:
                continue
            counts = Counter(a.answered_direction for a in selected if not a.failed)
            unmapped = sum(
                1 for a in selected if a.answer is not None and a.answered_direction is None
            )
            failed = sum(1 for a in selected if a.failed)
            confusion_rows.append(
                [
                    model,
                    direction_label(presented),
                    *[str(counts.get(direction, 0)) for direction in directions],
                    str(unmapped),
                    str(failed),
                ]
            )
    lines += _table(header, confusion_rows)
    return lines


def _render_confidence_table(result: BenchmarkResult) -> list[str]:
    """確信度（最大確率）の帯ごとの正答率の表を組み立てる。"""
    lines = ["", "### 確信度（最大確率）と正答率", ""]
    confidence_rows = []
    for model in result.plan.models:
        attempts = [a for a in result.attempts if a.presentation.model == model]
        confidence_rows.append(
            [
                model,
                *[_format_confidence(attempts, low, high) for low, high, _ in CONFIDENCE_BUCKETS],
            ]
        )
    lines += _table(["モデル", *[label for _, _, label in CONFIDENCE_BUCKETS]], confidence_rows)
    lines += [
        "",
        "最大確率の帯ごとの正答率。帯によって正答率が大きく変わるなら、最大確率を"
        "しきい値にして低確信の呈示を「判定不能」として切り分けられる"
        "（見送りにする・別の判定手段に回すなど）。",
        "",
    ]
    return lines


def _render_notes() -> list[str]:
    """補足（読み方と実行条件の注意）を組み立てる。"""
    return [
        "## 5. 補足",
        "",
        "* 正答率は「呈示数に対する正答」で、**回答が得られなかった呈示も分母に含む**"
        "（どの条件で回答が得られなかったかは「回答が得られなかった呈示」の表を参照）。",
        "* 同じ条件の反復は**同一の画像**を呈示している。応答が決定的なモデルでは毎回同じ回答に"
        "なるため、反復は再現性の確認になる（回答のばらつきは測れない）。",
        "* **方向ごとの正答率が偏るときは「4. 回答の偏り」を先に見ること。** 判定できないときに"
        "特定の回答ラベルへ落ちているだけ（回答ラベルの事前分布）の可能性がある。",
        "* GUI 版と違い、正否に応じた視力値の段階的な変更や、判定ごとの待ち時間は入れていない。",
        "* 画像は GUI 版と同じ生成処理（`domain/landolt.py`）を使っている。",
        f"* パラメータが `{NO_VALUE}` のモードは固有の設定値を持たない（清明）。",
        "",
    ]


def raw_data(result: BenchmarkResult) -> dict[str, Any]:
    """生データ（JSON 用の辞書）を組み立てる。

    Args:
        result: 実行結果。

    Returns:
        実行条件と呈示ごとの結果。
    """
    plan = result.plan
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "endpoint": result.endpoint,
        "question": result.question_name,
        "started_at": result.started_at.isoformat(timespec="seconds"),
        "finished_at": result.finished_at.isoformat(timespec="seconds"),
        "connection_checks": dict(result.connection_checks),
        "plan": {
            "models": list(plan.models),
            "modes": [mode.value for mode in plan.modes],
            "visions": list(plan.visions),
            "directions": list(plan.directions),
            "repeats": plan.repeats,
            "total": plan.total,
            "parameters": {mode.value: plan.parameters(mode) for mode in plan.modes},
        },
        "attempts": [attempt.as_dict() for attempt in result.attempts],
    }


def write_markdown(result: BenchmarkResult, path: Path) -> Path:
    """レポートを Markdown ファイルに書き出す。

    Args:
        result: 実行結果。
        path: 出力先。

    Returns:
        書き出したパス。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result), encoding="utf-8")
    return path


def write_raw_json(result: BenchmarkResult, path: Path) -> Path:
    """生データを JSON ファイルに書き出す（後から再集計できるように）。

    Args:
        result: 実行結果。
        path: 出力先。

    Returns:
        書き出したパス。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(raw_data(result), ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def load_raw_json(path: Path) -> BenchmarkResult:
    """生データ（JSON）から実行結果を復元する（中断からの再開と再集計に使う）。

    Args:
        path: :func:`write_raw_json` で書き出したファイル。

    Returns:
        復元した実行結果。

    Raises:
        ValueError: 実行条件と呈示の記録が読み取れない場合。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        plan_data = data["plan"]
        plan = BenchmarkPlan(
            models=tuple(plan_data["models"]),
            modes=tuple(GenerationMode(name) for name in plan_data["modes"]),
            visions=tuple(float(vision) for vision in plan_data["visions"]),
            directions=cast("tuple[Direction, ...]", tuple(plan_data["directions"])),
            repeats=int(plan_data["repeats"]),
        )
        attempts = tuple(
            Attempt(
                presentation=Presentation(
                    model=row["model"],
                    mode=GenerationMode(row["mode"]),
                    vision=float(row["vision"]),
                    direction=row["direction"],
                    repeat=int(row["repeat"]) - 1,  # 保存時は 1 始まり
                ),
                answer=row["answer"],
                is_correct=row["is_correct"],
                elapsed_sec=float(row["elapsed_sec"]),
                probabilities=dict(row.get("probabilities") or {}),
                answered_direction=row.get("answered_direction", row["answer"]),
            )
            for row in data["attempts"]
        )
        return BenchmarkResult(
            plan=plan,
            attempts=attempts,
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(data["finished_at"]),
            connection_checks=dict(data.get("connection_checks") or {}),
            endpoint=data.get("endpoint", config.CLEF_BASE_URL),
            question_name=data.get("question", DIRECTION_QUESTION.name),
            resumed_from=int(data.get("resumed_from", 0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"生データを読み取れません: {path}: {error}") from error
