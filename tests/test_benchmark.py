"""統計バッチ（systemone_vision_test.benchmark）のテスト。

GUI も音声も使わず、画像生成と通信は差し替えて検証する（HTTP には出ない）。
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from systemone_vision_test import config
from systemone_vision_test.benchmark import cli
from systemone_vision_test.benchmark.plan import BenchmarkPlan, Presentation
from systemone_vision_test.benchmark.report import (
    DIRECTION_LABELS,
    format_duration,
    format_tally,
    load_raw_json,
    raw_data,
    render_markdown,
    write_markdown,
    write_raw_json,
)
from systemone_vision_test.benchmark.results import Attempt, BenchmarkResult, Tally
from systemone_vision_test.benchmark.runner import BenchmarkRunner
from systemone_vision_test.domain.landolt import (
    GenerationMode,
    LandoltCParams,
    default_parameters,
)
from systemone_vision_test.domain.questions import (
    DIRECTION_QUESTION,
    DIRECTION_QUESTION_KEY,
    DIRECTION_QUESTIONS,
    NEUTRAL_KEY_QUESTION,
    DirectionQuestion,
)
from systemone_vision_test.services.answers import DirectionAnswer
from systemone_vision_test.services.clef_client import ClefClient

# 検証に使う小さな計画（モデル 2 × モード 2 × 視力 2 × 方向 2 × 反復 2 = 32 件）
SMALL_PLAN = BenchmarkPlan(
    models=("clef", "clef-flash"),
    modes=(GenerationMode.CLEAR, GenerationMode.MYOPIA),
    visions=(1.0, 0.5),
    directions=("up", "down"),
    repeats=2,
)


class EchoClient:
    """payload に入っている方向をそのまま答えるクライアント（＝常に正解）。

    検証では画像の代わりに方向の文字列を payload にしているため、正誤を検証できる。
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[str] = []
        self.check_calls = 0

    def check_connection(self) -> bool:
        """接続できる。"""
        self.check_calls += 1
        return True

    def inspect_landolt_c(
        self,
        image_base64: str,
        *,
        question: DirectionQuestion = DIRECTION_QUESTION,
    ) -> DirectionAnswer | None:
        """payload の方向をそのまま答える。"""
        self.calls.append(image_base64)
        return DirectionAnswer(
            choice=image_base64, probabilities={image_base64: 1.0}, confidence=1.0
        )


class SilentClient(EchoClient):
    """常に回答しないクライアント（通信エラーの模擬）。"""

    def inspect_landolt_c(
        self,
        image_base64: str,
        *,
        question: DirectionQuestion = DIRECTION_QUESTION,
    ) -> DirectionAnswer | None:
        """回答を返さない。"""
        self.calls.append(image_base64)
        return None


class OfflineClient(EchoClient):
    """接続できないモデルの模擬。"""

    def check_connection(self) -> bool:
        """接続できない。"""
        self.check_calls += 1
        return False


def recording_tools() -> tuple[list[LandoltCParams], object, object]:
    """画像生成を記録する差し替え（``generate`` / ``encode``）を返す。

    ``encode`` は画像の代わりに方向の文字列を返すので、偽のクライアントは payload を
    そのまま答えることができる（＝正誤を検証できる）。

    Returns:
        ``(記録された spec の一覧, generate, encode)``。
    """
    specs: list[LandoltCParams] = []

    def generate(spec: LandoltCParams) -> LandoltCParams:
        specs.append(spec)
        return spec

    def encode(image: LandoltCParams) -> str:
        return image.direction

    return specs, generate, encode


def make_runner(
    plan: BenchmarkPlan = SMALL_PLAN,
    *,
    clients: dict[str, EchoClient] | None = None,
    **kwargs,
) -> tuple[BenchmarkRunner, list[LandoltCParams], dict[str, EchoClient]]:
    """差し替えた画像生成と偽のクライアントを使うランナーを作る。

    Returns:
        ``(ランナー, 生成された spec の一覧, モデル名ごとのクライアント)``。
    """
    specs, generate, encode = recording_tools()
    created: dict[str, EchoClient] = {} if clients is None else clients

    def factory(model: str) -> EchoClient:
        return created.setdefault(model, EchoClient(model))

    runner = BenchmarkRunner(
        plan, client_factory=factory, generate=generate, encode=encode, **kwargs
    )
    return runner, specs, created


def run_small(**kwargs) -> tuple[BenchmarkResult, list[LandoltCParams], dict[str, EchoClient]]:
    """小さい計画を実行する（レポートのテスト用）。

    Returns:
        ``(結果, 生成された spec の一覧, モデル名ごとのクライアント)``。
    """
    runner, specs, clients = make_runner(**kwargs)
    return runner.run(), specs, clients


def attempt(
    model: str = "clef",
    mode: GenerationMode = GenerationMode.CLEAR,
    vision: float = 1.0,
    direction: str = "up",
    repeat: int = 0,
    answer: str | None = "up",
) -> Attempt:
    """検証用の呈示結果を作る（``answer=None`` なら回答なし）。"""
    presentation = Presentation(model, mode, vision, direction, repeat)
    if answer is None:
        return Attempt(presentation=presentation, answer=None, is_correct=None, elapsed_sec=0.1)
    return Attempt(
        presentation=presentation,
        answer=answer,
        is_correct=answer == direction,
        elapsed_sec=0.1,
        probabilities={answer: 1.0},
    )


def result_of(attempts: list[Attempt], *, seconds: float = 90.0) -> BenchmarkResult:
    """検証用の実行結果を作る。"""
    started = datetime(2026, 10, 25, 12, 0, 0)
    return BenchmarkResult(
        plan=SMALL_PLAN,
        attempts=tuple(attempts),
        started_at=started,
        finished_at=started + timedelta(seconds=seconds),
    )


def attempts_for(
    plan: BenchmarkPlan,
    model: str | None = None,
    mode: GenerationMode | None = None,
) -> list[Attempt]:
    """指定した条件の呈示結果を（すべて正答で）作る。

    Args:
        plan: 対象の計画。
        model: 絞り込むモデル（``None`` なら全モデル）。
        mode: 絞り込むモード（``None`` なら全モード）。

    Returns:
        呈示の並びに対応する結果。
    """
    return [
        attempt(
            model=presentation.model,
            mode=presentation.mode,
            vision=presentation.vision,
            direction=presentation.direction,
            repeat=presentation.repeat,
        )
        for presentation in plan.presentations()
        if (model is None or presentation.model == model)
        and (mode is None or presentation.mode is mode)
    ]


# ---------------------------------------------------------------------------
# 集計（results）
# ---------------------------------------------------------------------------


def test_tally_computes_accuracy_and_answer_rate():
    """正答率と回答率を計算する。"""
    tally = Tally(correct=3, wrong=1, failed=1)

    assert tally.total == 5
    assert tally.answered == 4
    assert tally.accuracy == pytest.approx(3 / 5)
    assert tally.answer_rate == pytest.approx(4 / 5)


def test_tally_is_empty_without_attempts():
    """呈示が無ければ率は ``None`` になる。"""
    empty = Tally()

    assert empty.total == 0
    assert empty.accuracy is None
    assert empty.answer_rate is None


def test_tally_adds_up_and_counts_failures():
    """集計は足し算でき、回答なしは失敗として数える。"""
    tally = Tally.of_attempts([attempt(), attempt(answer="down"), attempt(answer=None)])

    assert tally == Tally(correct=1, wrong=1, failed=1)
    assert tally + Tally(correct=1) == Tally(correct=2, wrong=1, failed=1)


def test_result_tally_filters_by_condition():
    """モデル・モード・視力値・方向で絞り込んで集計できる。"""
    result = result_of(
        [
            attempt(mode=GenerationMode.CLEAR, vision=1.0, direction="up"),
            attempt(mode=GenerationMode.CLEAR, vision=1.0, direction="up"),
            attempt(
                model="clef-flash",
                mode=GenerationMode.MYOPIA,
                vision=0.5,
                direction="down",
                answer="up",
            ),
        ]
    )

    assert result.tally(model="clef") == Tally(correct=2)
    assert result.tally(model="clef", vision=1.0, direction="up") == Tally(correct=2)
    assert result.tally(model="clef-flash") == Tally(wrong=1)
    assert result.tally(mode=GenerationMode.CATARACT) == Tally()


def test_result_duration_response_times_and_failures():
    """所要時間・応答時間・回答が得られなかった呈示を扱える。"""
    result = result_of(
        [attempt(), attempt(direction="down", answer=None), attempt(model="clef-flash")],
        seconds=30,
    )

    assert result.duration_sec == pytest.approx(30.0)
    assert result.response_times() == [0.1, 0.1]
    assert result.response_times(model="clef") == [0.1]
    assert len(result.failed_attempts()) == 1


# ---------------------------------------------------------------------------
# 実行（runner）
# ---------------------------------------------------------------------------


def test_runner_presents_every_combination_with_expected_specs():
    """計画どおりの条件（モード・視力値・方向・パラメータ）で画像を作って呈示する。"""
    result, specs, _ = run_small()

    assert len(result.attempts) == SMALL_PLAN.total
    assert [spec.vision for spec in specs] == [p.vision for p in SMALL_PLAN.presentations()]
    assert [spec.direction for spec in specs] == [p.direction for p in SMALL_PLAN.presentations()]
    assert [spec.mode for spec in specs] == [p.mode for p in SMALL_PLAN.presentations()]

    sizes = [spec.gaussian_ksize for spec in specs if spec.mode is GenerationMode.MYOPIA]
    assert sizes == [config.MYOPIA_GAUSSIAN_KSIZE_DEFAULT] * len(sizes)  # 既定パラメータ


def test_runner_counts_correct_answers():
    """正答を条件ごとに数える（この検証では常に正解が返る）。"""
    result, _, _ = run_small()

    tally = result.tally(model="clef", mode=GenerationMode.CLEAR, vision=1.0, direction="up")
    assert tally == Tally(correct=2)
    assert result.tally().accuracy == 1.0
    assert result.tally().failed == 0


def test_runner_uses_one_client_per_model():
    """モデルごとにクライアントを 1 つ作って使い回す（接続を張り直さない）。"""
    _, _, clients = run_small()

    assert set(clients) == set(SMALL_PLAN.models)
    per_model = SMALL_PLAN.total // len(SMALL_PLAN.models)
    for client in clients.values():
        assert len(client.calls) == per_model
        assert client.check_calls == 1  # 接続確認は実行前の 1 回だけ


def test_runner_records_missing_answer_as_failure():
    """回答が得られなかった呈示は失敗として記録する。"""
    clients = {model: SilentClient(model) for model in SMALL_PLAN.models}
    result, _, _ = run_small(clients=clients)

    assert result.tally().failed == SMALL_PLAN.total
    assert result.tally().accuracy == 0.0
    assert len(result.failed_attempts()) == SMALL_PLAN.total
    assert all(item.answer is None for item in result.attempts)


def test_runner_stops_when_a_model_is_unreachable():
    """接続できないモデルがあれば、実行を始めずにエラーにする。"""
    clients = {model: OfflineClient(model) for model in SMALL_PLAN.models}
    runner, specs, _ = make_runner(clients=clients)

    with pytest.raises(ConnectionError):
        runner.run()

    assert specs == []  # 画像は 1 枚も作っていない
    assert all(client.calls == [] for client in clients.values())


def test_runner_reports_progress():
    """進捗をコールバックに伝える（まとまりごとの結果と総数）。"""
    messages: list[str] = []
    run_small(progress=messages.append)

    assert "呈示 32 件のうち 32 件を実行します" in messages
    assert any(message.startswith("[clef] 清明 完了: 8/8 (100%)") for message in messages)
    assert any(message.startswith("完了: 32 件") for message in messages)


def test_runner_uses_the_real_image_pipeline():
    """既定の画像生成・エンコードでは 640×640 の PNG を送る。"""
    plan = BenchmarkPlan(
        models=("clef",),
        modes=(GenerationMode.CLEAR,),
        visions=(1.0,),
        directions=("up",),
        repeats=1,
    )
    clients: dict[str, EchoClient] = {}
    runner = BenchmarkRunner(
        plan, client_factory=lambda model: clients.setdefault(model, EchoClient(model))
    )

    result = runner.run()

    payload = clients["clef"].calls[0]
    image = cv2.imdecode(np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (config.CANVAS_SIZE, config.CANVAS_SIZE)
    assert result.attempts[0].failed is False  # 回答は返っている


# ---------------------------------------------------------------------------
# レポート（report）
# ---------------------------------------------------------------------------


def test_report_has_preconditions_and_result_tables():
    """レポートに前提条件（パラメータ）と結果の表が入る。"""
    result, _, _ = run_small()
    markdown = render_markdown(result)

    directions = " | ".join(DIRECTION_LABELS[d] for d in ("up", "down"))

    assert markdown.startswith("# clef / clef-flash 全モード統計レポート")
    assert "## 1. 前提条件" in markdown
    assert "### 実行環境" in markdown
    assert "### モードごとのパラメータ（各モードの既定値）" in markdown
    assert "| 近視 | ガウシアンブラー カーネルサイズ |" in markdown
    assert f"| {config.MYOPIA_GAUSSIAN_KSIZE_DEFAULT} |" in markdown
    assert "## 2. 結果（モデル × モード × 視力値 × 開いている向き）" in markdown
    assert "#### 清明" in markdown
    assert f"| 視力 | {directions} | 合計 |" in markdown
    assert "| 2/2 (100%) |" in markdown
    assert "## 3. まとめ" in markdown
    assert "### モデル × モード" in markdown
    assert "### 応答時間（回答が得られた呈示のみ）" in markdown
    assert "## 4. 回答の偏り（診断用）" in markdown
    assert "### 混同行列（提示した向き × 回答した向き）" in markdown
    assert "## 5. 補足" in markdown
    assert f"| 質問 | {DIRECTION_QUESTION.name} |" in markdown


def test_report_lists_failures():
    """回答が得られなかった呈示は、条件ごとに表へ出る。"""
    clients = {model: SilentClient(model) for model in SMALL_PLAN.models}
    result, _, _ = run_small(clients=clients)
    markdown = render_markdown(result)

    assert "### 回答が得られなかった呈示" in markdown
    assert "| clef | 清明 | 1 | 上 | 2 |" in markdown
    assert "0/8 (0%)" in markdown  # モデル × モードの正答率


def test_report_handles_an_empty_result():
    """呈示が無くてもレポートが壊れない。"""
    markdown = render_markdown(result_of([]))

    assert "| — |" in markdown
    assert "なし" in markdown


def test_report_formats_tally_and_duration():
    """集計と時間の表示を整える。"""
    assert format_tally(Tally()) == "—"
    assert format_tally(Tally(correct=3, wrong=1)) == "3/4 (75%)"
    assert format_duration(1034) == "17 分 14 秒"
    assert format_duration(45) == "45 秒"
    assert format_duration(3725) == "1 時間 2 分 5 秒"


def test_report_writes_markdown_and_raw_json(tmp_path):
    """レポートと生データを書き出せる（生データは後から再集計できる形）。"""
    result, _, _ = run_small()

    report = write_markdown(result, tmp_path / "reports" / "benchmark.md")
    raw = write_raw_json(result, tmp_path / "reports" / "benchmark.json")
    data = json.loads(raw.read_text(encoding="utf-8"))

    assert report.read_text(encoding="utf-8").startswith("# clef")
    assert data["plan"]["total"] == SMALL_PLAN.total
    assert data["plan"]["repeats"] == SMALL_PLAN.repeats
    assert data["plan"]["parameters"][GenerationMode.MYOPIA.value] == default_parameters(
        GenerationMode.MYOPIA
    )
    assert len(data["attempts"]) == SMALL_PLAN.total
    assert set(data["attempts"][0]) == {
        "model",
        "mode",
        "vision",
        "direction",
        "repeat",
        "answer",
        "answered_direction",
        "is_correct",
        "elapsed_sec",
        "probabilities",
    }
    assert data["question"] == result.question_name
    assert raw_data(result)["connection_checks"] == result.connection_checks


# ---------------------------------------------------------------------------
# コマンドライン（cli）
# ---------------------------------------------------------------------------


class StubRunner:
    """``BenchmarkRunner`` の代わり（実行せずに用意した結果を返す）。"""

    def __init__(self, result: BenchmarkResult) -> None:
        self._result = result

    def run(self) -> BenchmarkResult:
        """実行せずに結果を返す。"""
        return self._result


class FailingRunner:
    """接続できないときの ``BenchmarkRunner`` の代わり。"""

    def run(self) -> BenchmarkResult:
        """接続エラーを投げる。"""
        raise ConnectionError("clef API に接続できないモデルがあります: clef")


def test_cli_builds_plan_from_arguments():
    """引数から実行計画を組み立てる（部分実行もできる）。"""
    args = cli.parse_args(
        ["--models", "clef", "--modes", "清明", "--visions", "1.0", "--repeats", "1"]
    )

    plan = cli.build_plan(args)

    assert plan.models == ("clef",)
    assert plan.modes == (GenerationMode.CLEAR,)
    assert plan.visions == (1.0,)
    assert plan.repeats == 1
    assert plan.total == len(config.DIRECTIONS)  # 方向 4 つ × 1 回


def test_cli_uses_defaults_for_the_full_run():
    """引数なしなら 3 モデル × 4 モード × 13 視力値 × 4 方向 × 4 回 = 2,496 件になる。"""
    plan = cli.build_plan(cli.parse_args([]))

    assert plan.models == ("clef", "clef-flash", "jev")
    assert plan.total == 3 * 4 * len(config.VISION_LEVELS) * 4 * 4
    assert plan.total == 2496


def test_cli_rejects_invalid_repeats():
    """反復回数は 1 以上でなければ受け付けない。"""
    with pytest.raises(SystemExit):
        cli.parse_args(["--repeats", "0"])


def test_cli_resolves_output_paths(tmp_path):
    """出力先を決める（ディレクトリ指定なら既定名、``--no-raw`` なら生データ無し）。"""
    directory = cli.resolve_paths(cli.parse_args(["--output", str(tmp_path)]), "20261025_120000")

    assert directory == (
        tmp_path / "benchmark_20261025_120000.md",
        tmp_path / "benchmark_20261025_120000.json",
    )

    named = cli.resolve_paths(
        cli.parse_args(["--output", str(tmp_path / "mine.md"), "--no-raw"]), "20261025_120000"
    )
    assert named == (tmp_path / "mine.md", None)


def test_cli_writes_report_and_raw_data(tmp_path, monkeypatch):
    """実行結果をレポートと生データに書き出す。"""
    result, _, _ = run_small()
    monkeypatch.setattr(cli, "BenchmarkRunner", lambda *args, **kwargs: StubRunner(result))

    exit_code = cli.main(
        [
            "--models",
            "clef",
            "--modes",
            "清明",
            "--visions",
            "1.0",
            "--repeats",
            "1",
            "--output",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    reports = sorted(tmp_path.glob("benchmark_*.md"))
    assert len(reports) == 1
    assert reports[0].read_text(encoding="utf-8").startswith("# clef")
    assert (tmp_path / f"{reports[0].stem}.json").is_file()


def test_cli_reports_connection_failure(tmp_path, monkeypatch, capsys):
    """接続できない場合は 1 を返し、ファイルを書き出さない。"""
    monkeypatch.setattr(cli, "BenchmarkRunner", lambda *args, **kwargs: FailingRunner())

    exit_code = cli.main(["--output", str(tmp_path), "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "エラー" in captured.out
    assert "Ollama" in captured.out
    assert list(tmp_path.glob("*.md")) == []


# ---------------------------------------------------------------------------
# 途中経過の保存と再開（resume）
# ---------------------------------------------------------------------------

#: SMALL_PLAN と同じ条件を CLI 引数で表したもの（再開の検証に使う）
SMALL_PLAN_ARGS = [
    "--models",
    "clef",
    "clef-flash",
    "--modes",
    "清明",
    "近視",
    "--visions",
    "1.0",
    "0.5",
    "--directions",
    "up",
    "down",
    "--repeats",
    "2",
]


def test_cli_arguments_can_express_small_plan():
    """検証用の計画は CLI の引数でも表現できる（前提の確認）。"""
    assert cli.build_plan(cli.parse_args(SMALL_PLAN_ARGS)) == SMALL_PLAN


def test_result_detects_only_completed_groups():
    """呈示が揃ったまとまりだけを記録済みとみなし、揃っていない記録は捨てる。"""
    complete = attempts_for(SMALL_PLAN, "clef", GenerationMode.CLEAR)
    partial = attempts_for(SMALL_PLAN, "clef", GenerationMode.MYOPIA)[:3]
    result = result_of([*complete, *partial])

    assert result.completed_group_keys() == frozenset({("clef", GenerationMode.CLEAR)})

    trimmed = result.keep_complete_groups()

    assert len(trimmed.attempts) == len(complete)
    assert all(item.presentation.mode is GenerationMode.CLEAR for item in trimmed.attempts)


def test_runner_saves_a_checkpoint_after_each_group():
    """まとまり（モデル × モード）ごとに途中経過を保存する。"""
    checkpoints: list[BenchmarkResult] = []
    result, _, _ = run_small(checkpoint=checkpoints.append)

    assert len(checkpoints) == 4  # 2 モデル × 2 モード
    assert [len(checkpoint.attempts) for checkpoint in checkpoints] == [8, 16, 24, 32]
    assert len(result.attempts) == SMALL_PLAN.total
    assert result.resumed_from == 0


def test_runner_resumes_from_completed_groups():
    """記録済みのまとまりは呈示し直さず、残りだけを実行する。"""
    done = attempts_for(SMALL_PLAN, "clef", GenerationMode.CLEAR)
    messages: list[str] = []
    result, specs, _ = run_small(resume_from=result_of(done), progress=messages.append)

    assert len(result.attempts) == SMALL_PLAN.total  # 重複しない
    assert len(specs) == SMALL_PLAN.total - len(done)  # 残りだけ画像を作る
    assert result.resumed_from == len(done)
    assert any("記録済みのためスキップ" in message for message in messages)


def test_runner_does_not_present_for_recorded_models():
    """すべてのまとまりが記録済みのモデルには呈示しない（接続確認だけ行う）。"""
    previous = result_of(attempts_for(SMALL_PLAN, model="clef"))
    result, specs, clients = run_small(resume_from=previous)

    assert clients["clef"].calls == []  # 呈示し直さない
    assert len(clients["clef-flash"].calls) == SMALL_PLAN.total // 2
    assert len(specs) == SMALL_PLAN.total // 2
    assert len(result.attempts) == SMALL_PLAN.total


def test_report_and_raw_data_round_trip(tmp_path):
    """生データを書き出して読み戻せる（中断からの再開と再集計に使う）。"""
    result, _, _ = run_small()
    path = write_raw_json(result, tmp_path / "raw.json")

    loaded = load_raw_json(path)

    assert loaded.plan == result.plan
    assert [item.presentation for item in loaded.attempts] == [
        item.presentation for item in result.attempts
    ]
    assert [item.answer for item in loaded.attempts] == [item.answer for item in result.attempts]
    assert [item.is_correct for item in loaded.attempts] == [
        item.is_correct for item in result.attempts
    ]
    assert loaded.attempts[0].elapsed_sec == pytest.approx(result.attempts[0].elapsed_sec, abs=1e-3)
    # JSON には秒単位で保存するため、マイクロ秒は落ちる
    assert loaded.started_at.replace(microsecond=0) == result.started_at.replace(microsecond=0)
    assert loaded.finished_at.replace(microsecond=0) == result.finished_at.replace(microsecond=0)
    assert loaded.connection_checks == result.connection_checks


def test_load_raw_json_rejects_broken_data(tmp_path):
    """壊れた生データは分かる形で失敗する。"""
    path = tmp_path / "broken.json"
    path.write_text('{"plan": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="生データを読み取れません"):
        load_raw_json(path)


def test_report_notes_resumed_runs():
    """再開した実行はレポートに分かるように出す。"""
    previous = result_of(attempts_for(SMALL_PLAN, model="clef"))
    result, _, _ = run_small(resume_from=previous)

    assert f"| 中断からの再開 | あり（前回から {len(previous.attempts)} 件を引き継ぎ） |" in (
        render_markdown(result)
    )


def test_cli_uses_default_state_path():
    """途中経過の既定の保存先は artifacts/benchmark の下。"""
    assert cli.parse_args([]).state == config.BENCHMARK_OUTPUT_DIR / "benchmark.state.json"


def test_cli_resumes_from_state_and_saves_checkpoints(tmp_path, monkeypatch):
    """途中経過があれば引き継ぎ、保存用のコールバックも渡す。"""
    state = tmp_path / "state.json"
    previous = result_of(attempts_for(SMALL_PLAN, model="clef"))
    write_raw_json(previous, state)
    result, _, _ = run_small()
    captured: dict = {}

    def make_runner(*args, **kwargs):
        captured.update(kwargs)
        return StubRunner(result)

    monkeypatch.setattr(cli, "BenchmarkRunner", make_runner)

    exit_code = cli.main(
        [*SMALL_PLAN_ARGS, "--state", str(state), "--output", str(tmp_path), "--no-raw", "--quiet"]
    )

    assert exit_code == 0
    assert captured["resume_from"] is not None
    assert len(captured["resume_from"].attempts) == len(previous.attempts)
    assert captured["checkpoint"] is not None


def test_cli_removes_state_after_completion(tmp_path, monkeypatch):
    """完走したら途中経過を片付ける（次回は最初から実行できる）。"""
    state = tmp_path / "state.json"
    result, _, _ = run_small()
    write_raw_json(result_of(attempts_for(SMALL_PLAN, model="clef")), state)
    monkeypatch.setattr(cli, "BenchmarkRunner", lambda *args, **kwargs: StubRunner(result))

    exit_code = cli.main(
        [*SMALL_PLAN_ARGS, "--state", str(state), "--output", str(tmp_path), "--no-raw", "--quiet"]
    )

    assert exit_code == 0
    assert not state.exists()


def test_cli_fresh_ignores_existing_state(tmp_path, monkeypatch):
    """``--fresh`` なら既存の途中経過を無視する。"""
    state = tmp_path / "state.json"
    result, _, _ = run_small()
    write_raw_json(result, state)
    captured: dict = {}

    def make_runner(*args, **kwargs):
        captured.update(kwargs)
        return StubRunner(result)

    monkeypatch.setattr(cli, "BenchmarkRunner", make_runner)

    exit_code = cli.main(
        [
            *SMALL_PLAN_ARGS,
            "--state",
            str(state),
            "--output",
            str(tmp_path),
            "--no-raw",
            "--fresh",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert captured["resume_from"] is None


def test_cli_refuses_resume_with_different_plan(tmp_path, monkeypatch, capsys):
    """実行条件が違う途中経過では再開せず、分かる形で止まる。"""
    state = tmp_path / "state.json"
    write_raw_json(result_of(attempts_for(SMALL_PLAN, model="clef")), state)
    monkeypatch.setattr(cli, "BenchmarkRunner", lambda *args, **kwargs: StubRunner(result_of([])))

    exit_code = cli.main(["--state", str(state), "--output", str(tmp_path), "--no-raw", "--quiet"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--fresh" in captured.out


def test_cli_no_state_disables_checkpoints(tmp_path, monkeypatch):
    """``--no-state`` なら保存も再開もしない。"""
    result, _, _ = run_small()
    captured: dict = {}

    def make_runner(*args, **kwargs):
        captured.update(kwargs)
        return StubRunner(result)

    monkeypatch.setattr(cli, "BenchmarkRunner", make_runner)

    exit_code = cli.main(
        [*SMALL_PLAN_ARGS, "--output", str(tmp_path), "--no-raw", "--no-state", "--quiet"]
    )

    assert exit_code == 0
    assert captured["checkpoint"] is None
    assert captured["resume_from"] is None


# ---------------------------------------------------------------------------
# 質問のバリエーション（回答ラベルの事前分布の切り分け）
# ---------------------------------------------------------------------------


class KeyClient(EchoClient):
    """質問の回答キーで答えるクライアント。

    Args:
        model: モデル名。
        question: 使う質問（キーと方向の対応）。
        key: 常に返す回答キー。``None`` なら提示された方向のキーを返す（＝常に正解）。
    """

    def __init__(self, model: str, question: DirectionQuestion, key: str | None = None) -> None:
        super().__init__(model)
        self.question = question
        self.key = key
        self.by_direction = {direction: answer for answer, direction in question.labels.items()}

    def inspect_landolt_c(
        self,
        image_base64: str,
        *,
        question: DirectionQuestion = DIRECTION_QUESTION,
    ) -> DirectionAnswer | None:
        """固定の回答キー、または提示された方向のキーを返す。"""
        self.calls.append(image_base64)
        answer = self.key if self.key is not None else self.by_direction[image_base64]
        return DirectionAnswer(choice=answer, probabilities={answer: 1.0}, confidence=1.0)


def test_question_variants_map_their_keys_to_directions():
    """どの質問でも、回答キーを方向に変換できる（採点の前提）。"""
    for name, question in DIRECTION_QUESTIONS.items():
        assert set(question.labels.values()) == set(config.DIRECTIONS), name
        assert len(question.labels) == len(config.DIRECTIONS), name
        for answer, direction in question.labels.items():
            assert question.direction_of(answer) == direction
        assert question.direction_of("unknown") is None


def test_swapped_question_describes_the_opposite_direction():
    """説明を入れ替えた質問では、キーと説明の向きが逆になり、採点も説明側に従う。"""
    from systemone_vision_test.domain.questions import SWAPPED_DESCRIPTION_QUESTION

    assert SWAPPED_DESCRIPTION_QUESTION.criteria["down"] == "上の部分が開いている"
    assert SWAPPED_DESCRIPTION_QUESTION.direction_of("down") == "up"
    assert SWAPPED_DESCRIPTION_QUESTION.direction_of("up") == "down"
    assert SWAPPED_DESCRIPTION_QUESTION.direction_of("left") == "right"


def test_closed_side_question_asks_for_the_connected_side():
    """反対質問は「つながっている側」を尋ね、採点は切れ目の向きに直して行う。"""
    from systemone_vision_test.domain.questions import CLOSED_SIDE_QUESTION

    assert CLOSED_SIDE_QUESTION.criteria["down"] == "下側がつながっている"
    assert "つながっている側" in CLOSED_SIDE_QUESTION.instruction
    # 「下がつながっている」＝「切れ目は上」なので、採点では up として扱う
    assert CLOSED_SIDE_QUESTION.direction_of("down") == "up"
    assert CLOSED_SIDE_QUESTION.direction_of("up") == "down"


def test_alternative_key_questions_keep_the_same_meaning():
    """別語彙・かなのキーでも、意味は同じ 4 方向に対応する。"""
    from systemone_vision_test.domain.questions import (
        COMPASS_KEY_QUESTION,
        HIRAGANA_KEY_QUESTION,
    )

    assert COMPASS_KEY_QUESTION.direction_of("west") == "left"
    assert COMPASS_KEY_QUESTION.direction_of("north") == "up"
    assert HIRAGANA_KEY_QUESTION.direction_of("ひだり") == "left"
    assert HIRAGANA_KEY_QUESTION.direction_of("みぎ") == "right"


def test_compass_shift_keeps_east_position_but_changes_the_criterion():
    """別語彙＋説明ずらしでは ``east`` の位置を保ったまま説明（＝意味）が変わる。"""
    from systemone_vision_test.domain.questions import (
        COMPASS_KEY_QUESTION,
        COMPASS_SHIFTED_QUESTION,
    )

    assert list(COMPASS_SHIFTED_QUESTION.criteria) == list(COMPASS_KEY_QUESTION.criteria)
    assert COMPASS_SHIFTED_QUESTION.criteria["east"] == "上の部分が開いている"
    assert COMPASS_SHIFTED_QUESTION.criteria["west"] == "下の部分が開いている"
    assert COMPASS_SHIFTED_QUESTION.direction_of("east") == "up"
    assert COMPASS_SHIFTED_QUESTION.direction_of("south") == "right"


def test_compass_no_gloss_only_drops_the_hint_sentence():
    """補足なしの変種は、キーと説明が ``compass`` と同一で質問文だけが短い。"""
    from systemone_vision_test.domain.questions import (
        COMPASS_KEY_QUESTION,
        COMPASS_NO_GLOSS_QUESTION,
    )

    assert COMPASS_NO_GLOSS_QUESTION.criteria == COMPASS_KEY_QUESTION.criteria
    assert COMPASS_NO_GLOSS_QUESTION.labels == COMPASS_KEY_QUESTION.labels
    assert "（上）" not in COMPASS_NO_GLOSS_QUESTION.instruction
    assert "（上）" in COMPASS_KEY_QUESTION.instruction


def test_neutral_question_payload_uses_neutral_keys():
    """中立キーの質問では、選択肢のキーが a/b/c/d になり順序も保たれる。"""
    payload = ClefClient().inspection_payload("BASE64", question=NEUTRAL_KEY_QUESTION)
    question = payload["questions"][DIRECTION_QUESTION_KEY]

    assert list(question["criteria"]) == ["a", "b", "c", "d"]
    assert "a（上）" in question["instructions"]
    assert question["criteria"]["b"] == "下の部分が開いている"


def test_runner_scores_answers_through_the_question_variant():
    """回答キーが変わっても、提示した方向と照合して採点する。"""
    clients = {model: KeyClient(model, NEUTRAL_KEY_QUESTION) for model in SMALL_PLAN.models}

    result, _, _ = run_small(clients=clients, question=NEUTRAL_KEY_QUESTION)

    assert result.tally().accuracy == 1.0
    assert result.attempts[0].answer == "a"  # 上 を提示したときのキー
    assert result.attempts[0].answered_direction == "up"
    assert result.question_name == NEUTRAL_KEY_QUESTION.name


def test_runner_records_fallback_label_and_reports_it():
    """特定のラベルへ落ちるモデルの癖を、レポートの診断用の表で確認できる。"""
    clients = {
        model: KeyClient(model, NEUTRAL_KEY_QUESTION, key="b")  # 常に「下」と答える
        for model in SMALL_PLAN.models
    }

    result, _, _ = run_small(clients=clients, question=NEUTRAL_KEY_QUESTION)
    markdown = render_markdown(result)

    assert result.tally(direction="down").accuracy == 1.0  # 落ち先と一致する向きだけ正解
    assert result.tally(direction="up").accuracy == 0.0
    assert "## 4. 回答の偏り（診断用）" in markdown
    assert "### 回答ラベルの分布" in markdown
    assert "| モデル | b | 失敗 |" in markdown
    # 上 を提示した 8 件は、すべて 下 と回答した（SMALL_PLAN は上下の 2 方向のみ）
    assert "| clef | 上 | 0 | 8 | 0 | 0 |" in markdown


def test_report_splits_accuracy_by_confidence_bucket():
    """レポートは最大確率の帯ごとの正答率を出し、低確信を切り分けられる。"""
    plan = BenchmarkPlan(
        models=("clef",),
        modes=(GenerationMode.CLEAR,),
        visions=(1.0,),
        directions=("up",),
        repeats=3,
    )

    def attempt(repeat: int, probabilities: dict[str, float], correct: bool) -> Attempt:
        return Attempt(
            presentation=Presentation("clef", GenerationMode.CLEAR, 1.0, "up", repeat),
            answer="up" if correct else "down",
            is_correct=correct,
            elapsed_sec=1.0,
            probabilities=probabilities,
            answered_direction="up" if correct else "down",
        )

    result = BenchmarkResult(
        plan=plan,
        attempts=(
            attempt(0, {"up": 0.9, "down": 0.1}, correct=True),
            attempt(1, {"up": 0.5, "down": 0.4}, correct=False),
            attempt(2, {"up": 0.3, "down": 0.35}, correct=False),
        ),
        started_at=datetime(2026, 1, 1, 0, 0, 0),
        finished_at=datetime(2026, 1, 1, 0, 0, 1),
        connection_checks={"clef": True},
    )

    markdown = render_markdown(result)

    assert "### 確信度（最大確率）と正答率" in markdown
    assert "| モデル | 0.8 以上 | 0.6〜0.8 | 0.4〜0.6 | 0.4 未満 |" in markdown
    # 帯ごとに正解数を数える。該当が無い帯は — になる
    assert "| clef | 1/1 (100%) | — | 0/1 (0%) | 0/1 (0%) |" in markdown


def test_cli_selects_question_and_passes_it_to_the_runner(tmp_path, monkeypatch):
    """``--question`` で選んだ質問がランナーへ渡る。"""
    result, _, _ = run_small()
    captured: dict = {}

    def make_runner(*args, **kwargs):
        captured.update(kwargs)
        return StubRunner(result)

    monkeypatch.setattr(cli, "BenchmarkRunner", make_runner)

    exit_code = cli.main(
        [
            *SMALL_PLAN_ARGS,
            "--question",
            "neutral",
            "--output",
            str(tmp_path / "report.md"),
            "--no-raw",
            "--no-state",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert captured["question"] is NEUTRAL_KEY_QUESTION
    assert cli.parse_args(["--question", "reversed"]).question == "reversed"


def test_cli_refuses_resume_with_different_question(tmp_path, monkeypatch, capsys):
    """質問が違う途中経過では再開せず、分かる形で止まる。"""
    state = tmp_path / "state.json"
    write_raw_json(result_of(attempts_for(SMALL_PLAN, model="clef")), state)
    monkeypatch.setattr(cli, "BenchmarkRunner", lambda *args, **kwargs: StubRunner(result_of([])))

    exit_code = cli.main(
        [
            *SMALL_PLAN_ARGS,
            "--question",
            "neutral",
            "--state",
            str(state),
            "--output",
            str(tmp_path / "report.md"),
            "--no-raw",
            "--quiet",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--fresh" in captured.out


# ---------------------------------------------------------------------------
# 実行計画（plan）
# ---------------------------------------------------------------------------


def test_plan_total_counts_every_combination():
    """呈示の総数はモデル × モード × 視力値 × 方向 × 反復になる。"""
    plan = BenchmarkPlan(
        models=("a", "b"),
        modes=(GenerationMode.CLEAR, GenerationMode.MYOPIA),
        visions=(1.0, 0.5),
        directions=("up", "down"),
        repeats=3,
    )

    assert plan.total == 2 * 2 * 2 * 2 * 3
    assert len(plan) == plan.total


def test_default_plan_covers_all_models_modes_visions_directions():
    """既定の計画は 3 モデル・全モード・全視力値・全方向を 4 回ずつ呈示する（2,496 件）。"""
    plan = BenchmarkPlan()

    assert plan.models == config.BENCHMARK_MODELS == ("clef", "clef-flash", "jev")
    assert plan.modes == tuple(GenerationMode)
    assert plan.visions == config.VISION_LEVELS
    assert set(plan.directions) == set(config.DIRECTIONS)
    assert plan.repeats == 4
    assert plan.total == 3 * 4 * 13 * 4 * 4
    assert plan.total == 2496


def test_plan_order_is_model_mode_vision_direction_repeat():
    """呈示の順はモデル → モード → 視力値 → 方向 → 反復になる。"""
    plan = BenchmarkPlan(
        models=("a",),
        modes=(GenerationMode.CLEAR,),
        visions=(1.0, 0.5),
        directions=("up", "down"),
        repeats=2,
    )

    assert list(plan.presentations())[:4] == [
        Presentation("a", GenerationMode.CLEAR, 1.0, "up", 0),
        Presentation("a", GenerationMode.CLEAR, 1.0, "up", 1),
        Presentation("a", GenerationMode.CLEAR, 1.0, "down", 0),
        Presentation("a", GenerationMode.CLEAR, 1.0, "down", 1),
    ]


def test_plan_groups_are_per_model_and_mode():
    """まとまりはモデルとモードごとで、同じモデルが続く。"""
    groups = [(model, mode) for model, mode, _ in SMALL_PLAN.groups()]

    assert groups == [
        ("clef", GenerationMode.CLEAR),
        ("clef", GenerationMode.MYOPIA),
        ("clef-flash", GenerationMode.CLEAR),
        ("clef-flash", GenerationMode.MYOPIA),
    ]
    assert all(len(group) == 2 * 2 * 2 for _, _, group in SMALL_PLAN.groups())


def test_plan_parameters_use_mode_defaults():
    """パラメータは各モードの既定値を使う（清明は空）。"""
    assert SMALL_PLAN.parameters(GenerationMode.CLEAR) == {}
    assert SMALL_PLAN.parameters(GenerationMode.MYOPIA) == default_parameters(GenerationMode.MYOPIA)


def test_plan_parameter_rows_show_mode_labels_and_values():
    """パラメータ表はラベルと値を持ち、同じモードの 2 行目以降はモード名を省く。"""
    rows = BenchmarkPlan(modes=(GenerationMode.CLEAR, GenerationMode.CATARACT)).parameter_rows()

    assert rows[0] == (GenerationMode.CLEAR.value, "—", "—")
    assert rows[1] == (
        GenerationMode.CATARACT.value,
        "boxFilter カーネルサイズ",
        str(config.CATARACT_BOX_KSIZE_DEFAULT),
    )
    assert rows[2][0] == ""  # 同じモードの 2 行目
    assert rows[2][2] == str(config.CATARACT_CONTRAST_DEFAULT)
