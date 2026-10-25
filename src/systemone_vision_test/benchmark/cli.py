"""統計バッチのコマンドライン（計画の組み立て・実行・レポート出力）。"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import cast

from systemone_vision_test import config
from systemone_vision_test.benchmark.plan import DEFAULT_REPEATS, BenchmarkPlan
from systemone_vision_test.benchmark.report import load_raw_json, write_markdown, write_raw_json
from systemone_vision_test.benchmark.results import BenchmarkResult
from systemone_vision_test.benchmark.runner import BenchmarkRunner, ClientFactory
from systemone_vision_test.domain.landolt import Direction, GenerationMode
from systemone_vision_test.domain.questions import DIRECTION_QUESTIONS
from systemone_vision_test.services.clef_client import ClefClient
from systemone_vision_test.services.factory import create_client, is_online_model

#: 出力ファイル名に付ける日時の書式
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def positive_int(value: str) -> int:
    """1 以上の整数として解釈する。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1 以上の整数を指定してください")
    return number


def positive_float(value: str) -> float:
    """0 より大きい実数として解釈する。"""
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("0 より大きい数を指定してください")
    return number


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数のパーサを組み立てる（``--help`` の並び順はここで決まる）。"""
    parser = argparse.ArgumentParser(description=__doc__)
    _add_plan_arguments(parser)
    _add_connection_arguments(parser)
    _add_output_arguments(parser)
    _add_behavior_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解釈する。

    Args:
        argv: 引数の並び（``None`` なら ``sys.argv``）。

    Returns:
        解釈した引数。
    """
    return build_parser().parse_args(argv)


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    """実行計画（モデル・モード・視力値・方向・反復・質問）の引数を追加する。"""
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(config.BENCHMARK_MODELS),
        help="対象モデル（既定: %(default)s）。インターネット経由のモデル（jev）を"
        "指定するときは、接続先と API キーの設定が要る",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=[mode.value for mode in GenerationMode],
        default=[mode.value for mode in GenerationMode],
        help="対象モード（既定: すべて）",
    )
    parser.add_argument(
        "--visions",
        nargs="*",
        type=positive_float,
        default=list(config.VISION_LEVELS),
        help="対象の視力値（既定: %(default)s）",
    )
    parser.add_argument(
        "--directions",
        nargs="*",
        choices=list(config.DIRECTIONS),
        default=list(config.DIRECTIONS),
        help="対象の切れ目の方向（既定: すべて）",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=DEFAULT_REPEATS,
        help="各条件の呈示回数（既定: %(default)s）",
    )
    parser.add_argument(
        "--question",
        choices=sorted(DIRECTION_QUESTIONS),
        default="default",
        help="方向を尋ねる質問の種類。回答キーや言い回しを変えて回答の偏りを切り分ける"
        "（既定: %(default)s）",
    )


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    """接続（エンドポイント・タイムアウト）の引数を追加する。"""
    parser.add_argument(
        "--base-url",
        default=config.CLEF_BASE_URL,
        help="API のエンドポイント（既定: %(default)s）",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=config.REQUEST_TIMEOUT_SEC,
        help="1 リクエストのタイムアウト秒（既定: %(default)s）",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """出力（レポート・生データ・途中経過）の引数を追加する。"""
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"レポートの出力先。ディレクトリを指定した場合はその下に既定名で作る"
        f"（既定: {config.BENCHMARK_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="生データ（JSON）の出力先（既定: レポートと同じ名前の .json）",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=config.BENCHMARK_OUTPUT_DIR / "benchmark.state.json",
        help="途中経過（チェックポイント）の保存先。実行時にあれば続きから再開する"
        "（既定: %(default)s）",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="生データ（JSON）を書き出さない",
    )


def _add_behavior_arguments(parser: argparse.ArgumentParser) -> None:
    """実行の振る舞い（途中経過の扱いと進捗表示）の引数を追加する。"""
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="既存の途中経過を無視して最初から実行する",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="途中経過を保存しない（再開もしない）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="進捗を表示しない",
    )


def build_plan(args: argparse.Namespace) -> BenchmarkPlan:
    """引数から実行計画を組み立てる。

    Args:
        args: 解釈済みの引数。

    Returns:
        実行計画。
    """
    return BenchmarkPlan(
        models=tuple(args.models),
        modes=tuple(GenerationMode(name) for name in args.modes),
        visions=tuple(args.visions),
        directions=cast("tuple[Direction, ...]", tuple(args.directions)),
        repeats=args.repeats,
    )


def report_endpoint(plan: BenchmarkPlan, base_url: str) -> str:
    """レポートに載せる接続先を返す。

    インターネット経由のモデル（jev）が入っているときは、その接続先も併記する。

    Args:
        plan: 実行計画。
        base_url: ローカルモデルの接続先（``--base-url``）。

    Returns:
        前提条件に載せる接続先の文字列。
    """
    if not any(is_online_model(model) for model in plan.models):
        return base_url
    jev_url = f"JEV_BASE_URL: {config.JEV_BASE_URL or '（未設定）'} / model: {config.JEV_MODEL}"
    if not any(not is_online_model(model) for model in plan.models):
        return jev_url  # jev だけの実行ではローカルの接続先を出さない
    return f"{base_url} / {jev_url}"


def build_client_factory(args: argparse.Namespace) -> ClientFactory:
    """引数の接続先とタイムアウトを使うクライアントのファクトリを作る。

    Args:
        args: 解釈済みの引数。

    Returns:
        モデル名からクライアントを作る関数。
    """

    def factory(model: str) -> ClefClient:
        if is_online_model(model):
            # インターネット経由のモデルは、接続先とタイムアウトを config（環境変数）から取る
            return create_client(model)
        return ClefClient(base_url=args.base_url, model=model, timeout_sec=args.timeout)

    return factory


def resolve_paths(args: argparse.Namespace, timestamp: str) -> tuple[Path, Path | None]:
    """レポートと生データの出力先を決める。

    Args:
        args: 解釈済みの引数。
        timestamp: ファイル名に付ける日時。

    Returns:
        ``(レポートのパス, 生データのパスまたは None)``。
    """
    output: Path = args.output or config.BENCHMARK_OUTPUT_DIR
    if output.is_dir() or output.suffix == "":
        output = output / f"benchmark_{timestamp}.md"

    raw: Path | None = None
    if not args.no_raw:
        raw = args.raw_output or output.with_suffix(".json")
    return output, raw


def load_resume(
    state_path: Path | None,
    plan: BenchmarkPlan,
    *,
    question_name: str,
    fresh: bool,
) -> BenchmarkResult | None:
    """途中経過を読み込み、引き継ぐ結果を返す。

    Args:
        state_path: 途中経過の保存先。``None`` なら再開しない。
        plan: 今回の実行計画。
        question_name: 今回の質問の名前（前回と違えば再開しない）。
        fresh: ``True`` なら既存の途中経過を無視する。

    Returns:
        引き継ぐ結果。再開しない場合は ``None``。

    Raises:
        ValueError: 途中経過の実行条件や質問が今回と違う場合。
    """
    if state_path is None or fresh or not state_path.is_file():
        return None

    result = load_raw_json(state_path).keep_complete_groups()
    if result.plan != plan or result.question_name != question_name:
        raise ValueError(
            "途中経過の実行条件（または質問）が今回と違います。"
            "--fresh を付けて最初から実行してください。"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """統計を実行し、レポートを出力する。

    実行の途中で中断しても、まとまり（モデル × モード）ごとに保存した途中経過から
    続きを再開できる。

    Args:
        argv: コマンドライン引数（``None`` なら ``sys.argv``）。

    Returns:
        終了コード（接続できない場合や実行条件が合わない場合は 1）。
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    plan = build_plan(args)
    question = DIRECTION_QUESTIONS[args.question]
    report_path, raw_path = resolve_paths(args, datetime.now().strftime(TIMESTAMP_FORMAT))
    state_path: Path | None = None if args.no_state else args.state

    print(f"モデル: {', '.join(plan.models)}")
    print(
        f"モード: {' / '.join(mode.value for mode in plan.modes)}"
        f" / 視力値 {len(plan.visions)} 段階 / 方向 {len(plan.directions)}"
        f" / 反復 {plan.repeats} 回"
    )
    print(f"質問: {question.name}")
    print(f"呈示の総数: {plan.total} 件（GUI ではないので待ち時間なしで連続実行します）")

    try:
        resumed = load_resume(state_path, plan, question_name=question.name, fresh=args.fresh)
    except ValueError as error:
        print(f"エラー: {error}")
        return 1

    if resumed is not None:
        print(f"前回の途中経過 {len(resumed.attempts)} 件を引き継ぎます: {state_path}")
    elif state_path is not None and state_path.is_file() and args.fresh:
        print(f"既存の途中経過を無視します（--fresh）: {state_path}")
    if state_path is not None:
        print(f"途中経過: {state_path}（まとまりごとに保存し、再実行で続きから再開します）")

    runner = BenchmarkRunner(
        plan,
        client_factory=build_client_factory(args),
        endpoint=report_endpoint(plan, args.base_url),
        progress=None if args.quiet else print,
        checkpoint=None if state_path is None else partial(write_raw_json, path=state_path),
        resume_from=resumed,
        question=question,
    )
    try:
        result = runner.run()
    except ConnectionError as error:
        print(f"エラー: {error}")
        print("Ollama が起動しているか、対象モデルを取得済みかを確認してください。")
        if any(is_online_model(model) for model in plan.models):
            print(
                "インターネット経由のモデル（jev）は、接続先 config.JEV_BASE_URL"
                f"（環境変数 JEV_BASE_URL）と API キー（環境変数 {config.JEV_API_KEY_ENV} か"
                f"キーファイル {config.JEV_API_KEY_FILE.name}）の設定も確認してください。"
            )
        return 1

    if state_path is not None and len(result.attempts) >= plan.total:
        state_path.unlink(missing_ok=True)  # 完了したので途中経過は要らない
        print(f"完了したため途中経過を削除しました: {state_path}")

    print(f"レポート: {write_markdown(result, report_path)}")
    if raw_path is not None:
        print(f"生データ: {write_raw_json(result, raw_path)}")
    return 0
