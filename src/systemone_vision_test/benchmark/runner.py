"""統計用の自走実行。

GUI も効果音も使わないため、判定ごとの待ち時間は入れずに連続して送信する。
通信は :class:`~systemone_vision_test.services.systemone.SystemOneClient` に委譲し、
モデルごとに 1 つ作って使い回す（接続を張り直さない）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime

import numpy as np

from systemone_vision_test import config
from systemone_vision_test.benchmark.plan import BenchmarkPlan, Presentation
from systemone_vision_test.benchmark.results import (
    Attempt,
    BenchmarkResult,
    Tally,
    completed_groups,
)
from systemone_vision_test.domain.landolt import (
    GenerationMode,
    LandoltCParams,
    encode_png_base64,
    generate_landolt_c_image,
)
from systemone_vision_test.domain.questions import DIRECTION_QUESTION, DirectionQuestion
from systemone_vision_test.services.factory import create_client
from systemone_vision_test.services.systemone import SystemOneClient

#: 画像を生成する関数の形（テストで差し替えられるようにしている）
GenerateImage = Callable[[LandoltCParams], np.ndarray]
#: 画像を Base64 文字列にする関数の形
EncodeImage = Callable[[np.ndarray], str]
#: モデル名からクライアントを作る関数の形
ClientFactory = Callable[[str], SystemOneClient]

#: 進捗を何件ごとに報告するか（0 ならまとまりごとの報告だけ）
PROGRESS_EVERY = 25


def default_client_factory(model: str) -> SystemOneClient:
    """モデル名を指定してクライアントを作る（既定の作り方）。

    Args:
        model: 使用するモデル名。

    Returns:
        クライアント。
    """
    return create_client(model)


class BenchmarkRunner:
    """実行計画に沿って画像を呈示し、結果を集計する。

    判定ごとの待ち時間は入れない（画面表示が無いため）。接続確認は実行の前に
    1 回だけ行い、接続できないモデルがあれば :class:`ConnectionError` を投げて
    中止する（長時間の実行を無駄にしないため）。

    Args:
        plan: 実行計画。
        client_factory: モデル名からクライアントを作る関数。
        generate: 画像を生成する関数。
        encode: 画像を Base64 文字列にする関数。
        progress: 進捗を受け取る関数（``None`` なら何も出力しない）。
        checkpoint: まとまり（モデル × モード）ごとに途中経過を受け取る関数。
            ``None`` なら保存しない。
        resume_from: 前回の実行結果（中断からの再開に使う）。記録が揃っている
            まとまりは呈示し直さない。
        question: 使う質問（回答キーと言い回しのバリエーション）。
        clock: 応答時間を測る時計。
        now: 実行時刻を返す関数。
        endpoint: 接続先（レポートの前提条件に載せる）。
    """

    def __init__(
        self,
        plan: BenchmarkPlan,
        *,
        client_factory: ClientFactory = default_client_factory,
        generate: GenerateImage = generate_landolt_c_image,
        encode: EncodeImage = encode_png_base64,
        progress: Callable[[str], None] | None = None,
        checkpoint: Callable[[BenchmarkResult], None] | None = None,
        resume_from: BenchmarkResult | None = None,
        question: DirectionQuestion = DIRECTION_QUESTION,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = datetime.now,
        endpoint: str = config.CLEF_BASE_URL,
    ) -> None:
        self.plan = plan
        self._client_factory = client_factory
        self._generate = generate
        self._encode = encode
        self._progress = progress
        self._checkpoint = checkpoint
        self._resume_from = resume_from
        self._question = question
        self._clock = clock
        self._now = now
        self._endpoint = endpoint

    # -- 公開 API -----------------------------------------------------------

    def run(self) -> BenchmarkResult:
        """計画をすべて実行する。

        Returns:
            呈示ごとの結果と実行条件をまとめた結果。

        Raises:
            ConnectionError: 接続できないモデルがあった場合。
        """
        started_at = self._now()
        checks = self.verify_connection()
        unreachable = [model for model, ok in checks.items() if not ok]
        if unreachable:
            raise ConnectionError("接続できないモデルがあります: " + ", ".join(unreachable))

        resumed_attempts, completed = self._resumed()
        attempts: list[Attempt] = list(resumed_attempts)
        if resumed_attempts:
            self._report(
                f"前回の記録 {len(resumed_attempts)} 件を引き継ぎます"
                f"（記録済みの {len(completed)} まとまりはスキップ）"
            )
        remaining = self.plan.total - len(resumed_attempts)
        self._report(f"質問: {self._question.name}")
        self._report(f"呈示 {self.plan.total} 件のうち {remaining} 件を実行します")

        clients: dict[str, SystemOneClient] = {}
        for model, mode, group in self.plan.groups():
            if (model, mode) in completed:
                self._report(f"[{model}] {mode.value}: 記録済みのためスキップ")
                continue

            client = clients.setdefault(model, self._client_factory(model))
            parameters = self.plan.parameters(mode)
            self._report(f"[{model}] {mode.value}: {len(group)} 件")

            for presentation in group:
                attempts.append(self._present(client, presentation, parameters))
                if PROGRESS_EVERY and len(attempts) % PROGRESS_EVERY == 0:
                    self._report(f"  ... {len(attempts)}/{self.plan.total} 件")

            group_tally = Tally.of_attempts(attempts[-len(group) :])
            self._report(f"[{model}] {mode.value} 完了: {self._format(group_tally)}")
            self._save_checkpoint(attempts, started_at, checks, len(resumed_attempts))

        finished_at = self._now()
        elapsed = (finished_at - started_at).total_seconds()
        self._report(f"完了: {len(attempts)} 件 / 所要 {elapsed:.1f} 秒")
        return BenchmarkResult(
            plan=self.plan,
            attempts=tuple(attempts),
            started_at=started_at,
            finished_at=finished_at,
            connection_checks=checks,
            endpoint=self._endpoint,
            question_name=self._question.name,
            resumed_from=len(resumed_attempts),
        )

    # -- 中断からの再開 -----------------------------------------------------

    def _resumed(self) -> tuple[tuple[Attempt, ...], frozenset[tuple[str, GenerationMode]]]:
        """引き継ぐ呈示と、呈示し直さない（モデル, モード）を返す。

        Returns:
            ``(引き継ぐ呈示, 記録済みのまとまり)``。再開しない場合は空。
        """
        if self._resume_from is None:
            return (), frozenset()

        completed = completed_groups(self.plan, self._resume_from.attempts)
        attempts = tuple(
            attempt
            for attempt in self._resume_from.attempts
            if (attempt.presentation.model, attempt.presentation.mode) in completed
        )
        return attempts, completed

    def _save_checkpoint(
        self,
        attempts: list[Attempt],
        started_at: datetime,
        checks: dict[str, bool],
        resumed_from: int,
    ) -> None:
        """途中経過を保存する（まとまりが終わるたびに呼ぶ）。"""
        if self._checkpoint is None:
            return
        self._checkpoint(
            BenchmarkResult(
                plan=self.plan,
                attempts=tuple(attempts),
                started_at=started_at,
                finished_at=self._now(),
                connection_checks=checks,
                endpoint=self._endpoint,
                question_name=self._question.name,
                resumed_from=resumed_from,
            )
        )

    def verify_connection(self) -> dict[str, bool]:
        """モデルごとに接続性を確認する（モデルの読み込みを兼ねる）。

        Returns:
            モデル名をキーにした接続可否。
        """
        results: dict[str, bool] = {}
        for model in self.plan.models:
            client = self._client_factory(model)
            connected = client.check_connection()
            results[model] = connected
            self._report(f"[{model}] 接続確認: {'OK' if connected else 'NG'}")
        return results

    # -- 内部処理 -----------------------------------------------------------

    def _present(
        self,
        client: SystemOneClient,
        presentation: Presentation,
        parameters: dict[str, float | int],
    ) -> Attempt:
        """1 回呈示して結果を返す（待ち時間は入れない）。"""
        spec = LandoltCParams(
            vision=presentation.vision,
            direction=presentation.direction,
            mode=presentation.mode,
        ).with_parameters(parameters)
        image_base64 = self._encode(self._generate(spec))

        started = self._clock()
        answer = client.inspect_landolt_c(image_base64, question=self._question)
        elapsed_sec = self._clock() - started

        if answer is None:
            return Attempt(
                presentation=presentation,
                answer=None,
                is_correct=None,
                elapsed_sec=elapsed_sec,
            )

        # 回答キーは質問によって変わる（a/b/c/d など）ので、方向に変換して採点する
        direction = self._question.direction_of(answer.choice)
        return Attempt(
            presentation=presentation,
            answer=answer.choice,
            is_correct=direction == presentation.direction,
            elapsed_sec=elapsed_sec,
            probabilities=dict(answer.probabilities),
            answered_direction=direction,
        )

    def _report(self, message: str) -> None:
        """進捗を伝える。"""
        if self._progress is not None:
            self._progress(message)

    @staticmethod
    def _format(tally: Tally) -> str:
        """``正答数/呈示数 (正答率)`` の形にする（回答が無かった件数があれば併記）。"""
        rate = tally.accuracy
        percent = "—" if rate is None else f"{rate:.0%}"
        text = f"{tally.correct}/{tally.total} ({percent})"
        return f"{text} 失敗 {tally.failed}" if tally.failed else text
