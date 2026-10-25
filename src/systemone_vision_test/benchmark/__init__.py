"""統計用の自走バッチ（GUI・効果音を使わない）。

全モデル × 全モード × 全視力値 × 全方向を決まった回数だけ呈示して正答率を集計し、
Markdown のレポートと生データ（JSON）を出力する。

使い方::

    python -m systemone_vision_test.benchmark
    python -m systemone_vision_test.benchmark --models clef-flash
    python -m systemone_vision_test.benchmark --modes 清明 --visions 2.0 --repeats 1
"""

from systemone_vision_test.benchmark.plan import DEFAULT_REPEATS, BenchmarkPlan, Presentation
from systemone_vision_test.benchmark.report import (
    format_tally,
    load_raw_json,
    raw_data,
    render_markdown,
    write_markdown,
    write_raw_json,
)
from systemone_vision_test.benchmark.results import (
    Attempt,
    BenchmarkResult,
    Tally,
    completed_groups,
)
from systemone_vision_test.benchmark.runner import BenchmarkRunner, default_client_factory

__all__ = [
    "DEFAULT_REPEATS",
    "Attempt",
    "BenchmarkPlan",
    "BenchmarkResult",
    "BenchmarkRunner",
    "Presentation",
    "Tally",
    "completed_groups",
    "default_client_factory",
    "format_tally",
    "load_raw_json",
    "raw_data",
    "render_markdown",
    "write_markdown",
    "write_raw_json",
]
