"""アプリ全体で共有する設定値（パス・定数・既定値）。

この module を唯一の真実の源（single source of truth）とし、他の module は
数値やパスをここから import する。ここには他の module への依存を持ち込まない
（循環 import を避けるため）。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

#: 効果音の素材を置くディレクトリ（プロジェクト直下）
SOUND_DIR = PROJECT_ROOT / "sound"
#: 全モード・全視力の画像を出力するディレクトリ（プロジェクト直下）
IMAGE_OUTPUT_DIR = PROJECT_ROOT / "image"
#: テストや計測の出力先（プロジェクト直下）
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
#: 統計バッチ（systemone_vision_test.benchmark）のレポート出力先
BENCHMARK_OUTPUT_DIR = ARTIFACTS_DIR / "benchmark"

# ---------------------------------------------------------------------------
# clef API
# ---------------------------------------------------------------------------

CLEF_BASE_URL = "http://localhost:11434/v1/systemone"
#: 実際の接続に使う IPv4 表記。
#: Windows では ``localhost`` が IPv6(::1) を先に試し、Ollama が IPv4 のみを
#: 待ち受けていると接続のタイムアウトまで待ってから IPv4 にフォールバックするため、
#: ホスト名が ``localhost`` のときはこの表記に置き換えて接続する。
PREFER_IPV4_HOST = "127.0.0.1"

#: モデルロードを考慮した待ち時間（秒）
REQUEST_TIMEOUT_SEC = 120.0

#: 判定に使う既定のモデル
DEFAULT_MODEL = "clef"
#: ローカルの Ollama で動くモデル（clef は jev 互換の System One モデル）
LOCAL_MODELS: tuple[str, ...] = ("clef", "clef-flash")
#: インターネット経由で使うモデル（API キーが必要）
ONLINE_MODELS: tuple[str, ...] = ("jev",)
#: GUI のドロップダウンで選択できるモデル（先頭が既定）
AVAILABLE_MODELS: tuple[str, ...] = (*LOCAL_MODELS, *ONLINE_MODELS)
#: 統計バッチの既定モデル。ローカル 2 つと jev の **3 モデルすべて**を測る。
#: jev はインターネット経由なので、既定の実行には接続先と API キーの設定が要る
#: （ローカルだけで回すときは ``--models clef clef-flash`` を指定する）。
BENCHMARK_MODELS: tuple[str, ...] = AVAILABLE_MODELS

# ---------------------------------------------------------------------------
# jev API（TypeSafe / インターネット経由・API キーが必要）
# ---------------------------------------------------------------------------
# 仕様: https://docs.typesafe.ai/images ・ https://api.typesafe.ai/openapi.json
# ローカルの clef は jev 互換だが差分があるので、services/jev_client.py が吸収する。
#   * 接続先 ``https://api.typesafe.ai/v1/systemone``、認証 ``Authorization: Bearer <キー>``
#   * 画像は ``assets`` に置き、その参照名を state（または instructions / criteria）に書く。
#     画像を含むリクエストには ``TypeSafe-Danger-Zone-Images: true`` ヘッダが要る
#     （ローカルの clef は ``images: [Base64]`` を直接渡す）
#   * ``GET <接続先から /systemone を除いた URL>/models`` でモデル一覧を取得できる（接続確認に使う）
#
# **API キーはリポジトリに入れない。** 次のどちらかで与える。
#   1. 環境変数 ``JEV_API_KEY``（推奨。ファイルに残らない）
#   2. :data:`JEV_API_KEY_FILE`（既定 ``.jev_api_key``。1 行にキーだけを書く。git 管理外）
# キーは TypeSafe コンソール（https://console.typesafe.ai/keys）で発行する。
# ``scripts/check_tracked_info.py`` が、追跡対象ファイルへの混入を検出する。

#: jev の接続先を上書きする環境変数の名前。
JEV_BASE_URL_ENV = "JEV_BASE_URL"
#: jev の接続先（System One のエンドポイント）。
#: 既定は **OpenRouter 経由**の ``https://openrouter.ai/api/v1/systemone``
#: （OpenRouter も TypeSafe 互換の SystemOne API を提供している）。
#: TypeSafe に直接つなぐ場合は ``https://api.typesafe.ai/v1/systemone`` を指定する。
JEV_BASE_URL = os.environ.get(JEV_BASE_URL_ENV, "https://openrouter.ai/api/v1/systemone")
#: リクエストの ``model`` に載せる名前（``GET /v1/models`` で確認できる）。
#: ``jev-latest`` は OpenRouter では ``~typesafe/jev-latest`` にマップされるので、
#: 接続先を変えてもこの既定のままでよい。
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
#: 接続先の種類（``typesafe`` / ``openrouter``）。空なら接続先の URL から推定する。
#: ``openrouter`` のときはモデル一覧の名前空間が違うため、一覧との照合を行わない。
JEV_PROVIDER = os.environ.get("JEV_PROVIDER", "").strip().lower()
#: 画像入りのリクエストに Danger Zone ヘッダを付けるか（TypeSafe は必須）。
JEV_SEND_DANGER_ZONE_HEADER = os.environ.get("JEV_SEND_DANGER_ZONE_HEADER", "1") not in (
    "",
    "0",
    "false",
    "False",
)
#: 認証ヘッダの名前。
JEV_AUTH_HEADER = os.environ.get("JEV_AUTH_HEADER", "Authorization")
#: 認証ヘッダの値の書式。``{key}`` が API キーに置き換わる。
JEV_AUTH_SCHEME = os.environ.get("JEV_AUTH_SCHEME", "Bearer {key}")
#: API キーを読む環境変数の名前。
JEV_API_KEY_ENV = "JEV_API_KEY"
#: API キーを書いたファイル（環境変数が優先。git 管理外）。
JEV_API_KEY_FILE = Path(os.environ.get("JEV_API_KEY_FILE", PROJECT_ROOT / ".jev_api_key"))
#: 画像を入れる ``assets`` の参照名。この名前を state に書いて画像を指す。
JEV_IMAGE_ASSET_REFERENCE = os.environ.get("JEV_IMAGE_ASSET_REFERENCE", "landolt")
#: 画像の形式（``png`` / ``jpg``）。生成する画像は PNG。
JEV_IMAGE_FORMAT = os.environ.get("JEV_IMAGE_FORMAT", "png")
#: 1 リクエストのタイムアウト（秒）。インターネット越しなので長めにする。
JEV_REQUEST_TIMEOUT_SEC = float(os.environ.get("JEV_TIMEOUT_SEC", "180"))

# ---------------------------------------------------------------------------
# ランドルト環画像
# ---------------------------------------------------------------------------

#: キャンバス全体の一辺（px）。
#: 最大の環（視力 0.1 で外径 527px）に各フィルタのぼけの広がりを足しても収まる値。
#: 実測（視力 0.1・既定パラメータ）: 清明 264px / 近視 284px / 混濁 280px / 乱視 289px
#: （中心からの半径。640 なら半辺 320px なので最小でも約 31px の余白がある）
CANVAS_SIZE = 640
#: 背景のグレースケール値（白）
BACKGROUND_GRAY = 255
#: ランドルト環のグレースケール値（黒）
RING_GRAY = 0

#: ISO 8596 の比率（外径 : 内径 : 切れ目の幅 = 5 : 3 : 1）
ISO_INNER_DIAMETER_RATIO = 3 / 5
ISO_SLIT_WIDTH_RATIO = 1 / 5
#: 外径（px）= OUTER_DIAMETER_SCALE / 視力
OUTER_DIAMETER_SCALE = 52.7
#: サンプル表示用の視力値。
#: 「キャンバス全体に収まる限界の大きさの環」として、検査で使う最大の環（視力 0.1）を使う。
#: これより小さい視力値だと環が大きすぎてキャンバスからはみ出す
#: （例: 0.07 では外径 753px となり、640px のキャンバスに収まらない）。
SAMPLE_VISION = 0.1

#: 切れ目の開く方向（斜めは対象外）
DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

# ---------------------------------------------------------------------------
# 検査
# ---------------------------------------------------------------------------

#: 検査で使う視力値（高い順）
VISION_LEVELS: tuple[float, ...] = (
    2.0,
    1.5,
    1.2,
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
    0.1,
)
#: 検査開始時の視力値（下限）
INITIAL_VISION = VISION_LEVELS[-1]
#: 正否の表示状態を維持する時間（ms）
VERDICT_HOLD_MS = 3000

# ---------------------------------------------------------------------------
# 生成モードごとのパラメータ（既定値とスライダーの値域）
# ---------------------------------------------------------------------------
# スライダーは整数値なので、実値との換算係数（scale）もここで定義する。

#: 近視：ガウシアンブラーのカーネルサイズ
MYOPIA_GAUSSIAN_KSIZE_DEFAULT = 61
#: (最小, 最大, 刻み)
MYOPIA_GAUSSIAN_KSIZE_RANGE = (3, 121, 2)

#: 混濁：平滑化（boxFilter）のカーネルサイズ
CATARACT_BOX_KSIZE_DEFAULT = 41
CATARACT_BOX_KSIZE_RANGE = (3, 81, 2)

#: 混濁：コントラスト係数（スライダーは x100 の整数）
CATARACT_CONTRAST_DEFAULT = 0.15
CATARACT_CONTRAST_SLIDER_RANGE = (5, 100, 1)
CATARACT_CONTRAST_SCALE = 0.01

#: 乱視：カスタムカーネルのサイズ。
#: 上限 111 は、最大の環（視力 0.1）＋ぼけの広がりがキャンバス（640px）に収まる値から決めている。
ASTIGMATISM_KSIZE_DEFAULT = 61
ASTIGMATISM_KSIZE_RANGE = (15, 111, 2)

#: 乱視：ぼかす方向（度）
ASTIGMATISM_ANGLE_DEFAULT = 0.0
ASTIGMATISM_ANGLE_RANGE = (0, 180, 1)

#: 乱視：ぼかしの強度（スライダーは x100 の整数）
ASTIGMATISM_STRENGTH_DEFAULT = 0.95
ASTIGMATISM_STRENGTH_SLIDER_RANGE = (50, 98, 1)
ASTIGMATISM_STRENGTH_SCALE = 0.01

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

WINDOW_TITLE = "System Oneによる視力検査もどき"

#: 左の操作用パネルの幅（px）
CONTROL_PANEL_WIDTH = 300

#: デフォルトのウィンドウサイズ。
#: 「左の操作パネル + キャンバス（原寸）」で、本体部分の表示領域が
#: キャンバスと同じ大きさ（1:1）になるように決める。
#: 本体部分には枠（QFrame）を付けないので、この値がそのまま表示領域になる。
WINDOW_WIDTH = CONTROL_PANEL_WIDTH + CANVAS_SIZE
WINDOW_HEIGHT = CANVAS_SIZE

#: 中央のオーバーレイ（検査結果）の固定サイズ（px）
RESULT_OVERLAY_WIDTH = 240
RESULT_OVERLAY_HEIGHT = 150
#: 上辺のオーバーレイの高さ（px）
TOP_OVERLAY_HEIGHT = 88
