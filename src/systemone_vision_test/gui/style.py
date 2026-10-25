"""GUI の見た目（スタイルシートとオブジェクト名）をまとめる。

オーバーレイの枠は ID セレクタ（``#名前``）で指定する。
``QLabel`` は ``QFrame`` を継承しているため、``QFrame`` セレクタで指定すると
子のラベルにも枠が描かれて二重に見えてしまうため。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 上辺のオーバーレイ（補助情報）
# ---------------------------------------------------------------------------

TOP_OVERLAY_OBJECT_NAME = "topOverlay"
TOP_OVERLAY_STYLE = (
    f"#{TOP_OVERLAY_OBJECT_NAME} {{"
    " background-color: rgba(255, 255, 255, 225);"
    " border-bottom: 2px solid #bbb; }"
)

# ---------------------------------------------------------------------------
# 中央のオーバーレイ（検査結果）
# ---------------------------------------------------------------------------

RESULT_OVERLAY_OBJECT_NAME = "resultOverlay"
RESULT_OVERLAY_STYLE = (
    f"#{RESULT_OVERLAY_OBJECT_NAME} {{"
    " background-color: rgba(255, 255, 255, 225);"
    " border: 2px solid #999; border-radius: 10px; }"
)
#: 結果文字列のスタイル（``{}`` にフォントサイズ px を埋める）
RESULT_TEXT_STYLE_TEMPLATE = "color: black; font-weight: bold; font-size: {size}px;"
#: 枠の内側に確保する余白（px）。文字サイズの自動調整に使う。
RESULT_TEXT_PADDING = 16

# ---------------------------------------------------------------------------
# その他
# ---------------------------------------------------------------------------

CANVAS_STYLE = "QLabel { background-color: white; }"
SAMPLE_STYLE = "QLabel { background-color: #f0f0f0; }"

BOLD_STYLE = "font-weight: bold;"
MUTED_STYLE = "color: #444;"
MODEL_LABEL_STYLE = "font-weight: bold; color: #333;"
