"""モデルに投げる方向判定の質問（System One の ``questions`` の定義）。

既定は日本語の説明・英語のキー。**回答ラベルの事前分布を切り分ける**ために、言い回し・
回答キー・説明文を差し替えた変種を用意している。採点は :attr:`DirectionQuestion.labels`
を通すので、キーが変わっても方向として比較できる。

この module は HTTP も GUI も知らない純粋な定義だけを持つ（クライアントは
:mod:`systemone_vision_test.services.clef_client` /
:mod:`systemone_vision_test.services.jev_client`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from systemone_vision_test.domain.landolt import Direction

#: 方向判定に使う問題文
DIRECTION_STATE_TEXT = "視力検査用のランドルト環の画像"
DIRECTION_INSTRUCTION = "提示した画像のランドルト環の切れ目は、上下左右のどの方向に開いていますか？"
DIRECTION_CRITERIA = {
    "up": "上の部分が開いている",
    "down": "下の部分が開いている",
    "left": "左の部分が開いている",
    "right": "右の部分が開いている",
}

#: 方向判定の質問に使うキー（API の ``questions`` の名前）
DIRECTION_QUESTION_KEY = "landolt_c_direction"


@dataclass(frozen=True)
class DirectionQuestion:
    """方向判定の質問の作り方。

    Attributes:
        name: 質問の種類の名前（レポートやログに出す）。
        instruction: 質問文。
        criteria: 回答キー → 説明（この並び順で API に渡る）。
        labels: 回答キー → 方向（採点に使う）。
        key: API の ``questions`` に使うキー。
    """

    name: str
    instruction: str
    criteria: dict[str, str]
    labels: dict[str, Direction]
    key: str = DIRECTION_QUESTION_KEY

    def direction_of(self, answer: str | None) -> Direction | None:
        """回答キーを方向に変換する。

        Args:
            answer: API が返した回答キー。

        Returns:
            対応する方向。未知のキーや ``None`` なら ``None``。
        """
        return self.labels.get(answer or "")

    def question(self) -> dict[str, object]:
        """API の ``questions`` に載せる 1 問分の定義を返す。

        Returns:
            ``{"type": "choice", "instructions": ..., "criteria": {...}}``。
        """
        return {
            "type": "choice",
            "instructions": self.instruction,
            "criteria": dict(self.criteria),
        }


def _identity_labels() -> dict[str, Direction]:
    """英語のキーをそのまま方向として扱う対応表を作る。"""
    return {key: cast("Direction", key) for key in DIRECTION_CRITERIA}


#: 既定（日本語の説明・英語のキー）
DIRECTION_QUESTION = DirectionQuestion(
    name="既定（日本語の説明・英語のキー）",
    instruction=DIRECTION_INSTRUCTION,
    criteria=dict(DIRECTION_CRITERIA),
    labels=_identity_labels(),
)

#: 回答キーを中立化（ラベル名の語彙による事前を切り分ける）
NEUTRAL_KEY_QUESTION = DirectionQuestion(
    name="中立キー（a=上 / b=下 / c=左 / d=右）",
    instruction=(
        f"{DIRECTION_INSTRUCTION}"
        " 回答は a（上）/ b（下）/ c（左）/ d（右）の記号から選んでください。"
    ),
    criteria={
        "a": "上の部分が開いている",
        "b": "下の部分が開いている",
        "c": "左の部分が開いている",
        "d": "右の部分が開いている",
    },
    labels={"a": "up", "b": "down", "c": "left", "d": "right"},
)

#: 回答キーを日本語に（説明とキーの言語が食い違う負荷を外す）
JAPANESE_KEY_QUESTION = DirectionQuestion(
    name="日本語キー（上/下/左/右）",
    instruction=DIRECTION_INSTRUCTION,
    criteria={
        "上": "上の部分が開いている",
        "下": "下の部分が開いている",
        "左": "左の部分が開いている",
        "右": "右の部分が開いている",
    },
    labels={"上": "up", "下": "down", "左": "left", "右": "right"},
)

#: 説明の並び順を逆に（選択肢の順序による位置バイアーの確認）
REVERSED_ORDER_QUESTION = DirectionQuestion(
    name="逆順（右/左/下/上の順に提示）",
    instruction=DIRECTION_INSTRUCTION,
    criteria={key: DIRECTION_CRITERIA[key] for key in reversed(list(DIRECTION_CRITERIA))},
    labels=_identity_labels(),
)

#: 説明文を入れ替えた質問（キーと説明の向きが食い違う）。
#: 落ち先が「キーの語彙」なのか「説明文の意味」なのかを切り分けるために使う。
SWAPPED_DESCRIPTION_QUESTION = DirectionQuestion(
    name="説明の入れ替え（キーと説明の向きが逆）",
    instruction=DIRECTION_INSTRUCTION,
    criteria={
        "up": "下の部分が開いている",
        "down": "上の部分が開いている",
        "left": "右の部分が開いている",
        "right": "左の部分が開いている",
    },
    labels={"up": "down", "down": "up", "left": "right", "right": "left"},
)

#: 反対質問（切れ目が開いていない側＝枠がつながっている側を答える）。
#: 落ち先が「切れ目は下向き」という判断（意味）なのか、説明文の語句なのかを切り分ける。
CLOSED_SIDE_QUESTION = DirectionQuestion(
    name="反対質問（つながっている側を答える）",
    instruction=(
        "提示した画像のランドルト環について、"
        "切れ目が開いていない側（枠がつながっている側）は、上下左右のどの方向ですか？"
    ),
    criteria={
        "up": "上側がつながっている",
        "down": "下側がつながっている",
        "left": "左側がつながっている",
        "right": "右側がつながっている",
    },
    labels={"up": "down", "down": "up", "left": "right", "right": "left"},
)

#: 回答キーを別の語彙（方位）にした質問。``left`` という語そのものへの偏りを切り分ける。
COMPASS_KEY_QUESTION = DirectionQuestion(
    name="別語彙キー（north=上 / south=下 / west=左 / east=右）",
    instruction=(
        f"{DIRECTION_INSTRUCTION}"
        " 回答は north（上）/ south（下）/ west（左）/ east（右）から選んでください。"
    ),
    criteria={
        "north": "上の部分が開いている",
        "south": "下の部分が開いている",
        "west": "左の部分が開いている",
        "east": "右の部分が開いている",
    },
    labels={"north": "up", "south": "down", "west": "left", "east": "right"},
)

#: 回答キーをかなにした質問（日本語の語彙でも同じ偏りが出るか）
HIRAGANA_KEY_QUESTION = DirectionQuestion(
    name="かなキー（うえ / した / ひだり / みぎ）",
    instruction=f"{DIRECTION_INSTRUCTION} 回答は うえ / した / ひだり / みぎ から選んでください。",
    criteria={
        "うえ": "上の部分が開いている",
        "した": "下の部分が開いている",
        "ひだり": "左の部分が開いている",
        "みぎ": "右の部分が開いている",
    },
    labels={"うえ": "up", "した": "down", "ひだり": "left", "みぎ": "right"},
)

#: ``compass`` と同じキー語彙で、**説明の対応だけをずらした**質問。
#: ``east`` の位置（4 番目）は変えずに説明を「上」へ回し、``compass`` で ``east`` へ
#: 落ちたのが「語（east）」のせいか「説明の内容（右）」のせいかを切り分ける。
COMPASS_SHIFTED_QUESTION = DirectionQuestion(
    name="別語彙キー＋説明をずらす（north=左 / south=右 / west=下 / east=上）",
    instruction=(
        f"{DIRECTION_INSTRUCTION}"
        " 回答は north（左）/ south（右）/ west（下）/ east（上）から選んでください。"
    ),
    criteria={
        "north": "左の部分が開いている",
        "south": "右の部分が開いている",
        "west": "下の部分が開いている",
        "east": "上の部分が開いている",
    },
    labels={"north": "left", "south": "right", "west": "down", "east": "up"},
)

#: ``compass`` とキーは同じで、質問文の補足（日本語の言い換え）を外した質問。
#: 追加した補足が ``east`` への落ち込みを生んでいないかを確かめる。
COMPASS_NO_GLOSS_QUESTION = DirectionQuestion(
    name="別語彙キー（補足なし: north / south / west / east）",
    instruction=f"{DIRECTION_INSTRUCTION} 回答は north / south / west / east から選んでください。",
    criteria=dict(COMPASS_KEY_QUESTION.criteria),
    labels=dict(COMPASS_KEY_QUESTION.labels),
)

#: 選べる質問（``--question`` の名前で指定する）
DIRECTION_QUESTIONS: dict[str, DirectionQuestion] = {
    "default": DIRECTION_QUESTION,
    "neutral": NEUTRAL_KEY_QUESTION,
    "japanese": JAPANESE_KEY_QUESTION,
    "reversed": REVERSED_ORDER_QUESTION,
    "swapped": SWAPPED_DESCRIPTION_QUESTION,
    "closed": CLOSED_SIDE_QUESTION,
    "compass": COMPASS_KEY_QUESTION,
    "hiragana": HIRAGANA_KEY_QUESTION,
    "compass_shift": COMPASS_SHIFTED_QUESTION,
    "compass_nogloss": COMPASS_NO_GLOSS_QUESTION,
}
