"""System One の応答から回答を取り出す（API ごとの形の違いをここで吸収する）。

基本の形は ``answers.<質問キー>.{choice, probabilities, confidence}``。加えて、
``answer`` / ``value`` / ``result`` / ``choices[0].message.content``（JSON 文字列も可）/
トップレベルの ``choice`` / 日本語の方向語も解釈する。確率が無い応答でも動く。

HTTP も GUI も知らない純粋な関数だけを置き、通信は各クライアント
(:mod:`systemone_vision_test.services.clef_client` /
:mod:`systemone_vision_test.services.jev_client`) が担当する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from systemone_vision_test.domain.landolt import Direction
from systemone_vision_test.domain.questions import DirectionQuestion


@dataclass(frozen=True)
class DirectionAnswer:
    """モデルが返したランドルト環の方向判定。

    Attributes:
        choice: 選ばれた回答（質問のキー。方向へは :meth:`DirectionQuestion.direction_of`）。
        probabilities: 回答キーごとの確率。
        confidence: 回答の自信度（最大確率）。
    """

    choice: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


#: 方向の別名。API がキー以外の語（日本語・方位など）で答えたときに吸収する。
DIRECTION_ALIASES: dict[str, Direction] = {
    "up": "up",
    "top": "up",
    "upper": "up",
    "above": "up",
    "north": "up",
    "上": "up",
    "上側": "up",
    "うえ": "up",
    "down": "down",
    "bottom": "down",
    "lower": "down",
    "below": "down",
    "south": "down",
    "下": "down",
    "下側": "down",
    "した": "down",
    "left": "left",
    "west": "left",
    "左": "left",
    "左側": "left",
    "ひだり": "left",
    "right": "right",
    "east": "right",
    "右": "right",
    "右側": "right",
    "みぎ": "right",
}


def normalize_choice(value: Any) -> str:
    """回答の表記ゆれ（前後の空白・引用符・句読点）を落とす。

    Args:
        value: API が返した値。

    Returns:
        前後の飾りを落とした文字列。
    """
    text = "" if value is None else str(value)
    return text.strip().strip("\"'`「」。．.、,;：:")


def direction_of_text(value: Any) -> Direction | None:
    """回答から方向を取り出す（別名・日本語・文中の語も見る）。

    Args:
        value: API が返した値。

    Returns:
        方向。方向として読めなければ ``None``。
    """
    text = normalize_choice(value).lower()
    if not text:
        return None
    if text in DIRECTION_ALIASES:
        return DIRECTION_ALIASES[text]
    for alias, direction in DIRECTION_ALIASES.items():
        if alias in text:
            return direction
    return None


def answer_key_for(question: DirectionQuestion, choice: str) -> str:
    """回答を質問のキーに寄せる（採点はキーを方向に変換して行うため）。

    Args:
        question: 使った質問。
        choice: API が返した回答。

    Returns:
        ``question`` のキー。方向として読めるならそのキー、読めなければ元の文字列。
    """
    if choice in question.criteria:
        return choice
    lowered = {key.lower(): key for key in question.criteria}
    if choice.lower() in lowered:
        return lowered[choice.lower()]
    direction = direction_of_text(choice)
    if direction is None:
        return choice
    for key, value in question.labels.items():
        if value == direction:
            return key
    return choice


def response_body(response: Any, limit: int = 400) -> str:
    """応答の本文を、ログに出せる長さに切って返す（読めなければその旨）。

    Args:
        response: ``requests.Response`` 相当。
        limit: 最大文字数。

    Returns:
        本文（改行は空白に置き換える）。
    """
    text = getattr(response, "text", "") or ""
    if not text:
        return "(本文なし)"
    return " ".join(str(text).split())[:limit]


def response_json(data: Any, limit: int = 400) -> str:
    """応答の辞書を、ログに出せる 1 行の文字列にする。

    Args:
        data: 応答の JSON。
        limit: 最大文字数。

    Returns:
        1 行にした JSON（長ければ切る）。
    """
    try:
        return json.dumps(data, ensure_ascii=False)[:limit]
    except (TypeError, ValueError):  # pragma: no cover - 想定外の型
        return str(data)[:limit]


def _is_choice(value: Any) -> bool:
    """回答として使える値かどうか（文字列・数値・真偽値）。"""
    return isinstance(value, (str, int, float, bool))


def _choice_of(source: dict[str, Any]) -> str | None:
    """辞書から回答を取り出す（キー名の違いを吸収する）。"""
    for name in ("choice", "answer", "value", "label", "response", "text", "content", "result"):
        if _is_choice(source.get(name)):
            text = normalize_choice(source[name])
            if text:
                return text
    return None


def _probabilities_of(source: dict[str, Any]) -> dict[str, float]:
    """辞書から確率を取り出す（``logprobs`` 形式も見る。無ければ空）。"""
    for name in ("probabilities", "probability", "logprobs"):
        raw = source.get(name)
        if not isinstance(raw, dict):
            continue
        probabilities: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, (int, float)):
                probabilities[str(key)] = float(value)
            elif isinstance(value, dict) and isinstance(value.get("probability"), (int, float)):
                probabilities[str(key)] = float(value["probability"])
        if probabilities:
            return probabilities
    return {}


def _confidence_of(source: dict[str, Any], probabilities: dict[str, float]) -> float:
    """確信度を取り出す（明示が無ければ確率の最大値）。"""
    value = source.get("confidence")
    if isinstance(value, (int, float)):
        return float(value)
    return max(probabilities.values(), default=0.0)


def _json_dict(text: str) -> dict[str, Any] | None:
    """文字列が JSON オブジェクトなら辞書にして返す。"""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _sources(data: dict[str, Any], key: str) -> list[Any]:
    """応答 JSON から「回答が入っていそうな値」を、確からしい順に集める。"""
    sources: list[Any] = []

    answers = data.get("answers")
    if isinstance(answers, dict):
        if isinstance(answers.get(key), dict):
            sources.append(answers[key])
        if _choice_of(answers) is not None:
            sources.append(answers)  # 質問が 1 つなら answers 自体が回答のこともある
    for name in ("answer", "result", "output", "response", "data", "content"):
        value = data.get(name)
        if isinstance(value, dict):
            nested = value.get(key)
            sources.append(nested if isinstance(nested, dict) else value)
        elif isinstance(value, str):
            sources.append(value)

    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        message = first.get("message")
        sources.append(message if isinstance(message, dict) else first)
        sources.append(first.get("text"))

    sources.append(data)  # トップレベルに choice / answer がある形
    return sources


def extract_answer(data: dict[str, Any], key: str) -> DirectionAnswer | None:
    """応答 JSON から ``(回答, 確率, 確信度)`` を取り出す（形の差を吸収する）。

    Args:
        data: 応答の JSON。
        key: 質問のキー（``landolt_c_direction`` など）。

    Returns:
        取り出した回答。取り出せなければ ``None``。
    """
    for source in _sources(data, key):
        if isinstance(source, dict):
            choice = _choice_of(source)
            if choice is None:
                continue
            probabilities = _probabilities_of(source)
            return DirectionAnswer(
                choice=choice,
                probabilities=probabilities,
                confidence=_confidence_of(source, probabilities),
            )
        if isinstance(source, str):
            text = normalize_choice(source)
            if not text:
                continue
            nested = _json_dict(text)
            if nested is not None:
                answer = extract_answer(nested, key)
                if answer is not None:
                    return answer
            return DirectionAnswer(choice=text)
    return None
