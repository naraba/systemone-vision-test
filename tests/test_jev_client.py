"""jev API クライアント（services.jev_client）とモデルの振り分けのテスト。

``requests.Session`` をフェイクに差し替えるため、実際のサーバは不要（ネットワークにも出ない）。
API キーはテスト内で組み立て、リポジトリに実物を置かない。
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import requests

from systemone_vision_test import config
from systemone_vision_test.benchmark import cli
from systemone_vision_test.benchmark.plan import BenchmarkPlan
from systemone_vision_test.domain.questions import DIRECTION_QUESTION, NEUTRAL_KEY_QUESTION
from systemone_vision_test.services import factory
from systemone_vision_test.services.answers import answer_key_for, direction_of_text, extract_answer
from systemone_vision_test.services.clef_client import ClefClient
from systemone_vision_test.services.jev_client import (
    DANGER_ZONE_IMAGE_HEADER,
    JevClient,
    api_model_name,
    data_uri,
    load_api_key,
)

#: 想定どおりの応答（接続確認と方向判定の両方のキーを含む）
GOOD_RESPONSE = {
    "answers": {
        "check_color": {"choice": "red"},
        "landolt_c_direction": {
            "choice": "right",
            "probabilities": {"up": 0.01, "down": 0.02, "left": 0.03, "right": 0.94},
            "confidence": 0.94,
        },
    }
}

#: テストで使うダミーの API キー（実物は使わない）
DUMMY_KEY = "test-key-" + "a" * 20

#: OpenRouter 経由の接続先（同じ System One API）
OPENROUTER_URL = "https://openrouter.ai/api/v1/systemone"


#: モデル一覧（``GET /v1/models``）の応答
MODELS_RESPONSE: dict[str, Any] = {"models": [{"name": "jev-latest"}, {"name": "jev-1.13.0"}]}


class _FakeResponse:
    """``requests.Response`` の最小限の代替。"""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """常に成功扱い。"""

    def json(self) -> dict[str, Any]:
        """固定の JSON を返す。"""
        return self._payload


class FakeSession:
    """``requests.Session`` の代替。送信内容（ヘッダ含む）を記録する。

    Args:
        payload: ``POST`` が返す JSON。
        error: 送信時に発生させる例外。
        status_code: ``GET``（モデル一覧）が返すステータス（200 以外は接続 NG の模擬）。
        models: ``GET`` が返す JSON。
    """

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
        status_code: int = 200,
        models: dict[str, Any] | None = None,
    ):
        self.payload = payload if payload is not None else GOOD_RESPONSE
        self.error = error
        self.status_code = status_code
        self.models = models if models is not None else MODELS_RESPONSE
        self.calls: list[dict[str, Any]] = []
        self.trust_env = False

    def get(self, url, timeout=None, headers=None):
        """モデル一覧の取得を記録して固定の応答を返す。"""
        self.calls.append({"method": "GET", "url": url, "timeout": timeout, "headers": headers})
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.models, self.status_code)

    def post(self, url, json=None, timeout=None, headers=None):
        """送信内容を記録して固定の応答を返す。"""
        self.calls.append(
            {"method": "POST", "url": url, "json": json, "timeout": timeout, "headers": headers}
        )
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.payload)


@pytest.fixture
def session() -> FakeSession:
    """フェイクセッション。"""
    return FakeSession()


@pytest.fixture
def client(session: FakeSession) -> JevClient:
    """API キーと接続先を渡したクライアント。"""
    return JevClient(
        base_url="https://api.example.test/v1/systemone",
        credential=DUMMY_KEY,
        session=session,
    )


# ---------------------------------------------------------------------------
# 設定（API キーの置き場所）
# ---------------------------------------------------------------------------


def test_api_key_prefers_the_environment(monkeypatch, tmp_path):
    """環境変数があればキーファイルより優先する。"""
    key_file = tmp_path / ".jev_api_key"
    key_file.write_text("file-key\n", encoding="utf-8")

    from_file = load_api_key(environ={}, path=key_file)
    from_env = load_api_key(environ={config.JEV_API_KEY_ENV: "env-key"}, path=key_file)

    assert from_file == "file-key"
    assert from_env == "env-key"


def test_api_key_file_skips_blank_and_comment_lines(tmp_path):
    """キーファイルは空行とコメント行を無視し、最初の値を使う。"""
    path = tmp_path / ".jev_api_key"
    path.write_text("# メモ\n\nsecret-value\n", encoding="utf-8")

    assert load_api_key(environ={}, path=path) == "secret-value"


def test_api_key_is_empty_without_environment_or_file(tmp_path):
    """環境変数もファイルも無ければ空（未設定として扱う）。"""
    assert load_api_key(environ={}, path=tmp_path / "missing") == ""


def test_unconfigured_client_reports_ng_without_network(session):
    """接続先と API キーが無いときは通信せずに NG を返す。"""
    client = JevClient(base_url="", credential="", session=session)

    assert client.configured is False
    assert client.check_connection() is False
    assert session.calls == []  # 通信していない


def test_default_settings_come_from_config(session):
    """既定では config の jev 設定（接続先・モデル・タイムアウト）を使う。"""
    client = JevClient(credential=DUMMY_KEY, session=session)

    assert client.base_url == config.JEV_BASE_URL
    assert client.model == config.JEV_MODEL
    assert client.timeout_sec == config.JEV_REQUEST_TIMEOUT_SEC


def test_online_client_keeps_environment_proxies():
    """インターネット経由なので、環境のプロキシ設定は使う（ローカルの clef は使わない）。"""
    assert JevClient(credential=DUMMY_KEY).session.trust_env is True
    assert ClefClient().session.trust_env is False


# ---------------------------------------------------------------------------
# リクエスト（認証ヘッダとボディ）
# ---------------------------------------------------------------------------


def test_auth_header_carries_the_api_key(client, session):
    """API キーは既定で ``Authorization: Bearer`` で送る。"""
    client.check_connection()

    assert session.calls[-1]["headers"] == {"Authorization": f"Bearer {DUMMY_KEY}"}


def test_auth_header_format_is_configurable(monkeypatch, session):
    """ヘッダ名と書式は config で変えられる（API 仕様の差を吸収する）。"""
    monkeypatch.setattr(config, "JEV_AUTH_HEADER", "X-API-Key")
    monkeypatch.setattr(config, "JEV_AUTH_SCHEME", "{key}")
    client = JevClient(base_url="https://api.example.test", credential=DUMMY_KEY, session=session)

    client.check_connection()

    assert session.calls[-1]["headers"] == {"X-API-Key": DUMMY_KEY}


def test_payload_uses_the_jev_assets_shape(client, session):
    """画像は ``assets`` に置き、参照名を state に書く（jev の仕様）。"""
    client.inspect_landolt_c("BASE64-IMAGE")

    payload = session.calls[-1]["json"]
    reference = config.JEV_IMAGE_ASSET_REFERENCE
    assert payload["model"] == config.JEV_MODEL
    assert "images" not in payload  # ローカルの clef の形は送らない
    assert payload["assets"] == {
        reference: {
            "type": "image",
            "format": config.JEV_IMAGE_FORMAT,
            "data_base64": "BASE64-IMAGE",
        }
    }
    assert payload["state"] == {"image": reference, "text": "視力検査用のランドルト環の画像"}
    question = payload["questions"]["landolt_c_direction"]
    assert question["type"] == "choice"
    assert question["criteria"]["right"] == "右の部分が開いている"


def test_images_require_the_danger_zone_header(client, session):
    """画像を含むリクエストには ``TypeSafe-Danger-Zone-Images: true`` を付ける。"""
    client.inspect_landolt_c("BASE64-IMAGE")
    image_headers = session.calls[-1]["headers"]

    client.check_connection()
    plain_headers = session.calls[-1]["headers"]

    assert image_headers[DANGER_ZONE_IMAGE_HEADER] == "true"
    assert DANGER_ZONE_IMAGE_HEADER not in plain_headers


def test_models_url_is_derived_from_the_endpoint():
    """接続確認に使うモデル一覧の URL は、接続先から組み立てる。"""
    systemone = JevClient(base_url="https://api.typesafe.ai/v1/systemone", credential=DUMMY_KEY)
    root = JevClient(base_url="https://api.typesafe.ai", credential=DUMMY_KEY)

    assert systemone.models_url == "https://api.typesafe.ai/v1/models"
    assert root.models_url == "https://api.typesafe.ai/models"


# ---------------------------------------------------------------------------
# OpenRouter 経由（同じ System One API を提供している）
# ---------------------------------------------------------------------------


def test_provider_is_inferred_from_the_endpoint(monkeypatch):
    """接続先の URL から種類（typesafe / openrouter）を推定する。"""
    monkeypatch.setattr(config, "JEV_PROVIDER", "")

    assert JevClient(base_url=OPENROUTER_URL, credential=DUMMY_KEY).provider == "openrouter"
    typesafe = JevClient(base_url="https://api.typesafe.ai/v1/systemone", credential=DUMMY_KEY)
    assert typesafe.provider == "typesafe"


def test_provider_can_be_set_explicitly(monkeypatch):
    """``JEV_PROVIDER`` で明示できる（接続先がプロキシでも指定できる）。"""
    monkeypatch.setattr(config, "JEV_PROVIDER", "openrouter")

    client = JevClient(base_url="https://example.test/v1/systemone", credential=DUMMY_KEY)

    assert client.provider == "openrouter"


def test_openrouter_uses_the_same_endpoint_layout(monkeypatch):
    """OpenRouter でも URL の組み立てと認証は同じ（変えるのは URL とキーだけ）。"""
    monkeypatch.setattr(config, "JEV_PROVIDER", "")
    client = JevClient(base_url=OPENROUTER_URL, credential=DUMMY_KEY)

    assert client.models_url == "https://openrouter.ai/api/v1/models"
    assert client.auth_headers() == {"Authorization": f"Bearer {DUMMY_KEY}"}


def test_openrouter_skips_the_model_list_check(monkeypatch, session, caplog):
    """OpenRouter の一覧は名前空間が違うので、照合せずに接続 OK とする。"""
    monkeypatch.setattr(config, "JEV_PROVIDER", "")
    session.models = {"data": [{"id": "typesafe/jev-1.13"}, {"id": "openai/gpt-sol"}]}
    client = JevClient(base_url=OPENROUTER_URL, credential=DUMMY_KEY, session=session)

    with caplog.at_level(logging.WARNING, logger="systemone_vision_test.services.jev_client"):
        assert client.check_connection() is True

    assert "一覧にありません" not in caplog.text


def test_openrouter_payload_puts_the_image_in_the_state_array(monkeypatch, session):
    """OpenRouter は state を配列にして ``image_url``（data URI）で画像を渡す。"""
    monkeypatch.setattr(config, "JEV_PROVIDER", "")
    client = JevClient(base_url=OPENROUTER_URL, credential=DUMMY_KEY, session=session)

    client.inspect_landolt_c("BASE64-IMAGE")

    payload = session.calls[-1]["json"]
    assert "assets" not in payload
    assert payload["state"] == [
        {"type": "text", "text": "視力検査用のランドルト環の画像"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64-IMAGE"}},
    ]
    assert "landolt_c_direction" in payload["questions"]
    # 画像は state 配列で渡すので Danger Zone ヘッダは不要（付かない）
    assert DANGER_ZONE_IMAGE_HEADER not in session.calls[-1]["headers"]


def test_data_uri_uses_the_configured_format(monkeypatch):
    """data URI の MIME タイプは ``JEV_IMAGE_FORMAT`` に従う。"""
    monkeypatch.setattr(config, "JEV_IMAGE_FORMAT", "jpg")

    assert data_uri("BASE64") == "data:image/jpeg;base64,BASE64"


def test_selector_name_is_mapped_to_the_api_model(monkeypatch):
    """``--models jev`` のような選択名は API のモデル名に変換して送る。"""
    monkeypatch.setattr(config, "JEV_MODEL", "jev-latest")

    assert api_model_name(None) == "jev-latest"
    assert api_model_name("jev") == "jev-latest"  # 選択名（そのまま送ると typesafe/jev になる）
    assert api_model_name("typesafe/jev-1.13") == "typesafe/jev-1.13"


def test_client_sends_the_api_model_name(monkeypatch, session):
    """選択名で作ったクライアントでも、リクエストには API のモデル名が載る。"""
    monkeypatch.setattr(config, "JEV_MODEL", "jev-latest")
    client = JevClient(base_url=OPENROUTER_URL, model="jev", credential=DUMMY_KEY, session=session)

    client.inspect_landolt_c("BASE64-IMAGE")

    assert client.model == "jev-latest"
    assert session.calls[-1]["json"]["model"] == "jev-latest"


def test_factory_maps_the_online_selector_to_the_api_model():
    """統計バッチの ``--models jev`` でも、API には既定のモデル名で問い合わせる。"""
    assert factory.create_client("jev").model == config.JEV_MODEL


def test_danger_zone_header_can_be_disabled(monkeypatch, client, session):
    """``JEV_SEND_DANGER_ZONE_HEADER`` を切ると、画像入りでもヘッダを付けない。"""
    monkeypatch.setattr(config, "JEV_SEND_DANGER_ZONE_HEADER", False)

    client.inspect_landolt_c("BASE64-IMAGE")

    assert DANGER_ZONE_IMAGE_HEADER not in session.calls[-1]["headers"]


def test_timeout_uses_the_jev_default(session):
    """タイムアウトは jev 用の設定値を使う。"""
    client = JevClient(base_url="https://api.example.test", credential=DUMMY_KEY, session=session)

    client.check_connection()

    assert session.calls[-1]["timeout"] == config.JEV_REQUEST_TIMEOUT_SEC


# ---------------------------------------------------------------------------
# 応答の解釈（API 仕様の差の吸収）
# ---------------------------------------------------------------------------


def test_check_connection_uses_the_models_endpoint_and_systemone(client, session):
    """接続確認は ``GET``（モデル一覧）と System One への 1 回の両方を行う。"""
    assert client.check_connection() is True

    assert [call["method"] for call in session.calls] == ["GET", "POST"]
    assert session.calls[0]["url"] == "https://api.example.test/v1/models"
    assert session.calls[0]["headers"] == {"Authorization": f"Bearer {DUMMY_KEY}"}
    assert "check_color" in session.calls[1]["json"]["questions"]


def test_check_connection_false_when_the_models_endpoint_fails(client, session):
    """モデル一覧が 200 以外なら接続 NG（System One は呼ばない）。"""
    session.status_code = 401

    assert client.check_connection() is False
    assert [call["method"] for call in session.calls] == ["GET"]


@pytest.mark.parametrize(
    "payload",
    [pytest.param({}, id="empty"), pytest.param({"answers": {}}, id="no-answers")],
)
def test_check_connection_false_without_answers(client, session, payload):
    """System One の応答に ``answers`` が無ければ接続 NG。"""
    session.payload = payload

    assert client.check_connection() is False


def test_model_outside_the_list_only_warns(client, session, caplog):
    """モデル一覧に無い名前でも、警告だけ出して接続は OK とする。"""
    session.models = {"models": [{"name": "jev-1.14.0"}]}

    with caplog.at_level(logging.WARNING, logger="systemone_vision_test.services.jev_client"):
        assert client.check_connection() is True

    assert config.JEV_MODEL in caplog.text
    assert "jev-1.14.0" in caplog.text


def test_communication_errors_are_reported_as_ng():
    """通信エラーは NG（例外を投げない）。"""
    client = JevClient(
        base_url="https://api.example.test",
        credential=DUMMY_KEY,
        session=FakeSession(error=requests.ConnectionError("接続不可")),
    )

    assert client.check_connection() is False
    assert client.inspect_landolt_c("BASE64-IMAGE") is None


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(GOOD_RESPONSE, id="systemone"),
        pytest.param(
            {"answers": {"landolt_c_direction": {"answer": "right"}}}, id="answers-answer"
        ),
        pytest.param({"answer": {"landolt_c_direction": {"value": "right"}}}, id="answer-value"),
        pytest.param({"choices": [{"message": {"content": "right"}}]}, id="openai-content"),
        pytest.param(
            {"choices": [{"message": {"content": '{"choice": "right"}'}}]}, id="openai-json"
        ),
        pytest.param({"result": {"choice": "RIGHT."}}, id="result-uppercase"),
        pytest.param({"choice": "右"}, id="japanese"),
        pytest.param({"output": "切れ目は右に開いています"}, id="sentence"),
    ],
)
def test_inspect_absorbs_response_shapes(client, session, payload):
    """応答の形が違っても、回答を取り出して質問のキーに寄せる。"""
    session.payload = payload

    answer = client.inspect_landolt_c("BASE64-IMAGE")

    assert answer is not None
    assert answer.choice == "right"


def test_inspect_keeps_probabilities_and_confidence(client):
    """確率と確信度があればそのまま記録する。"""
    answer = client.inspect_landolt_c("BASE64-IMAGE")

    assert answer is not None
    assert answer.probabilities["right"] == pytest.approx(0.94)
    assert answer.confidence == pytest.approx(0.94)


def test_inspect_works_without_probabilities(client, session):
    """確率が無い応答でも動く（確率は空、確信度は 0）。"""
    session.payload = {"answers": {"landolt_c_direction": {"choice": "left"}}}

    answer = client.inspect_landolt_c("BASE64-IMAGE")

    assert answer is not None
    assert answer.choice == "left"
    assert answer.probabilities == {}
    assert answer.confidence == 0.0


def test_confidence_falls_back_to_the_max_probability(client, session):
    """``confidence`` が無ければ確率の最大値を使う。"""
    session.payload = {
        "answers": {
            "landolt_c_direction": {"choice": "up", "probabilities": {"up": 0.7, "down": 0.3}}
        }
    }

    answer = client.inspect_landolt_c("BASE64-IMAGE")

    assert answer is not None
    assert answer.confidence == pytest.approx(0.7)


def test_logprobs_style_probabilities_are_read():
    """``logprobs`` 形式（``{キー: {"probability": 値}}``）も読む。"""
    payload = {
        "answers": {
            "landolt_c_direction": {
                "choice": "up",
                "logprobs": {"up": {"probability": 0.8}, "down": {"probability": 0.2}},
            }
        }
    }

    answer = extract_answer(payload, "landolt_c_direction")

    assert answer is not None
    assert answer.probabilities == {"up": 0.8, "down": 0.2}


def test_unknown_answer_is_kept_as_is(client, session):
    """方向として読めない回答はそのまま残す（未対応として集計できる）。"""
    session.payload = {"answers": {"landolt_c_direction": {"choice": "分かりません"}}}

    answer = client.inspect_landolt_c("BASE64-IMAGE")

    assert answer is not None
    assert answer.choice == "分かりません"
    assert DIRECTION_QUESTION.direction_of(answer.choice) is None


def test_missing_answer_returns_none(client, session):
    """回答が取り出せない応答は ``None``。"""
    session.payload = {"usage": {"prompt_tokens": 10}}

    assert client.inspect_landolt_c("BASE64-IMAGE") is None


def test_answers_are_mapped_to_the_question_keys():
    """方向で答えた場合も、質問のキー（a/b/c/d など）に寄せる。"""
    assert answer_key_for(DIRECTION_QUESTION, "右") == "right"
    assert answer_key_for(NEUTRAL_KEY_QUESTION, "right") == "d"
    assert answer_key_for(NEUTRAL_KEY_QUESTION, "D") == "d"
    assert answer_key_for(NEUTRAL_KEY_QUESTION, "分かりません") == "分かりません"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("右", "right"), ("LEFT", "left"), ("上側が開いている", "up"), ("west", "left"), ("?", None)],
)
def test_direction_aliases(text, expected):
    """日本語・方位・文中の語から方向を判定できる。"""
    assert direction_of_text(text) == expected


# ---------------------------------------------------------------------------
# モデルの振り分け（GUI・統計バッチ）
# ---------------------------------------------------------------------------


def test_online_models_are_distinguished():
    """jev だけがインターネット経由として扱われる。"""
    assert factory.is_online_model("jev") is True
    assert factory.is_online_model("clef") is False
    assert factory.is_online_model("clef-flash") is False


def test_factory_creates_the_right_client_for_each_model():
    """ローカルは ClefClient、jev は JevClient を作る。"""
    assert type(factory.create_client("clef")) is ClefClient
    assert type(factory.create_client("clef-flash")) is ClefClient
    assert isinstance(factory.create_client("jev"), JevClient)


def test_factory_uses_the_configured_endpoint_for_online_models():
    """jev の接続先は config（環境変数）から取る。"""
    assert factory.create_client("jev").base_url == config.JEV_BASE_URL


def test_benchmark_factory_does_not_leak_the_local_url_to_jev():
    """統計バッチの ``--base-url`` はローカルのモデルにだけ効く。"""
    args = cli.parse_args(["--models", "jev", "--base-url", "http://127.0.0.1:9999/v1/systemone"])

    local = cli.build_client_factory(args)("clef")
    online = cli.build_client_factory(args)("jev")

    assert local.base_url == "http://127.0.0.1:9999/v1/systemone"
    assert isinstance(online, JevClient)
    assert online.base_url == config.JEV_BASE_URL


def test_cli_hints_the_jev_settings_when_the_connection_fails(monkeypatch, capsys, tmp_path):
    """jev を含む実行で接続できないとき、設定先の案内を出す。"""

    class FailingRunner:
        """接続エラーを投げるだけのランナー。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run(self) -> None:
            raise ConnectionError("接続できないモデルがあります: jev")

    monkeypatch.setattr(cli, "BenchmarkRunner", FailingRunner)

    exit_code = cli.main(
        [
            "--models",
            "jev",
            "--modes",
            "清明",
            "--visions",
            "0.2",
            "--repeats",
            "1",
            "--output",
            str(tmp_path / "report.md"),
            "--no-raw",
            "--no-state",
            "--quiet",
        ]
    )

    printed = capsys.readouterr().out
    assert exit_code == 1
    assert config.JEV_API_KEY_ENV in printed
    assert "JEV_BASE_URL" in printed
    assert config.JEV_API_KEY_FILE.name in printed


def test_report_endpoint_mentions_jev_when_it_is_in_the_plan():
    """jev を含む実行では、レポートの接続先に jev 側も示す。"""
    local_url = "http://127.0.0.1:11434/v1/systemone"

    assert cli.report_endpoint(BenchmarkPlan(models=("clef",)), local_url) == local_url
    mixed = cli.report_endpoint(BenchmarkPlan(models=("clef", "jev")), local_url)
    only_jev = cli.report_endpoint(BenchmarkPlan(models=("jev",)), local_url)

    expected = f"JEV_BASE_URL: {config.JEV_BASE_URL} / model: {config.JEV_MODEL}"
    assert mixed == f"{local_url} / {expected}"
    assert only_jev == expected  # ローカルを使わないので出さない
