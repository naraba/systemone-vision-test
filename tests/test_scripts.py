"""scripts/ の補助スクリプトのテスト（環境に依存しない）。

出力先は ``tmp_path`` に逃がすため、リポジトリの ``image/`` は変更しない。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from systemone_vision_test import config
from systemone_vision_test.domain.landolt import GenerationMode

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str) -> Any:
    """``scripts/<name>.py`` をモジュールとして読み込む。"""
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generate_script():
    """画像一括生成スクリプトのモジュール。"""
    return load_script("generate_all_images")


# ---------------------------------------------------------------------------
# generate_all_images.py
# ---------------------------------------------------------------------------


def test_generate_all_images_writes_every_mode_and_vision(generate_script, tmp_path):
    """全モード・全視力の PNG を出力する。"""
    exit_code = generate_script.main(["--output", str(tmp_path), "--seed", "1"])

    assert exit_code == 0
    for mode in GenerationMode:
        files = sorted((tmp_path / mode.value).glob("*.png"))
        assert len(files) == len(config.VISION_LEVELS)
        assert files[0].name.startswith(f"landolt_{mode.value}_")
        assert files[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_all_images_accepts_arguments(generate_script, tmp_path):
    """引数の解釈（``--output`` と ``--seed``）ができる。"""
    args = generate_script.parse_args(["--output", str(tmp_path), "--seed", "7"])

    assert args.output == tmp_path
    assert args.seed == 7


def test_generate_all_images_is_reproducible_with_seed(generate_script, tmp_path):
    """同じシードなら同じ画像が生成される（切れ目の方向も再現される）。"""
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_script.main(["--output", str(first), "--seed", "42"])
    generate_script.main(["--output", str(second), "--seed", "42"])

    for path in first.glob("清明/*.png"):
        other = second / "清明" / path.name
        assert path.read_bytes() == other.read_bytes()


# ---------------------------------------------------------------------------
# check_tracked_info.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def audit_script():
    """追跡対象の情報チェックスクリプトのモジュール。"""
    return load_script("check_tracked_info")


def test_check_tracked_info_applies_gitignore(audit_script, tmp_path):
    """``.gitignore`` を解釈して、無視されるファイルを対象から外す。"""
    (tmp_path / ".gitignore").write_text("artifacts/\nsound/*\n!sound/.gitkeep\n", encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "out.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sound").mkdir()
    (tmp_path / "sound" / "right.mp3").write_bytes(b"")
    (tmp_path / "sound" / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / "main.py").write_text("print(1)", encoding="utf-8")

    tracked = {
        path.relative_to(tmp_path).as_posix() for path in audit_script.iter_tracked_files(tmp_path)
    }

    assert tracked == {".gitignore", "main.py", "sound/.gitkeep"}


def test_check_tracked_info_ignores_api_key_files(audit_script, tmp_path):
    """API キーの置き場所（.env / .jev_api_key / secrets/）は追跡対象から外れる。"""
    (tmp_path / ".gitignore").write_text(".env\n.jev_api_key\nsecrets/\n", encoding="utf-8")
    # このテスト自身が検出されないよう、値は分割して組み立てる
    (tmp_path / ".env").write_text("API" + "_KEY=value\n", encoding="utf-8")
    (tmp_path / ".jev_api_key").write_text("secret-value\n", encoding="utf-8")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "key.txt").write_text("secret-value\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print(1)\n", encoding="utf-8")

    tracked = {
        path.relative_to(tmp_path).as_posix() for path in audit_script.iter_tracked_files(tmp_path)
    }

    assert tracked == {".gitignore", "main.py"}


def test_repository_gitignore_excludes_api_key_files(audit_script):
    """このリポジトリの ``.gitignore`` が、API キーの置き場所を実際に除外している。"""
    rules = audit_script.load_gitignore_rules(audit_script.PROJECT_ROOT)

    assert audit_script.is_ignored(".jev_api_key", rules) is True
    assert audit_script.is_ignored(".env", rules) is True
    assert audit_script.is_ignored("src/systemone_vision_test/config.py", rules) is False


def test_check_tracked_info_finds_secrets_and_paths(audit_script, tmp_path):
    """シークレット・絶対パス・メールアドレスを検出する。

    このテスト自身が検出されないよう、値は分割して組み立てる。
    """
    path = tmp_path / "sample.py"
    quote = chr(34)
    path.write_text(
        "\n".join(
            [
                "KEY = " + quote + "sk-" + "a" * 24 + quote,
                "PATH = " + quote + "C:" + "\\Users\\someone\\work" + quote,
                "MAIL = " + quote + "someone" + chr(64) + "example.com" + quote,
            ]
        ),
        encoding="utf-8",
    )

    labels = {label for _, label, _ in audit_script.find_findings(path)}

    assert {"既知のキー形式", "絶対パス", "メールアドレス"} <= labels


def test_check_tracked_info_allows_loopback_and_documentation_addresses(audit_script, tmp_path):
    """loopback とドキュメント用の予約アドレスは検出しない。"""
    path = tmp_path / "sample.py"
    quote = chr(34)
    path.write_text(
        "\n".join(
            [
                "URL = " + quote + "http://127.0.0.1:11434/" + quote,
                "DOC = " + quote + "http://192.0.2.1/" + quote,
            ]
        ),
        encoding="utf-8",
    )

    assert audit_script.find_findings(path) == []


def test_check_tracked_info_finds_private_address(audit_script, tmp_path):
    """個人環境らしきプライベートアドレスは検出する。"""
    path = tmp_path / "sample.py"
    quote = chr(34)
    path.write_text("HOST = " + quote + "192.168." + "0.10" + quote + "\n", encoding="utf-8")

    labels = {label for _, label, _ in audit_script.find_findings(path)}

    assert "IPv4 アドレス" in labels


# ---------------------------------------------------------------------------
# show_jev_config.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jev_config_script():
    """jev の設定表示スクリプトのモジュール。"""
    return load_script("show_jev_config")


def test_show_jev_config_prints_settings_without_the_key(
    jev_config_script, capsys, monkeypatch, tmp_path
):
    """設定の解決結果を表示し、API キーの値そのものは出さない。"""
    key_file = tmp_path / ".jev_api_key"
    key_file.write_text("very-secret-value\n", encoding="utf-8")
    monkeypatch.setattr(config, "JEV_API_KEY_FILE", key_file)
    monkeypatch.setattr(config, "JEV_BASE_URL", "https://example.test/v1/systemone")
    monkeypatch.delenv(config.JEV_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(config.JEV_API_KEY_ENV, raising=False)

    exit_code = jev_config_script.main([])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "https://example.test/v1/systemone" in printed
    assert "typesafe" in printed  # example.test は OpenRouter ではないので typesafe 扱い
    assert "キーファイル .jev_api_key" in printed
    assert "very-secret-value" not in printed
    assert "--check" in printed


def test_repository_has_no_tracked_secrets_or_local_paths(audit_script):
    """このリポジトリの追跡対象ファイルに、シークレットや個人環境依存の情報が無い。"""
    results = audit_script.scan(audit_script.PROJECT_ROOT)

    assert [f"{path.name}:{number} [{label}]" for path, number, label, _ in results] == []
