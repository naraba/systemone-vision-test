"""追跡対象ファイルに、シークレットや個人環境に依存する情報が無いか確認する。

``.gitignore`` を自前で解釈し、**git 管理対象になるファイル**だけを対象に次を検出する。

* シークレット: API キー・アクセストークン・パスワード・メールアドレス
* 個人環境に依存する情報: 絶対パス（Windows のドライブ / Unix のホーム）・
  loopback とドキュメント用予約域以外の IPv4 アドレス・マシン名

検出があれば終了コード 1（CI でも使える）。値の形（``key = "..."``）だけを拾うため、
コード中の語彙（変数名としての ``password`` など）は検出しない。テキストとして読めない
ファイル（画像・音声など）は対象外なので、バイナリに埋め込まれた情報は別途確認すること。

使い方::

    python scripts/check_tracked_info.py
    python scripts/check_tracked_info.py --root .
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path

# src レイアウトのため、未インストールでも実行できるようにする簡易ブートストラップ
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

#: プロジェクトのルート
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 環境に依存しないとみなすアドレス。``.`` で終わるものは前方一致の接頭辞
#: （192.0.2.0/24 などはドキュメント用に予約された範囲）
IGNORED_ADDRESSES: tuple[str, ...] = (
    "127.0.0.1",
    "0.0.0.0",
    "192.0.2.",
    "198.51.100.",
    "203.0.113.",
)

#: 検出するパターン（ラベル, 正規表現）
CHECKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("メールアドレス", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")),
    (
        "シークレットの代入",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|password|passwd|access[_-]?token|auth[_-]?token)"
            r"\b\s*[:=]\s*\S"
        ),
    ),
    ("ベアラトークン", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}")),
    (
        "既知のキー形式",
        re.compile(
            r"sk-[A-Za-z0-9]{16,}"
            r"|ghp_[A-Za-z0-9]{20,}"
            r"|AKIA[0-9A-Z]{16}"
            r"|xox[baprs]-[A-Za-z0-9-]{10,}"
        ),
    ),
    ("絶対パス", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?![\\/])")),
    ("UNC パス", re.compile(r"\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9$_.-]+")),
    ("ホーム/システムパス", re.compile(r"(?<![\w.])/(home|Users|mnt|opt|var|usr)/")),
    ("IPv4 アドレス", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("マシン名", re.compile(r"(?i)\b(DESKTOP|LAPTOP|WIN)-[A-Z0-9]{4,}\b")),
)


def is_ignored_address(address: str) -> bool:
    """環境に依存しないとみなせるアドレスかどうか。

    Args:
        address: 判定する IPv4 アドレス。

    Returns:
        loopback か、ドキュメント用に予約された範囲なら ``True``。
    """
    return any(
        address.startswith(item) if item.endswith(".") else address == item
        for item in IGNORED_ADDRESSES
    )


def load_gitignore_rules(root: Path) -> list[tuple[str, bool]]:
    """``.gitignore`` を ``(パターン, 打ち消しかどうか)`` の並びに変換する。

    Args:
        root: ``.gitignore`` を探すディレクトリ。

    Returns:
        記述順の規則。``!`` で始まる行は打ち消し（``True``）。
    """
    path = root / ".gitignore"
    if not path.is_file():
        return []

    rules: list[tuple[str, bool]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        rules.append((line[1:] if negate else line, negate))
    return rules


def matches(relative: str, pattern: str) -> bool:
    """git のパターン 1 つが相対パスに一致するかどうか。

    Args:
        relative: ``/`` 区切りの相対パス。
        pattern: ``.gitignore`` の 1 行。

    Returns:
        一致すれば ``True``。``/`` を含まないパターンはどの階層の名前にも一致し、
        ``/`` で終わるパターンは途中のディレクトリ名にも一致する。
    """
    if pattern.endswith("/"):
        head = pattern[:-1]
        return any(fnmatch.fnmatch(part, head) for part in relative.split("/"))
    if "/" in pattern:
        return fnmatch.fnmatch(relative, pattern)
    return any(fnmatch.fnmatch(part, pattern) for part in relative.split("/"))


def is_ignored(relative: str, rules: list[tuple[str, bool]]) -> bool:
    """``.gitignore`` の規則（後勝ち・``!`` で打ち消し）を適用する。

    Args:
        relative: ``/`` 区切りの相対パス。
        rules: :func:`load_gitignore_rules` の結果。

    Returns:
        git の管理対象から外れるなら ``True``。
    """
    ignored = False
    for pattern, negate in rules:
        if matches(relative, pattern):
            ignored = not negate
    return ignored


def iter_tracked_files(root: Path) -> list[Path]:
    """``.gitignore`` を適用して、git 管理対象になるファイルを返す。

    無視されるディレクトリには降りない（``.venv`` などで遅くならないように）。

    Args:
        root: 探索するディレクトリ。

    Returns:
        対象ファイルの並び。
    """
    rules = load_gitignore_rules(root)
    tracked: list[Path] = []

    for directory, dirnames, filenames in os.walk(root):
        base = Path(directory)
        relative_dir = "" if base == root else base.relative_to(root).as_posix()

        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if name != ".git" and not is_ignored(f"{relative_dir}/{name}".lstrip("/"), rules)
        ]
        tracked.extend(
            base / name
            for name in sorted(filenames)
            if not is_ignored(f"{relative_dir}/{name}".lstrip("/"), rules)
        )
    return tracked


def find_findings(path: Path) -> list[tuple[int, str, str]]:
    """1 ファイルを検査する。

    Args:
        path: 検査するファイル。テキストとして読めない場合は空を返す。

    Returns:
        ``(行番号, ラベル, 行の内容)`` の並び。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    findings: list[tuple[int, str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in CHECKS:
            for match in pattern.finditer(line):
                if label == "IPv4 アドレス" and is_ignored_address(match.group()):
                    continue
                findings.append((number, label, line.strip()))
    return findings


def scan(root: Path) -> list[tuple[Path, int, str, str]]:
    """追跡対象ファイルをすべて検査する。

    Args:
        root: プロジェクトのルート。

    Returns:
        ``(ファイル, 行番号, ラベル, 行の内容)`` の並び。
    """
    return [
        (path, number, label, line)
        for path in iter_tracked_files(root)
        for number, label, line in find_findings(path)
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解釈する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="調べるディレクトリ（既定: %(default)s）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """チェックを実行する。

    Args:
        argv: コマンドライン引数（``None`` なら ``sys.argv``）。

    Returns:
        検出があれば 1、無ければ 0。
    """
    args = parse_args(argv)
    root: Path = args.root
    tracked = iter_tracked_files(root)
    print(f"対象: {len(tracked)} ファイル（git 管理対象のみ）")

    results = scan(root)
    for path, number, label, line in results:
        print(f"  {path.relative_to(root).as_posix()}:{number} [{label}] {line[:100]}")

    if results:
        print(f"\n検出 {len(results)} 件。シークレットや個人環境に依存する情報を見直してください。")
        return 1

    print("検出なし（シークレット・個人環境に依存する情報は見つかりませんでした）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
