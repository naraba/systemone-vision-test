"""jev（System One）の設定が今どう解決されるかを表示する。

**API キーの値は表示しない**（環境変数とキーファイルのどちらから読んだかだけを示す）。
``--check`` を付けると実際に接続確認まで行う。

使い方::

    python scripts/show_jev_config.py
    python scripts/show_jev_config.py --check
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# src レイアウトのため、未インストールでも実行できるようにする簡易ブートストラップ
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from systemone_vision_test import config
from systemone_vision_test.services.jev_client import DANGER_ZONE_IMAGE_HEADER, JevClient


def describe(client: JevClient) -> list[tuple[str, str]]:
    """表示する項目（名前, 値）を組み立てる。

    Args:
        client: 設定を読み込んだクライアント。

    Returns:
        ``(項目名, 値)`` の並び。API キーの値そのものは含めない。
    """
    key_file = config.JEV_API_KEY_FILE
    return [
        ("接続先", client.request_url or "(空)"),
        ("接続先の由来", client.base_url_source),
        ("種類（provider）", client.provider),
        ("モデル一覧の URL", client.models_url),
        ("モデル", client.model),
        ("認証ヘッダ", f"{config.JEV_AUTH_HEADER}: {config.JEV_AUTH_SCHEME}"),
        ("API キーの取得元", f"{client.credential_source}（値は表示しません）"),
        ("キーファイル", f"{key_file}（{'あり' if key_file.is_file() else 'なし'}）"),
        (
            "画像のヘッダ",
            f"{DANGER_ZONE_IMAGE_HEADER}: true を送る"
            if config.JEV_SEND_DANGER_ZONE_HEADER
            else f"{DANGER_ZONE_IMAGE_HEADER} は送らない",
        ),
        ("タイムアウト", f"{config.JEV_REQUEST_TIMEOUT_SEC} 秒"),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解釈する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="実際に接続確認する（API キーを使う。数トークン消費する）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """設定を表示する（``--check`` なら接続確認もする）。

    Args:
        argv: コマンドライン引数（``None`` なら ``sys.argv``）。

    Returns:
        接続確認が NG なら 1、それ以外は 0。
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = JevClient()

    rows = describe(client)
    width = max(len(name) for name, _ in rows)
    print("jev（System One）の設定:")
    for name, value in rows:
        print(f"  {name:<{width}} : {value}")

    if not args.check:
        print("\n（--check を付けると接続確認まで行います）")
        return 0

    ok = client.check_connection()
    print("\n接続確認:", "OK" if ok else "NG（上の WARNING に理由が出ます）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
