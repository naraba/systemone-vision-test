# サードパーティライセンス

本ソフトウェア本体は **MIT ライセンス**（[LICENSE](LICENSE)）で公開している。以下は、本ソフトウェアが
利用するサードパーティ製ソフトウェアと、その利用にあたっての注意点である。これらは本リポジトリには
含まれておらず、利用者が各自のライセンスに従って導入する。

| パッケージ | 用途 | ライセンス | 配布元 |
|---|---|---|---|
| PySide6（Qt for Python）・shiboken6・Qt | GUI・画像表示・音声再生 | LGPL-3.0-only（または GPL-2.0-only / GPL-3.0-only） | <https://pypi.org/project/PySide6/> |
| opencv-python（OpenCV） | 画像の生成・加工 | Apache-2.0 | <https://pypi.org/project/opencv-python/> |
| NumPy | 数値計算 | BSD-3-Clause | <https://pypi.org/project/numpy/> |
| Requests | System One API（clef / jev）との HTTP 通信 | Apache-2.0 | <https://pypi.org/project/requests/> |

## PySide6 / Qt（LGPL-3.0）の扱い

* 本ソフトウェアは PySide6 を **動的リンク**（Python の ``import``）で利用しており、Qt / PySide6 自体は
  改変していない。LGPL-3.0 は「動的リンクで使う限り、利用するアプリのライセンスを自由にできる」ため、
  本体を MIT で公開できる。
* **ソースコードのみを公開する場合**、上記以外に LGPL-3.0 が課す義務は生じない。
* **実行ファイル（PyInstaller 等）やインストーラとして配布する場合**は、次が必要になる。
  1. LGPL-3.0 の全文と、Qt / PySide6 の著作権表示の同梱
  2. 同梱したライブラリを利用者が差し替えられる形（分離した共有ライブラリとして配置する等）での提供
  3. Apache-2.0 の依存（OpenCV・Requests）を同梱する場合は、それぞれのライセンス文の同梱

## モデル

本ソフトウェアはローカルの Ollama 上の `clef` / `clef-flash` モデルを API 経由で利用する。
インターネット経由の `jev`（TypeSafe / OpenRouter のホスト型 System One）を使う場合も、API 経由で
呼び出すだけで重みは扱わない。**モデルの重みは含まない**ため、モデルそのもののライセンスは
本リポジトリとは無関係である（`jev` の利用条件は各サービスの規約に従う）。
