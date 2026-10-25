# System One による疑似視覚検査

System One のモデル（ローカルの `clef` / `clef-flash`、インターネット経由の `jev`）に
ランドルト環の画像を判定させ、見え方のシミュレーション（生成モード）ごとに「どこまで見えるか」を
測るプログラム。

仕様は [docs/spec.md](docs/spec.md) を参照。劣化フィルタの既定値の根拠は
[docs/filter_parameters.md](docs/filter_parameters.md)、方向回答の偏りの測定は
[docs/landolt_direction_bias.md](docs/landolt_direction_bias.md) にまとめている（一覧は
[docs/README.md](docs/README.md)）。

## 必要な環境

* Python 3.10 以上
* **System One 互換のエンドポイント**（既定はローカルの `http://localhost:11434/v1/systemone`）
  * 本リポジトリには**サービスの実装もモデルの重みも含まれていない**。ローカル推論サーバ
    （Ollama など）に System One 互換のエンドポイントを用意するか、インターネット経由の
    `jev` を使う（`--models jev`。この場合はローカルサービスは不要）
  * 判定させるモデル名は、そのエンドポイントが公開している名前を使う（既定は
    `clef` / `clef-flash`）。利用できる名前は `curl http://localhost:11434/v1/models` で確認できる。
    名前や接続先が違う場合は `--models <モデル名>` と `--base-url <URL>` で指定する
    （GUI の接続先は `config.CLEF_BASE_URL`）
* 動作確認済みの OS: Windows 11
* `jev`（インターネット経由）を使う場合のみ、接続先と API キーの設定が要る
  （[jev（TypeSafe の System One）を使う](#jevtypesafe-の-system-oneを使う) を参照）

## セットアップ

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# 本体＋開発用ツール（pytest / ruff）
pip install -e ".[dev]"
```

## 実行

```powershell
python -m systemone_vision_test
```

インストール後はコンソールスクリプトでも起動できる。

```powershell
systemone-vision-test
```

## 統計（GUI なしの自走バッチ）

GUI も効果音も使わない自走バッチで、**全モデル × 全モード × 全視力値 × 全方向**を同じ回数ずつ
呈示して正答率を集計し、Markdown レポートと生データ（JSON）を出力する。

```powershell
python -m systemone_vision_test.benchmark                      # 既定（2,496 呈示）
systemone-vision-benchmark                                     # インストール後はこれでも可
python -m systemone_vision_test.benchmark --models clef clef-flash   # ローカルだけで回す
python -m systemone_vision_test.benchmark --models clef-flash --modes 清明 --visions 2.0 --repeats 1
```

* 既定の呈示数は **モデル 3（`clef` / `clef-flash` / `jev`）× モード 4 × 視力値 13 × 方向 4 × 反復 4
  = 2,496 件**。画面表示が無いので判定ごとの待ち時間は入れず、連続して送信する
* 既定には **jev（OpenRouter 経由）**が入るため、実行には接続先と API キーの設定が要る
  （ローカルだけで回すときは `--models clef clef-flash` を指定する）
* jev を含めると実行時間と API 利用料がかかる（ローカルの 2 モデルだけなら無料）
* 出力は `artifacts/benchmark/benchmark_<日時>.md`（前提条件・モードごとのパラメータ・結果表）と、
  同じ名前の `.json`（呈示ごとの生データ。後から再集計できる）
* モデルごとに接続確認を 1 回行い、接続できないモデルがあれば実行を始めずに終了する
* **まとまり（モデル × モード）ごとに途中経過を `artifacts/benchmark/benchmark.state.json` に保存**し、
  途中で止めても同じコマンドで続きから再開できる（`--fresh` で最初から、`--no-state` で保存なし）
* 正答率は「モデル × モード × 視力値 × 開いている向き」でまとめ、回答が得られなかった呈示も
  分母に含める（件数はレポートの表に出る）
* `--question` で質問の種類を変えられる。**方向ごとの正答率が偏るときに、原因が
  「回答ラベルの事前分布」か「知覚の失敗」かを切り分ける**ために使う
  * `default`＝日本語の説明・英語のキー / `neutral`＝回答キーを a/b/c/d /
    `japanese`＝上・下・左・右 / `reversed`＝説明の順序を逆に
  * `swapped`＝説明文を逆向きに（キーと説明が食い違う）/ `closed`＝反対質問
    （切れ目が開いていない側を答える）
  * `compass`＝回答キーを north/south/west/east / `hiragana`＝うえ/した/ひだり/みぎ /
    `compass_shift`＝キーは同じで説明の対応だけをずらす / `compass_nogloss`＝補足文なし
* レポートには「回答の偏り（診断用）」として、**回答ラベルの分布・回答キーごとの平均確率・
  混同行列（提示した向き × 回答した向き）・確信度（最大確率）ごとの正答率**を載せる。
  方向別の正答率が偏ったときは、まずここを見る（低確信の呈示を「判定不能」として切り分けられる）

## jev（TypeSafe の System One）を使う

`clef` / `clef-flash` はローカルの jev 互換モデル。**本体の jev をインターネット越しに**使うこともできる
（仕様: [docs.typesafe.ai/images](https://docs.typesafe.ai/images) と
[OpenAPI](https://api.typesafe.ai/openapi.json)）。

1. **API キー**を [TypeSafe コンソール](https://console.typesafe.ai/keys)で発行し、次のどちらかで与える
   （**リポジトリには絶対に入れない**）
   * 環境変数（推奨。ファイルに残らない）

     ```powershell
     $env:JEV_API_KEY = "（発行されたキー）"
     ```

   * または プロジェクト直下の `.jev_api_key` にキーだけを 1 行書く（`.gitignore` 済みで `git status` に出ない）
2. 実行する（GUI ではモデルの選択肢に `jev` が出る）

   ```powershell
   python -m systemone_vision_test.benchmark --models jev --repeats 1 --visions 2.0 1.0 0.5
   ```

### OpenRouter 経由で使う（既定。設定する場所は同じ）

OpenRouter も TypeSafe 互換の SystemOne API（`https://openrouter.ai/api/v1/systemone`）を提供しているので、
**設定する場所はそのままで、値だけ**を変えればよい。接続先の既定が OpenRouter になっているので、
必要なのは API キー（OpenRouter のもの）だけ。

```powershell
$env:JEV_API_KEY = "（OpenRouter の API キー）"   # または .jev_api_key に 1 行（同じ場所・同じ名前）
# TypeSafe に直接つなぐ場合だけ: $env:JEV_BASE_URL = "https://api.typesafe.ai/v1/systemone"
# モデル名は既定の jev-latest のままでよい（OpenRouter が ~typesafe/jev-latest にマップする）
```

設定がどう効いているかは次のコマンドで確認できる（**API キーの値は表示しません**）。

```powershell
python scripts/show_jev_config.py            # 解決後の設定（接続先・モデル・キーの取得元）を表示
python scripts/show_jev_config.py --check    # 実際に接続確認まで行う
```

OpenRouter 経由のときの違い（`services/jev_client.py` が吸収します）:

* **画像は state を配列にして `image_url`（data URI）で渡す**。OpenRouter は TypeSafe の `assets` を
  受け付けず**無視する**ため（画像なしとして処理され、ほぼ当てずっぽうの回答になる）
* `--models jev` は**選択名**（レポート上の名前）。リクエストの `model` には `JEV_MODEL`
  （既定 `jev-latest`）を送る。`typesafe/jev-1.13` のような API のモデル名を指定した場合はそのまま使う
* 課金は OpenRouter のアカウントに乗る（応答に `id` / `provider` / `usage.cost` が増える。本プログラムは使いません）
* `GET /v1/models` は OpenRouter の一覧（`typesafe/jev-*` など別の名前空間）なので、**モデル名の照合は行わない**
* 画像入りの Danger Zone ヘッダは TypeSafe 直接のときだけ必要（OpenRouter の state 配列では付かない）

| 設定（すべて環境変数で上書き可） | 既定 | 内容 |
|---|---|---|
| `config.JEV_BASE_URL` | `https://openrouter.ai/api/v1/systemone` | System One のエンドポイント（TypeSafe 直接は `https://api.typesafe.ai/v1/systemone`） |
| `config.JEV_PROVIDER` | （空＝URL から推定） | `typesafe` / `openrouter`。一覧照合の有無が変わる |
| `config.JEV_SEND_DANGER_ZONE_HEADER` | `true` | 画像入りのとき Danger Zone ヘッダを付ける |
| `config.JEV_API_KEY_ENV` | `JEV_API_KEY` | API キーを読む環境変数の名前 |
| `config.JEV_API_KEY_FILE` | `.jev_api_key` | API キーを書いたファイル（git 管理外） |
| `config.JEV_MODEL` | `jev-latest` | リクエストの `model`（`GET /v1/models` で確認できる） |
| `config.JEV_AUTH_HEADER` / `config.JEV_AUTH_SCHEME` | `Authorization` / `Bearer {key}` | 認証ヘッダの形 |
| `config.JEV_IMAGE_ASSET_REFERENCE` / `config.JEV_IMAGE_FORMAT` | `landolt` / `png` | 画像を入れる `assets` の参照名と形式 |
| `config.JEV_REQUEST_TIMEOUT_SEC` | `180` | 1 リクエストのタイムアウト（秒） |

ローカルの clef との差分（画像の渡し方とヘッダ、接続確認の方法）は `services/jev_client.py` が吸収する。
**注意**: API キーは**環境変数が優先**です。`JEV_API_KEY` を設定したままにすると `.jev_api_key` の内容は
使われません（一時的に試した値が残っていないか、`scripts/show_jev_config.py` で確認できます）。

キーの混入は `python scripts/check_tracked_info.py` が検出する。統計の既定モデル（`config.BENCHMARK_MODELS`）は
**3 モデルすべて（`clef` / `clef-flash` / `jev`）**なので、既定の実行には API キーの設定が要る。
ローカルだけで回すときは `--models clef clef-flash` を指定する。

### 外部へ出る通信

* **通信先は System One のエンドポイントだけ**（ローカルの `config.CLEF_BASE_URL` と、
  jev を選んだときの `config.JEV_BASE_URL`）。それ以外のローカル外通信は行わない
* HTTP を扱うのは `services/` のクライアント（`systemone.py` / `clef_client.py` / `jev_client.py`）だけで、
  他の module はネットワークライブラリを import しない。この限定は
  `tests/test_network_policy.py` が検証している
* jev を使わなければ（`--models clef clef-flash`）インターネット接続は不要

## テスト

```powershell
pytest                 # 通常のテスト（Ollama 不要・ネットワーク I/O なし）
pytest -m integration  # 実際の clef API に対する統合テスト（Ollama が必要）
ruff check .           # 静的解析
ruff format --check .  # フォーマット確認
```

**既定のテストは環境に依存しない。** API 呼び出しはすべてスタブ
（`FakeClefClient` や `requests.post` の差し替え）で行い、`conftest.py` の
autouse フィクスチャがテストからの実 HTTP 呼び出しを検知したら失敗させる。
jev の設定は環境変数で上書きできるため、テストのあいだは既定値に固定している
（`JEV_BASE_URL` / `JEV_MODEL` / `JEV_PROVIDER` などを設定したままでも結果は変わらない）。

Ollama に接続する統合テストは `integration` マーカーで既定の実行から除外され、
`pytest -m integration` で実行した場合もサーバが応答しなければ自動的にスキップされる。
音声バックエンドが使えない環境では、再生を伴うテストだけがスキップされる。

テストの出力（描画結果の PNG など）は `artifacts/` に保存される。

同じ検査を **GitHub Actions（`.github/workflows/ci.yml`）**でも実行している
（push / Pull Request ごとに、Windows ランナー・Python 3.10 で `ruff check .` /
`ruff format --check .` / `pytest`）。
既定のテストはネットワークを使わず jev もスタブなので、**CI に API キーなどは登録していない**
（リポジトリの Secrets は不要）。

## スクリプト

```powershell
python scripts/generate_all_images.py     # 全モード・全視力の画像を image/ に生成
python scripts/show_jev_config.py         # jev の接続先・モデル・キーの取得元を表示（--check で接続確認）
python scripts/check_tracked_info.py      # 追跡対象ファイルのシークレット・個人環境依存をチェック
```

モード別・視力別の正答率の実測は、統計バッチ（上記「統計」）を使う。

## 効果音

正答・誤答の効果音は `sound/` に置く（`sound/right.mp3` と `sound/wrong.mp3`）。
配置しなくてもアプリは動作し、起動時に「効果音が見つかりません」と警告が出るだけ。

## 動作のメモ

* **HTTP 接続は永続セッションで使い回す**（`SystemOneClient`）。毎回新しい接続を張ると
  Windows では `localhost` が IPv6(::1) を先に試して待たされるため、ホスト名 `localhost` は
  `127.0.0.1` に置き換えて接続している（`config.PREFER_IPV4_HOST`）。
* **モデルを切り替えた直後の 1 回目**は Ollama がモデルを入れ替えるため数秒〜十数秒かかる
  （`clef` ↔ `clef-flash` の切替時）。2 回目以降はモデルごとの定常速度で返る。
* **効果音は Windows の Media Foundation バックエンド（`QT_MEDIA_BACKEND=windows`）で再生する**
  （`create_application` が Qt の初期化前に設定する）。Qt 既定の FFmpeg バックエンドは音声を
  開くたびにログをターミナルへ出し、読み込みも遅い。Windows 用プラグインが無い環境では
  FFmpeg のログだけを抑える。

## ディレクトリ構成

```
.
├── src/systemone_vision_test/       # 本体パッケージ（src レイアウト）
│   ├── config.py               # パス・定数・既定値（唯一の真実の源）
│   ├── domain/                 # GUI に依存しない中核ロジック
│   │   ├── landolt.py          # ランドルト環の生成（ISO 8596・輪郭は被覆率で描画）
│   │   ├── inspection.py       # 検査フロー（判定のステートマシン）
│   │   └── questions.py        # 方向判定の質問（言い回し・回答キーの変種を含む）
│   ├── services/               # 外部 I/O（HTTP を扱うのはここだけ）
│   │   ├── systemone.py        # System One クライアントの共通部分（送信・認証・エラー処理）
│   │   ├── answers.py          # 応答から回答を取り出す（API ごとの形の違いを吸収）
│   │   ├── clef_client.py      # ローカルの clef / clef-flash（画像は images 配列）
│   │   ├── jev_client.py       # jev（TypeSafe / OpenRouter。API キーが必要）
│   │   ├── factory.py          # モデル名 → クライアントの選択
│   │   └── sound.py            # 効果音（再生のたびに新しい再生パイプラインを作って破棄）
│   ├── benchmark/              # GUI なしの統計バッチ（python -m systemone_vision_test.benchmark）
│   │   ├── plan.py             # 実行計画（モデル × モード × 視力値 × 方向 × 反復）
│   │   ├── runner.py           # 実行（待ち時間なし・接続はモデルごとに使い回し）
│   │   ├── results.py          # 呈示結果と条件別の集計
│   │   ├── report.py           # Markdown レポートと生データ（JSON）
│   │   └── cli.py              # コマンドライン（引数・出力先・進捗）
│   └── gui/                    # PySide6 による画面
│       ├── app.py              # エントリーポイント（起動時の設定）
│       ├── main_window.py      # 組み立てと検査フローの進行
│       ├── canvas_view.py      # キャンバスの表示とオーバーレイの配置
│       ├── control_panel.py    # 左の操作用パネル
│       ├── probability_bar.py  # 応答確率の棒グラフ
│       ├── overlays.py         # 上辺の補助情報と中央の検査結果のオーバーレイ
│       ├── images.py           # 画像（numpy ↔ QPixmap）の変換
│       ├── workers.py          # 通信を担うバックグラウンドワーカー
│       └── style.py            # スタイルシート
├── tests/                      # pytest によるテスト
├── scripts/                    # 手動実行用の補助スクリプト
├── docs/                       # 仕様とドキュメント（docs/README.md に一覧）
├── .github/workflows/ci.yml    # CI（push / PR ごとに ruff と pytest を実行）
├── sound/                      # 効果音の素材（right / wrong）
├── image/                      # generate_all_images.py の出力
├── artifacts/                  # テスト・計測の出力（統計レポートは benchmark/ の下）
├── README.md
└── pyproject.toml
```

## 画像の描画

ランドルト環の輪郭は**画素の被覆率**（境界からの距離を 0..1 に均した値）で描く。
2 値でラスタライズすると、環が小さいほど円のギザつきが目立つため。
中間調になるのは輪郭の 1px 以内だけで、環の内部と背景は 2 値のまま。

## 設定値の一元管理

数値やパスは `src/systemone_vision_test/config.py` に集約している。生成モードごとの
パラメータ（既定値・スライダーの値域・実値への換算係数）は
`domain/landolt.py` の `PARAMETER_SPECS` が単一の定義であり、
`LandoltCParams` の既定値・GUI のスライダー・「デフォルトに戻す」ボタン・
計測スクリプトはすべてそこを参照する。

## 生成モードと測定結果

`clef`（27B）と `clef-flash`（9.1B）を GUI のドロップダウンで切り替えられる。
劣化モードの既定値は「清明モードとの差がはっきり出る」ように調整しており、
根拠と実測結果は `docs/filter_parameters.md` にある。

## ライセンス

本ソフトウェア本体は **MIT ライセンス**（[LICENSE](LICENSE)）で公開している。
利用しているサードパーティ製ソフトウェア（PySide6 / Qt、OpenCV、NumPy、Requests）の
ライセンスと、バイナリ配布時の注意点は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照。

## 免責

* 本ソフトウェアは **疑似**視覚検査（見え方のシミュレーションをモデルに判定させる実験用ツール）であり、
  医療機器ではない。診断・治療など医療用途には使用できない。
* `clef` / `clef-flash` / System One は、本ソフトウェアが API 経由で利用するモデルの名称であり、
  本プロジェクトはその提供元とは関係がない。モデルの重みは同梱していない。
