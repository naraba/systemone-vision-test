# ドキュメント

仕様と測定の記録。入口はリポジトリ直下の [../README.md](../README.md)。

| ファイル | 内容 |
|---|---|
| [spec.md](spec.md) | プログラムの仕様。API の形、画像の要件（ISO 8596・アンチエイリアス）、検査フロー、GUI の要件 |
| [filter_parameters.md](filter_parameters.md) | 劣化モード（近視・混濁・乱視）の既定値の根拠と実測結果、キャンバスサイズの余白の検討 |
| [landolt_direction_bias.md](landolt_direction_bias.md) | ランドルト環の方向回答の偏り（clef / clef-flash / jev の比較、2,496 呈示の全量測定に基づく） |

## メモ

* 実測値は測定時の環境（Windows 11 / ローカル Ollama）に依存する
* 数値やパスの定義は `src/systemone_vision_test/config.py` が唯一の真実の源で、
  仕様書の数値と食い違ったときはコードを優先する
