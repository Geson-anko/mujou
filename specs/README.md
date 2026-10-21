# mujou 仕様書

開発者向けの設計仕様。今の設計とその理由だけを書く。決定の経緯は git の履歴に、利用者向けの使い方は [docs/](../docs/)（英語と日本語）にある。

## 目次（読む順）

| 文書 | 内容 |
| --- | --- |
| [概要](overview.md) | 目的、名前の由来、不変の原則 |
| [構成](architecture.md) | 層と各モジュールの責務、Rust と Python の境界、free-threaded の規則、内部の名前の規約 |
| [提供シナリオとガード](scenarios.md) | 支える使い方、サポートしないもの、ガードの判定規則と台帳（ガードの正本） |
| [ノード](nodes.md) | ノード、hook と呼ぶ順、Driver、状態の規則、周期、CUDA |
| [実行モデル](execution-model.md) | 共有スロット、連続時間、RunClock、停止・取得・復元、状態の木の意味、終了と例外 |
| [ネットワーク](network.md) | Port・MultiPort・Channel、connect、パス、`__dict__` の走査、起動の 3 段、ホストの入出力 |
| [メッセージ](messages.md) | spec、torch とコアの変換（codec）、DLPack で必ずコピーすること、バッファプール |
| [ネイティブコア](native-core.md) | Rust の責務、M:N スケジューラ、ABI、ツールチェーン、性能の予算 |
| [更新手法](update-rules.md) | 更新手法の差し替え、`mujou.pc` が基盤の外の利用者であること |
| [mujou.pc](pc.md) | PC-ALM のノードの設計（状態、メッセージ、`on_update` の段、LearnClock、時定数、Chain） |
| [PC-ALM の連続時間系](pc-alm-continuous-time.md) | 連続時間形の各式の意味、時定数、平衡点、離散化 |
| [PC の比較実験](research/pc-experiments.md) | `mujou.pc` の設計を決めた実験の表と、採らなかった案 |
| [性能の基準値](research/benchmarks.md) | 性能の基準値と測り方 |
| [未決定事項](open-decisions.md) | まだ決めていない設計判断 |

## 知識の置き場所

1 つの事実は 1 か所に置く。ほかの場所は正本へリンクする。

| 知識 | 正本 | ほかの場所の扱い |
| --- | --- | --- |
| 公開名の一覧 | `__all__` と `tests/mujou/test_init.py` | specs と memory には書かない |
| API の振る舞い（引数、例外） | docstring | docs は mkdocstrings で取り込む。specs には書かない |
| 状態の木の形 | `Run.state_dict` の docstring | `tests/mujou/test_state.py` で固定する。docs と specs はリンクだけ |
| 状態に入るものの規則 | `Node` の docstring | `docs/*/concepts.md` は 1 文とリンク |
| ノードが守る規則と注意書き | `docs/*/concepts.md` の 1 節 | 各ページは 1 行のリンクだけ |
| 使い方のコード | `examples/` | docs は snippet で取り込む |
| 設計とその理由 | `specs/` | skill は specs にリンクする |
| import の層 | `tests/test_layers.py` | [構成](architecture.md) は理由を書いてリンクする |
| 内部の名前の規約 | [構成](architecture.md#内部の名前の規約) | skill `refactor-conventions` はリンクする |
| ガード | [提供シナリオとガード](scenarios.md) の台帳 | 利用者向けには docstring の `Raises:` |
| PC の理論 | skill `pc-alm` | [mujou.pc](pc.md) は mujou の選択だけ |
| 実験の数値と性能の基準値 | `specs/research/` | ほかの文書はリンクする |
| 決定とユーザーの好み | `memory/`（1 ファイル 1 決定。今の決定と理由だけ） | 履歴は git |
| dtype の対応表 | `mujou.spec.DLPACK` | `TensorSpec.dtypes` で公開する |
| hook の名前 | Rust の `Hook` と、bindings の対応表 | — |
| 属性の所属（ポート、子） | インスタンスの `__dict__` | レジストリを持たない |

## 関連資料

PC-ALM の式・平衡点の性質・実装上の落とし穴は skill `pc-alm` に、実行モデルの不変条件と禁止パターンは skill `locality-and-async` にある。
