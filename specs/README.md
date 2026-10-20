# mujou 仕様書

mujou の設計方針と仕様をまとめた文書群。2026-10-18 時点で `memory/` に記録されている決定事項を、読み物として再構成した。

開発者向けの設計仕様で、設計判断とその理由を書く。利用者向けの使い方は [docs/](../docs/)（mkdocs のサイト、英語と日本語）に書く。

> 状態: 初期開発中。ここに書いた仕様は設計上の決定であり、まだ実装されていないものを含む。

## 目次

| 文書                                             | 内容                                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------------- |
| [概要](overview.md)                              | mujou の目的、名前の由来、狙う用途、設計判断の拠り所                            |
| [ノード](nodes.md)                            | 計算単位「ノード」の定義、大きさ、ライフサイクル hook                         |
| [実行モデル](execution-model.md)                 | 共有スロット、グローバルタイマー、連続時間、停止と状態の取得・復元、終了と例外  |
| [メッセージ](messages.md)                        | 値の型（`torch.Tensor`）とメッセージの型、スキーマ、スロット、メモリ管理       |
| [ネットワーク](network.md)                       | ノード・ネットワーク・Port・Channel、双方向の接続、入れ子、入出力、分岐と合流 |
| [ネイティブコア](native-core.md)                 | Rust コア、M:N スケジューリング、メッセージの値の型                             |
| [更新手法](update-rules.md)                      | 更新手法の差し替え、既定の PC-ALM の位置づけと連続時間形                        |
| [PC-ALM の連続時間系](pc-alm-continuous-time.md) | 連続時間形の各式の意味、時定数、平衡点での BP との一致、離散化                  |
| [mujou.pc の実装計画](pc-module.md) | `mujou.pc` のノード・状態・メッセージ・θ の更新間隔・時定数・版番号によるリセット・MNIST の example |
| [PyTorch の利用](torch.md)                       | モジュール構成、GPU の扱い、`mujou.Node`、プロセス全体に効く設定                 |
| [Python 環境](python-environment.md)             | free-threaded Python 3.14t での開発と GIL ありビルドのサポート                  |
| [未決定事項](open-decisions.md)                  | まだ決めていない設計判断                                                        |

## 読む順番

初めて読む場合は、[概要](overview.md)、[ノード](nodes.md)、[実行モデル](execution-model.md) の順に読むと、残りの文書の前提がそろう。

## 関連資料

PC-ALM の式・平衡点の性質・実装上の落とし穴は skill `pc-alm`（`.claude/skills/pc-alm/SKILL.md`）に、実行モデルの不変条件と禁止パターンは skill `locality-and-async`（`.claude/skills/locality-and-async/SKILL.md`）にある。決定の経緯と個別のメモは [memory/](../memory/) を参照。
