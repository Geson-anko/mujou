# mujou 仕様書

mujou の設計方針と仕様をまとめた文書群。2026-10-13 時点で `memory/` に記録されている決定事項を、読み物として再構成した。

> 状態: 初期開発中。ここに書いた仕様は設計上の決定であり、まだ実装されていないものを含む。

## 目次

| 文書                                 | 内容                                                                 |
| ------------------------------------ | -------------------------------------------------------------------- |
| [概要](overview.md)                  | mujou の目的、名前の由来、狙う用途、設計判断の拠り所                 |
| [ブロック](blocks.md)                | 計算単位「ブロック」の定義、大きさ（ニューロン・レイヤー・スタック） |
| [実行モデル](execution-model.md)     | 共有スロットによる非同期通信、グローバルタイマー、連続時間の定式化   |
| [ネイティブコア](native-core.md)     | Rust コア、M:N スケジューリング、メッセージの値の型                  |
| [更新手法](update-rules.md)          | 更新手法の差し替え、既定の PC-ALM の位置づけと連続時間形             |
| [Python 環境](python-environment.md) | free-threaded Python 3.14t での開発と GIL ありビルドのサポート       |
| [未決定事項](open-decisions.md)      | まだ決めていない設計判断                                             |

## 読む順番

初めて読む場合は、[概要](overview.md)、[ブロック](blocks.md)、[実行モデル](execution-model.md) の順に読むと、残りの文書の前提がそろう。

## 関連資料

PC-ALM の式・平衡点の性質・実装上の落とし穴は skill `pc-alm`（`.claude/skills/pc-alm/SKILL.md`）に、実行モデルの不変条件と禁止パターンは skill `locality-and-async`（`.claude/skills/locality-and-async/SKILL.md`）にある。決定の経緯と個別のメモは [memory/](../memory/) を参照。
