---
name: project-open-decisions
description: 2026-10-11 の初期セットアップ時点で未決定の設計判断。決まったらここから消して個別メモリにする
metadata:
  type: project
---

2026-10-11 の初期セットアップ時点で、次は未決定。決めたら該当行を消し、決定と理由を個別のメモリファイルに書く。

- **値の型の詳細**: コアが独自に持つメッセージの値の型（[[project-native-core-rust]]）の具体形（dtype、shape、メモリ配置など）
- **複数マシンへの分散**: プロセス内は Rust の M:N 実行（[[project-native-core-rust]]）に決定。マシンをまたぐスロットの実装（KV ストアなど）は未決定
- **時定数**: 連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い（skill `pc-alm` 7.1 節）
- **双対変数の扱い**: ストリームのステップ間で λ をリセットするか持ち越すか、dual leak を入れるか（skill `pc-alm` 7 節）
- **ライセンス**: LICENSE ファイル未作成

**How to apply:** これらに触れるタスクでは、勝手に決めずにユーザーに確認する（CLAUDE.md 開発原則 1）。

関連: [[project-purpose]], [[project-compute-granularity]], [[project-native-core-rust]]
