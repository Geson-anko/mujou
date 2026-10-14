---
name: project-open-decisions
description: 2026-10-11 の初期セットアップ時点で未決定の設計判断。決まったらここから消して個別メモリにする
metadata:
  type: project
---

2026-10-11 の初期セットアップ時点で、次は未決定。決めたら該当行を消し、決定と理由を個別のメモリファイルに書く。

- **ネットワークスキーマの API**: ブロックの組み合わせと双方向の接続を定義するネットワークオブジェクトを define-and-run で記述し、スキーマオブジェクトを作ることと、Port・Channel による双方向接続の仕組み（[[project-network-ports]]）は決定済み（[[project-message-types]]）。実行と外からの読み書きは暫定案あり（[[project-host-io]]）。API の具体的な形（クラス名・メソッド名、shape の決め方など）は未決定
- **実行の配置**: 複数ブロックを同じワーカーで決まった順に回すことの指定方法、実行クラス（タスク・専用スレッド）と目標周期の指定方法（[[project-compute-granularity]]）
- **ライフサイクルの細部**: hook の構成と停止・状態の取得・復元の流れは決定済み（[[project-block-lifecycle]], [[project-pause-and-state]]）。例外時は全体を終了し、設定で状態の木を取得（[[project-shutdown-on-exception]]）。保存はユーザー。停止タイムアウトの既定値、スロットの識別名、例外時の木を受け取る API の名前は未決定
- **時定数**: 連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い（skill `pc-alm` 7.1 節）
- **双対変数の扱い**: ストリームのステップ間で λ をリセットするか持ち越すか、dual leak を入れるか（skill `pc-alm` 7 節）
- **ライセンス**: LICENSE ファイル未作成

**How to apply:** これらに触れるタスクでは、勝手に決めずにユーザーに確認する（CLAUDE.md 開発原則 1）。

関連: [[project-purpose]], [[project-compute-granularity]], [[project-native-core-rust]]
