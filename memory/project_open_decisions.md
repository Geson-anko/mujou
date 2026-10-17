---
name: project-open-decisions
description: 2026-10-11 の初期セットアップ時点で未決定の設計判断。決まったらここから消して個別メモリにする
metadata:
  type: project
---

2026-10-11 の初期セットアップ時点で、次は未決定。決めたら該当行を消し、決定と理由を個別のメモリファイルに書く。

- **ネットワークスキーマの API**: define-and-run、Port・Channel、Composite、`a.connect(b)`、PrevPort・NextPort は決定済み（[[project-network-ports]], [[project-message-types]]）。名前とホスト入出力（`mujou.run`、`write`/`read`/`subscribe`/`state_dict`/`load_state_dict`/`shutdown`/`join`）は 2026-10-16 に推奨案で実装、ユーザー確認待ち（[[project-host-io]]）。shape の決め方は未決定
- **実行の配置**: 複数ノードを同じワーカーで決まった順に回すことの指定方法、実行クラス（タスク・専用スレッド）と目標周期の指定方法（[[project-compute-granularity]]）
- **ライフサイクルの細部**: hook の構成と停止・状態の取得・復元の流れは決定済み（[[project-node-lifecycle]], [[project-pause-and-state]]）。例外時は全体を終了し、状態の木は取得しない（[[project-shutdown-on-exception]]）。保存はユーザー。停止タイムアウトの既定値（30 秒）と `shutdown`/`join` の `timeout`（既定 `math.inf`）は 2026-10-16 にユーザーが決定。スロットの識別名（`"body.l1.next/out"` など）と例外終了後のタイマー再開は推奨案で実装、ユーザー確認待ち
- **時定数**: 連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い（skill `pc-alm` 7.1 節）
- **双対変数の扱い**: ストリームのステップ間で λ をリセットするか持ち越すか、dual leak を入れるか（skill `pc-alm` 7 節）
- **mujou.Node の細部**: モジュール構成と `mujou.Node` の構成は決定済み（[[project-torch-module]]）。2026-10-16 に推奨案で実装、ユーザー確認待ち: 設定は `on_start` の前に呼ぶ `_on_thread_start()` で行う／トップレベルの公開名は `Node, Network, Port, PrevPort, NextPort, Channel, TensorSpec, DictSpec, NetworkError, PauseTimeoutError, run`（PC-ALM は `mujou.pc` だけ、[[project-pc-module]]）／torch の内部スレッド数は mujou が設定せずホストが決める
- **グローバルタイマーとスケジューラの境界**（2026-10-15 に挙がり、2026-10-16 にランタイムを実装）: 制御は 1 本のスレッドに直列化し、Δt=0 の待ちは制御の要求でも起きる（pamiq-core の `thread_control.py` に倣った）。残り: 時刻の読み取りがグローバルな Mutex を取る（性能が問題になったら seqlock など）／実経過を f64 で扱うので、約 104 日後に ns の分解能が落ちる／1 つのネットワークを停止するとプロセスのグローバル時間が止まり、同時に走る他のネットワークも止まる（この意味でよいかユーザー確認待ち）
- **ライセンス**: LICENSE ファイル未作成

**How to apply:** これらに触れるタスクでは、勝手に決めずにユーザーに確認する（CLAUDE.md 開発原則 1）。

関連: [[project-purpose]], [[project-compute-granularity]], [[project-native-core-rust]]
