---
name: project-open-decisions
description: 2026-10-11 の初期セットアップ時点で未決定の設計判断。決まったらここから消して個別メモリにする
metadata:
  type: project
---

2026-10-11 の初期セットアップ時点で、次は未決定。決めたら該当行を消し、決定と理由を個別のメモリファイルに書く。

- **ネットワークスキーマの API**: ノードの組み合わせと双方向の接続を定義するネットワークオブジェクトを define-and-run で記述し、スキーマオブジェクトを作ることと、Port・Channel による双方向接続の仕組み（[[project-network-ports]]）は決定済み（[[project-message-types]]）。実行と外からの読み書きは暫定案あり（[[project-host-io]]）。API の具体的な形（クラス名・メソッド名、shape の決め方など）は未決定
- **実行の配置**: 複数ノードを同じワーカーで決まった順に回すことの指定方法、実行クラス（タスク・専用スレッド）と目標周期の指定方法（[[project-compute-granularity]]）
- **ライフサイクルの細部**: hook の構成と停止・状態の取得・復元の流れは決定済み（[[project-node-lifecycle]], [[project-pause-and-state]]）。例外時は全体を終了し、設定で状態の木を取得（[[project-shutdown-on-exception]]）。保存はユーザー。停止タイムアウトの既定値、スロットの識別名、例外時の木を受け取る API の名前は未決定
- **時定数**: 連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い（skill `pc-alm` 7.1 節）
- **双対変数の扱い**: ストリームのステップ間で λ をリセットするか持ち越すか、dual leak を入れるか（skill `pc-alm` 7 節）
- **mujou.Node の細部**: モジュール構成と `mujou.Node` の構成は決定済み（[[project-torch-module]]）。`mujou.*` のトップレベルに公開する名前の一覧（PC-ALM のクラスは `mujou.pc` からだけ公開すると決定済み、[[project-pc-module]]）、`on_start` を override したときに設定が漏れない仕組み、torch の内部スレッド数を誰がどう決めるかは未決定
- **グローバルタイマーとスケジューラの境界**（2026-10-15、タイマーの実装レビューで出た。スケジューラを実装するときに決める）
  - スレッドを起こす方法と例外の伝え方は pamiq-core の `thread_control.py` に倣う方向（ユーザーの指示、2026-10-15）。`ThreadController`（再開・終了のイベント、終了は再開を先に set して待ち手を起こす、制御は 1 本のスレッドだけ、ノードには読み取り専用の view）、`ThreadStatus`（停止したこと・例外のイベント）、`ThreadStatusesMonitor`（全スレッドの停止をタイムアウト付きで待つ）
  - これで答える見込みの 2 点: 目標周期の期限まで待つノードを停止要求で起こす（周期の待ちを制御のイベントでも起きるようにする。今の `wait_until_ns` は目標時刻かタイムアウトでしか戻らない）／pause の持ち主（制御を 1 本のスレッドに直列化する）。例外で終了した後に停止したまま残るシングルトンの扱いも制御スレッドの終了処理で決める
  - 時刻の読み取りがグローバルな Mutex を取る。スロットへの書き込みのたびに時刻を添えるので、ノードが多いと競合しうる。性能が問題になったら seqlock や atomic のスナップショットを検討する
  - 実経過を f64 で扱うので、anchor の張り替えから約 104 日で ns の分解能が落ちる（単調性は保たれる）。長く動かすなら定期的に張り替える
- **ライセンス**: LICENSE ファイル未作成

**How to apply:** これらに触れるタスクでは、勝手に決めずにユーザーに確認する（CLAUDE.md 開発原則 1）。

関連: [[project-purpose]], [[project-compute-granularity]], [[project-native-core-rust]]
