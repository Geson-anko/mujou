---
name: project-open-decisions
description: 2026-10-11 の初期セットアップ時点で未決定の設計判断。決まったらここから消して個別メモリにする
metadata:
  type: project
---

2026-10-11 の初期セットアップ時点で、次は未決定。決めたら該当行を消し、決定と理由を個別のメモリファイルに書く。

- **ネットワークスキーマの API**: define-and-run、Port・Channel、Composite、Port 同士の接続（2026-10-17 に `a.connect(b)` を廃止、[[project-port-registry]]）、PrevPort・NextPort は決定済み（[[project-network-ports]], [[project-message-types]]）。名前とホスト入出力は決定済み（2026-10-16 に `Run` ハンドルと `inputs=` / `state=`、2026-10-17 に `mujou.run(component, *components, ...)`。[[project-runtime-api-revision]]、[[project-run-node-and-write-time]]）。shape の決め方は未決定
- **実行の配置**: 複数ノードを同じワーカーで決まった順に回すことの指定方法（[[project-compute-granularity]]）。実行クラスは 2026-10-18 に決定（既定はワーカー、専用スレッドは `Node(dedicated_thread=True)`、プールは実行ごとで `mujou.run(workers=None)`、[[project-mn-scheduler]]）。目標周期は 2026-10-17 に `Node(period=...)` として実装（[[project-node-period]]、シグネチャはユーザー確認待ち）。数え方とクラス属性の扱いは同日ユーザーが決定。プールでも同じ規則（期限の早い順）。残り: 既定のワーカー数が適切か（M:N の PR 3 のベンチマーク）
- **ライフサイクルの細部**: hook の構成と停止・状態の取得・復元の流れは決定済み（[[project-node-lifecycle]], [[project-pause-and-state]]）。例外時は全体を終了し、状態の木は取得しない（[[project-shutdown-on-exception]]）。保存はユーザー。停止タイムアウトの既定値（30 秒）と `shutdown`/`join` の `timeout`（既定 `math.inf`）は 2026-10-16 にユーザーが決定。スロットの識別名（`"body.l1.next/out"` など）と例外終了後のタイマー再開は推奨案で実装、ユーザー確認待ち
- **時定数**: 連続時間形の $\tau_h, \tau_\lambda, \tau_\theta$ の比と、$\Delta t$ の上限の扱い（skill `pc-alm` 7.1 節）。周期を入れた後も `max_dt` はまだ無い（2026-10-17）
- **双対変数の扱い**: ストリームのステップ間で λ をリセットするか持ち越すか、dual leak を入れるか（skill `pc-alm` 7 節）
- **mujou.Node の細部**: モジュール構成と `mujou.Node` の構成は決定済み（[[project-torch-module]]）。2026-10-16 に推奨案で実装、ユーザー確認待ち: 設定は内部の hook で行う（2026-10-17 に `_on_thread_start()` とその後に呼ぶ公開の `on_thread_start()` に分けた、[[project-runtime-api-revision]]。2026-10-18 に `_on_thread_start()` を、切り替えのたびに呼ぶ `_on_enter()` に置き換えた。2026-10-19 に公開の `on_thread_start()` を廃止した、[[project-mn-scheduler]]）／トップレベルの公開名は `Channel, DictSpec, Graph, Group, Network, NetworkError, NextPort, Node, PauseTimeoutError, Port, PrevPort, Run, StateDict, TensorSpec, run, __version__`（`src/mujou/__init__.py` の `__all__`。ほかにサブモジュールの `mujou.time`・`mujou.typing`・`mujou.testing`、PC-ALM は `mujou.pc` だけ、[[project-pc-module]]）／torch の内部スレッド数は mujou が設定せずホストが決める
- **グローバルタイマーとスケジューラの境界**（2026-10-15 に挙がり、2026-10-16 にランタイムを実装）: 制御は 1 本のスレッドに直列化し、Δt=0 の待ちは制御の要求でも起きる（pamiq-core の `thread_control.py` に倣った）。残り: 時刻の読み取りがグローバルな Mutex を取る（性能が問題になったら seqlock など）／実経過を f64 で扱うので、約 104 日後に ns の分解能が落ちる。同じタイマーで動くネットワーク同士は停止で時間を止め合う。止め合いたくないネットワークは別のタイマーで動かす（2026-10-16 にユーザーが決定、[[project-timer-per-run]]。`mujou.run(..., timer=...)` は 2026-10-16 に実装済み）
- **2026-10-17 の整備で出た確認事項**: 状態の木のテンソルを CPU へ移すか（2026-10-17、ユーザーの決定でデバイスに置いたまま、利用者が `StateDict.cpu()` / `to()` で移す。決着）、`Timer.pause/resume` を残すか、`pause_timeout` を `Run.state_dict(timeout=)` へ移すか（2026-10-17、ユーザーの決定で移した。決着、`specs/open-decisions.md`）、`Compose` の名前（2026-10-17 に `Group` へ改名して決着）、相手が未送信かを判定する API、手動の実行で `period` を守る選択肢、`Run.timer`、GPU の CI。一覧は `specs/open-decisions.md` の「2026-10-17 の推奨案」

**How to apply:** これらに触れるタスクでは、勝手に決めずにユーザーに確認する（CLAUDE.md 開発原則 1）。

関連: [[project-purpose]], [[project-compute-granularity]], [[project-native-core-rust]]
