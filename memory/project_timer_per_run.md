---
name: project-timer-per-run
description: 時間軸は実行ごとに 1 本。1 つの実行の全ノード・全スロット・ホストの入力はその実行のタイマーを使い、既定はグローバルタイマー。時刻は Timestamp 型
metadata:
  type: project
---

2026-10-16 ユーザー決定。2026-10-11 の「グローバルタイマーはシングルトン（システム全体で時間は 1 つ）」（[[project-async-node-execution]]）を改めた。

- **時間軸は実行ごとに 1 本。** 1 つの実行の全ノード・全スロット・ホストの入力は、その実行のタイマーを使う。既定は全実行が共有するグローバルタイマー
- 将来の API は `mujou.run(*networks, timer=...)`。`timer=None` なら既定のグローバルタイマー。独立に回したいネットワークは別のタイマーで動かす。ノードは自分の属性としてタイマーを持つ予定
- 時刻は `Timestamp` 型で表す。秒とナノ秒は同じものなので、`float(ts)` / `int(ts)` は秒（int は整数部）、ナノ秒は `ts.ns`。`Timer.now()` が `Timestamp` を返し、`Timestamp` 同士の差も `Timestamp`

**Why:** 1 つのネットワークの停止（状態の取得など）が、同じプロセスで独立に動くほかのネットワークの時間まで止めるのを避けたい。1 つの実行の中では時間軸が 1 本でないと、停止中の時刻 ≤ T の保証とメッセージの時刻の意味が崩れる。

**How to apply:**

- スロット（`Outlet` / `InputGroup`）とランタイムは同じタイマーで作る。今はコアも bindings も一致を確かめない。一致を強制するのは、run の API の作り直し（スロットを run の時点でその実行のタイマーで作る）で入れる
- 停止・復元の排他はタイマーごと。同じタイマーで走るランタイムがあれば時刻を戻さない
- 実装（2026-10-16）: Rust の `mujou_core::time::Timer`（`Arc` で共有）と `time::global()`、Python の `mujou._core.Timer` / `GlobalTimer` / `Timestamp` と各部品の `timer=`（`docs/execution-model.md` のグローバルタイマーの実装節）

関連: [[project-async-node-execution]], [[project-pause-and-state]], [[project-open-decisions]]
