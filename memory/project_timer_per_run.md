---
name: project-timer-per-run
description: 時間軸は実行ごとに 1 本。既定はグローバルタイマー、渡せば実行ごとの Timer、復元した実行は保存した時刻から始まる専用の Timer（R8）。ランタイムの唯一の時刻源は RunClock（min(timer, end)）。Timer(start=, time_scale=)、時刻は Timestamp、差は秒の float
metadata:
  type: project
---

2026-10-16 ユーザーの決定（時間軸は実行ごとに 1 本）。詳細は `specs/execution-model.md` の「タイマー」「実行ごとのタイマー」「RunClock」。

- **時間軸は実行ごとに 1 本。** 1 つの実行の全ノード・全スロット・ホストの入力は、その実行のタイマーを使う。既定は全実行が共有する `mujou.time.global_timer`。独立に回したい実行は `mujou.run(..., timer=Timer())` で別のタイマーにする。ホストは `run.timer` で待ち、止める
- **復元した実行は専用のタイマーで走る。** `mujou.run(..., state=tree)` は、木の時刻と time scale から始まる新しい `Timer`（testing では `VirtualTimer`）で走る。`timer=` との同時指定は断る（K23）。復元した実行は `global_timer.pause()` では止まらない。R8 の推奨案で進めた。ユーザーの確認待ち（[[project-restructure-2026-10-20]]）
- **ランタイムの時刻源は `RunClock` の 1 本。** `now = min(timer, end)`。end は停止を求めた時刻か、失敗を検出した時刻。dt、最後の更新、スロットの時刻はすべてこれから取る。公開名は `ReadOnlyTimer`（ノードの `self.timer`）
- **API:** `Timer(*, start=Timestamp(ns=0), time_scale=1.0)`、`time_scale()` / `set_time_scale()`、`VirtualTimer(*, start=)`（`advance` でだけ進む）、`ReadOnlyTimer(timer)`、`Timestamp(*, ns)`。`Timestamp` 同士の差は秒の float

**Why:** 1 つの実行の停止が、独立に動くほかの実行の時間まで止めるのを避けたい。1 つの実行の中では時間軸が 1 本でないと、停止中の時刻 ≤ T の保証とメッセージの時刻の意味が崩れる。復元した実行に専用のタイマーを与えると、共有のタイマーを巻き戻す場合、走っている実行の数え上げ、段 2 での読み込みが要らなくなる。時刻源を RunClock 1 本にすると、終わった後の木の時刻が全スロットと全ノードの最後の更新以上になることが、データの形で保たれる。

**How to apply:** スロット（Outlet / InputGroup）、ランタイム、Driver は同じ RunClock で作る。止まった後の `on_exception` / `on_stop` で経過時間が要るなら実時間で測る。

関連: [[project-async-node-execution]], [[project-pause-and-state]], [[project-timer-pause]]
