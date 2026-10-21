---
name: project-timer-pause
description: タイマーの一時停止（Timer.pause / resume / is_paused）は公開する。ホストが時間を止めてその間に状態を取る、複数の実行の状態を同じ時刻でそろえるため。時刻の読み方は now() -> Timestamp だけ、Timestamp 同士の差は秒の float
metadata:
  type: project
---

2026-10-17 ユーザーの決定。

- `Timer.pause()` / `resume()` / `is_paused()` を公開する。グローバルタイマーなら `mujou.time.global_timer.pause()`、実行のタイマーなら `run.timer.pause()`（モジュール関数の `mujou.time.pause()` などは R6 の推奨案で廃止した。ユーザーの確認待ち: [[project-restructure-2026-10-20]]）。hook の中から呼ぶと `RuntimeError`（K16）。ランタイムの保持とは独立のスイッチで、回数を数えない
- 時刻の読み方は `now() -> Timestamp` だけ。秒は `float(ts)`、ナノ秒は `ts.ns`
- `Timestamp - Timestamp` は符号付きの秒（`float`）

**Why:** ユーザーの指摘「pause/resume は状態保存のときに必須のもの。practical なシナリオのテストが無いだけで、本来の存在意図を忘れている」。pamiq-core の TimeController と同じく、ホストが時間を止めてその間に状態を取る・保存する。同じタイマーで走る複数の実行の状態を 1 つずつ取ると、その間に時間が進んで時刻がずれる。

**How to apply:** テストの基盤だけが使っている API を「使われていない」と判断して消す前に、本来の用途（ホストの操作）を確かめる。詳細は `specs/execution-model.md` の「タイマー」。
