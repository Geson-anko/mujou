---
name: project-timer-pause
description: タイマーの一時停止（pause / resume / is_paused）を公開し、時刻の読み方を now() に一本化した決定。複数の実行の状態を同じ時刻でそろえて取るためにホストが時間を止める
metadata:
  type: project
---

2026-10-17 ユーザーの決定。

- `mujou.time.Timer` に `pause()` / `resume()` / `is_paused()` を公開し、グローバルタイマー用に `mujou.time.pause()` / `resume()` / `is_paused()` を置く。`pause` / `resume` はノードの hook からは `RuntimeError`。意味は native のスイッチのまま（ランタイムの保持とは独立、回数を数えない）
- 時刻の読み方は `now() -> Timestamp` だけにする。`mujou.time.time()` / `time_ns()` は消した。秒は `Timestamp.float()`、ナノ秒は `.ns`
- `Timestamp - Timestamp` は符号付きの秒（`float`、`(a.ns - b.ns) / 1e9` と同じ丸め）

**Why:** ユーザーの指摘「pause/resume は状態保存のときに必須のもの。practical なシナリオのテストが無いだけで、本来の存在意図を忘れている」。pamiq-core の TimeController と同じく、ホストが時間を止めてその間に状態を取る・保存する。同じタイマーで走る複数の実行の状態を 1 つずつ `Run.state_dict()` で取ると、その間に時間が進んで実行ごとに時刻がずれる。

**How to apply:** テストの基盤だけが使っている API を「使われていない」と判断して消す前に、本来の用途（ホストの操作）を確かめる。詳細は `specs/execution-model.md` のグローバルタイマーの実装。
