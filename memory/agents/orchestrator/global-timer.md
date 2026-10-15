# グローバルタイマー（PR 3）

2026-10-15、solo-dev-cycle。PR 2（Rust コア）の上に積む。

## 計画

### Rust（`crates/mujou-core/src/time.rs`）

- `GlobalTimer`: `Mutex<State>` + `Condvar`。State は anchor（グローバル時刻 ns・`Instant`）、scale、paused、仮想時計か
- 時刻 = anchor_ns + 実経過 × scale。停止中と仮想時計では anchor_ns のまま。時刻は u64 ns が正で、秒 f64 はそこから作る
- scale の変更・停止・再開・時計の切り替えの前に anchor を今の時刻へ張り替える（pamiq-core の TimeController と同じ）
- `wait_until_ns(target, timeout) -> bool`: Condvar で待つ。停止中は進まない。仮想時計では `advance` されるまで待つ。状態が変わるたびに notify_all
- `state() -> TimerState { time_ns, scale }` / `load_state`。停止中かどうかと時計の種類は木に入れない（取得は必ず停止中に行うので、入れても常に「停止中」になるため）
- `global()`: `LazyLock` のシングルトン。グローバル時刻は最初に使った時点を 0 とする
- `advance` は仮想時計でだけ有効、停止中は不可。scale は有限の正の数だけ

### Python

- `_core.timer`: `frozen` な pyclass のインスタンス（シングルトンに委譲）。`sleep` は GIL を外し、50 ms 刻みで待って刻みごとに `check_signals`（Ctrl-C を効かせる）
- `mujou.time`（公開、暫定）: `time` / `time_ns` / `sleep` / `get_time_scale` / `set_time_scale`。停止・状態・仮想時計はフレームワークとテスト用に `_core.timer` に留める

## テスト観点（オラクル）

- 仮想時計: `advance` した量だけ ns で厳密に進む。scale に依らない。sleep は advance で目標に達したときだけ戻る
- 実時計: 実時間の計測で挟んだ上下界 `scale·(r1−r0) ≤ g1−g0 ≤ scale·(r_after−r_before)`（負荷で落ちない）
- 停止: 停止中は時刻が変わらない。再開後に飛ばない（上界で確認）。停止中の sleep は再開まで戻らない
- 状態: 取得した時刻から再開する（別インスタンスに load）
- 境界: scale が 0・負・NaN、sleep の負・NaN を拒否。実時計で advance、停止中の advance を拒否

## 計画外の判断

- テストの修正: `state::restores_the_time_and_scale` が新しいタイマーの時刻を 0 と仮定していた。仮想時計に切り替えるまでの実時間ぶん進むので、切り替え後の時刻を起点にした（仕様の変更ではない）
