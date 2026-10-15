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
- `mujou.time` の属性 docstring は `check-docstring-first` に弾かれるので、モジュールの docstring に一覧で書いた。関数の説明は Rust の doc comment が `__doc__` になる
- sleep のテストは、sleeper スレッドが起動する前に advance されると目標がずれて落ちた。スレッドの中で自分の sleep の長さを測り、メインは少しずつ advance して戻るまで回す形にした（起動のタイミングに依らない）。fixture `measure_sleep`（tests/conftest.py）
- 30 回（Rust）・10 回（pytest）の繰り返しで不安定なテストなし

## 自己レビュー（段階 4）

- 欠陥: 大きな秒数で `sleep` すると、ns が u64::MAX に飽和したあと `time_ns() + dt_ns` が release ビルドで wrap し、すぐ戻る。`advance_ns` も同じ → `saturating_add` に直し、テスト 2 件を追加（段階 2・3 に戻した）
- 整理: 1 箇所でしか使わない `_start_sleeper` をインライン化
- 局所性・非同期性: タイマーはノード間の通信ではなく、ノードの更新則に触れない。グローバルなバリアも入れていない（停止は制御側の操作だけ）

## レビュー（code-reviewer・algorithm-reviewer、どちらも approve）と対応

- 実時計の上界が、実時間の計測より前の区間を含まず理論上 flaky → 計測を先に取る（2 件）
- `time_ns * 1e-9` は正しく丸められない → `/ 1e9`
- `set_time_scale` などに doc comment が無く `mujou.time.set_time_scale.__doc__` が None → 追加
- 公開した名前を `_core.timer` から直接テストしていた → `mujou.time` を通すテストに移し、`test_core.py` は private な制御だけ。getter/setter の往復テストは削除
- 仮想時計の原点が非決定的 → fixture で `load_state_dict({"time_ns": 0, ...})`。fixture は `tests/mujou/conftest.py` に移し、`measure_sleep` は唯一の利用者 `test_time.py` の関数にした
- `wait_until_ns` に大きな timeout を渡すと panic → `checked_add`、表せない期限は無期限
- `reanchor` が `Instant::now()` を 2 回取り、張り替えのたびに数十 ns 遅れる → 1 回にした
- 復元の順序（停止 → load → 起動 → 再開）と、Δt を ns の差から作ることを `docs/execution-model.md` に追記
- 見送り（`memory/project_open_decisions.md` に記録）: 停止要求で待機を起こす方法、pause の持ち主、Mutex の競合、f64 の分解能
- 見送り: `state_dict` の TypedDict（実行時に存在しない型を stub に置くことになる）、`load_state_dict` の例外の型の統一（private）、`import mujou` で `mujou.time` を読み込まない（`torch` 同様に明示 import でよい）
