# core-runtime WP-B（ノードの実行と制御）の判断ログ

計画書: `memory/agents/implementation-planner/core-runtime.md` の WP-B。

## IF 変更通知

**Python から見える `_core` の IF（`Node` / `Runtime` / `NetworkError` / `PauseTimeoutError` の名前・引数・戻り値・例外）は計画書のとおりで、変更なし。** 計画書が決めていなかった振る舞いを次のように決めた（A/C/D への影響は D の呼び方だけ）。

| 呼び出し | `start` 前 | 正常に終了した後 | 例外で終了した後 |
| --- | --- | --- | --- |
| `shutdown()` | 何もしない | 何もしない | `NetworkError` |
| `check()` | 何もしない | 何もしない | `NetworkError` |
| `step()` / `capture()` | `RuntimeError` | `RuntimeError` | `NetworkError` |
| `join()` | `RuntimeError` | `True` | `NetworkError` |
| `last_update_ns()` | `RuntimeError` | 最後の値 | 最後の値 |

- `capture` の `build` と `on_exception_capture` は制御スレッドで呼ばれる。そこから待つメソッド（`step`・`capture`・`shutdown`・`join`）を呼ぶと、デッドロックの代わりに `RuntimeError` を送出する。ノードの hook から呼んだ場合も同じ。D が `Network` のメソッドをノードから呼べるようにする場合はこの制約を受ける
- `Runtime` を参照が切れて解放すると、待たずに終了を要求する（GIL を持ったまま待つとデッドロックするため）
- `NetworkError` のメッセージは `the network stopped because node '<name>' raised in <hook>`。Rust の panic のときの `__cause__` は panic のメッセージを持つ `RuntimeError`
- `_core.pyi` は先頭の import 行（`Callable, Mapping, Sequence`、`Self`）も変えた。WP-A が同じ行を変えていれば、マージで行が衝突する

## 計画から変えた点（Rust の IF。使うのは B の bindings だけ）

- **ホストが待つメソッドは、ブロックせずに `Reply<T>` を返す**（`Runtime::start` は `(Runtime, Reply<()>)`、`step`・`capture`・`shutdown` も同様）。bindings は `Reply::wait_timeout(50 ms)` を GIL を外して繰り返し、その合間に `check_signals` する。計画書の「ブロックして `Result` を返す」形では、Ctrl-C を見る合間を作れない
- `capture(timeout: Option<Duration>, build)` は timeout を取る（`None` なら `RuntimeConfig::pause_timeout`）。`join(timeout) -> Result<bool, RuntimeError>` は例外で終了していれば `Err(Failed)`
- `RuntimeError` に `InvalidArgument(String)`（長さの不一致・範囲外のノード番号・不正なタイマーの状態 → `ValueError`）と `Reentrant`（上記 → `RuntimeError`）を足した。`Runtime::names()` を足した
- `time.rs`: `Wait` enum を足した。既存の `wait_until_ns` は `wait_until_ns_or(.., || false)` に委ねる形に変えた（振る舞いは同じ。同じループを 2 つ持たないため）。テストは既存の `crates/mujou-core/tests/time.rs` に足した
- Rust のテストは既存の慣例（`crates/mujou-core/tests/`）に合わせ、公開 IF を使うものを `tests/runtime.rs` に置いた。`#[cfg(test)]` には private な凍結の理由の組（`Freeze`）だけを置いた
- 起動時のタイマーの凍結と `load_state` は、制御スレッドではなく `Runtime::start`（ホストのスレッド）が、スレッドを作る前に行う。まだランタイムのスレッドが無いので競合しない。タイマーの状態の検証をスレッドを作る前に済ませて、不正なら何も起動せずに `ValueError` にできる

## 実装中の判断

- **停止から再開した直後に終了が来ても `on_resume` を呼ぶ。** `capture` が再開（`Run`）を出し、ノードが起きる前に `shutdown` が `Stop` を出すと、ノードは `Stop` だけを見て `on_resume` を飛ばしていた（Rust のテストで 1/10 程度で再現）。`Control::run_epoch`（最後に `Run` にした epoch）を持ち、自分の停止より後に `Run` があったら `on_resume` を呼んでから区切りで止まる。停止中のまま終了したとき（例外など）は従来どおり `on_resume` を呼ばない
- 停止の要求ごとに epoch を進め、ノードは「自分が止まった epoch」を報告する。タイムアウトで再開した直後に次の `capture` が来ても、前の停止の報告を新しい停止と取り違えない
- 区切りで指示を見るのと時刻を読むのを同じ制御の lock の下で行う。停止の要求より前に読んだ時刻は凍結時刻 T 以下になり、要求の後は必ず停止を見る
- hook の例外のログは、捕まえた場所（bindings の `PyNode`）で必ず出す。最初の例外もログに出す（`docs/execution-model.md` の「例外による終了」1）。コアは最初の 1 つだけを `NodeFailure` に記録する
- Rust の panic は `catch_unwind` で捕まえて例外と同じに扱う。hook 名はコアのメソッド名（`update` など）
- `now <= last_ns` を「Δt = 0」として扱う（復元した `last_update_ns` がタイマーより先にある場合も更新しない）。自由実行では `wait_until_ns_or(last_ns + 1, ..)` で待ち、制御の要求（epoch の変化）でも起きる
- `_core.Node` は `frozen` の pyclass（Python のサブクラスの属性は `__dict__` に入るので影響なし。借用フラグを持たない）
- 手動モードの `step` で、ノードの時刻が前回から進んでいなければ `on_update` を呼ばずに完了とする

## 確認したこと

- `cargo test` の runtime を 60 回、`tests/mujou/test_core_runtime.py` を 30 回、`PYTHON_GIL=1`（3.14t で GIL を有効にする）で 10 回、6 並列の負荷の下で計 48 回回し、すべて通った

## 残った懸念・確認事項

- 正常な終了と例外の終了処理にはタイムアウトが無い（計画どおり）。止まらないノードがあると `shutdown` は戻らない（Ctrl-C では抜けられる）
- 自由実行・実時計のノードはコアを使い切る（計画の想定リスクどおり。目標周期は future work）
- インタプリタ終了時に走ったままのランタイムは D の `atexit` で止める前提
