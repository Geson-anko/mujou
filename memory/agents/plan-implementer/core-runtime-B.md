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

## レビュー差し戻しへの対応（2026-10-16）

対象: `memory/agents/code-reviewer/core-runtime-B.md`・`memory/agents/algorithm-reviewer/core-runtime-B.md`（main checkout 側）。orchestrator の裁定に従い全件対応した。

### IF 変更通知（追加分）

- **`GlobalTimer` に停止の保持を足した（Rust）。** `GlobalTimer::hold() -> PauseHold<'_>` は保持を 1 つ増やし、`PauseHold` を drop すると減る。保持が 1 つでもあるか、`pause()` のスイッチが入っていれば時刻は止まる。`resume()` はスイッチを切るだけで、保持は解かない。`is_paused()` はどちらかで止まっていれば真。仮想時計の `advance` は、どちらかで止まっていれば `TimerError::Paused`。Python の `_core.timer` の API は変えていない（`pause`/`resume` がスイッチ）。`lib.rs` の `pause` の docstring（"For the framework's pause and state capture"）は、ランタイムがもうスイッチを使わないので古くなった。`lib.rs` は触らない約束なので、D か後続で直す。`docs/execution-model.md` のタイマーの実装の節に 1 項目足した
- ランタイムはタイマーの停止にスイッチを使わず、保持だけを使う。そのため、ホストが `pause()` してから `start` しても、ホストの停止は残る。2 つの Runtime の状態の取得が重なっても、片方の終わりがもう片方の停止を解かない
- `NodeHooks` に既定の実装つきで 2 つ足した: `fn on_thread<R: Send>(body: impl FnOnce() -> R + Send) -> R`（ランタイムのスレッドの本体を包む口。bindings は `Python::attach(|py| py.detach(body))`）と `fn panicked(&mut self, error: &HookError)`（panic を知らせてログに出させる）
- `RuntimeError::Internal(String)`（Python では `RuntimeError`）を足した。スレッドを作れなかったときと、`capture` の `build` が panic したとき
- panic のときの hook 名を Python と同じ名前（`on_update` など）にそろえた
- 再入の判定（`Reentrant`）を Runtime ごとにした。Runtime A の `build` から Runtime B の `capture` を呼べる
- Python: `start` が `ValueError`（引数の誤り）で失敗したときは何も起動せず、もう一度 `start` を呼べる

### 対応の内容

- M1: ノードのスレッドと制御スレッドの本体を `NodeHooks::on_thread` で包んだ。bindings はスレッドの寿命の間 Python のスレッド状態を 1 つ保ち、本体の間は detach する。hook ごとの attach は既存のスレッド状態を使い直すので、`threading.local` と `threading.current_thread()` が hook 間で同じになる（回帰テスト `test_hooks_share_one_python_thread_state`）
- 停止の保持: 上記。`Freeze` は理由の集合が空でない間 `PauseHold` を 1 つ持つ。回帰テストは Rust の `tests/time.rs` の `hold` モジュールと、Python の `test_one_capture_does_not_end_the_pause_of_another`（A の `build` の中で B の `capture` を呼ぶ。スレッドを使わずに決定的にできる）・`test_a_pause_by_the_host_outlasts_the_runtime`
- N2: 自由実行の区切りは、指示と epoch を 1 語にした atomic（`epoch << 2 | command`）を読み、時刻を読み、もう一度読んで一致したときだけその時刻を使う。停止中の時刻が T 以下になる理由は `Shared::boundary` のコメントに書いた（制御側は語を書いてから凍結し、凍結がタイマーの lock の下で先に起きていれば、2 回目の読みが新しい語を見て時刻を捨てる）。手動モードと停止・終了の処理は従来どおり lock の下
- N7: 2 回以上呼ぶ hook（`on_update`・`on_pause`・`on_resume`・`on_exception`・`on_stop`・`state_dict`）の bound method を、`start` の最初にノードのスレッドで 1 回だけ引いて持つ
- algorithm should-fix 1: `last_update_ns` は `Runtime::start` でタイマーを凍結して状態を読み込んだ直後、スレッドを作る前に設定する（起動中の例外で取得した値が 0 にならない。テスト `test_the_start_of_each_dt_is_known_when_starting_fails`）
- algorithm should-fix 2: `test_the_pause_takes_no_global_time`（実時計で `build` の中で 0.3 s 眠り、再開後の最初の Δt < 0.25）
- algorithm should-fix 3 / code S3: `test_restores_the_time_and_where_each_dt_starts`（`on_start`・`load_state_dict` の中では時刻が T でタイマーが停止中、最初の Δt が `(k + h) / 1e9` に完全一致、T より先の `last_update_ns` では更新しない）
- code S2: 上記 IF。テスト `test_invalid_arguments_start_nothing`（3 通り。何も起動せず、タイマーは止まったまま残らず、その後 `start` できる）
- code S4: `test_a_runtime_without_references_stops`
- N1: `test_waits_without_spinning_until_the_time_moves` は、0.2 s の間のプロセスの CPU 時間が 0.05 s 未満であることを見るようにした（空回りなら 1 コアを使い切る。負荷が高いと見逃すことはあるが、誤って落ちることはない）
- N4: 上記（panic の hook 名とログ）。N5: スレッドを作れなかったら、作ったスレッドを `Stop` と `stop_allowed` で自分で終わらせ、保持を解いて `Internal` を返す（GIL を持っているかもしれないので join しない）。N6: 上記。N8: `start` の docstring に Ctrl-C のときの振る舞いを書いた
- algorithm nit 3: `build` の panic は捕まえて再開してから `Internal` を返す。`ExceptionCapture` の panic は捕まえて終了処理を続ける
- algorithm nit 1（記録だけ）: タイマーを先に再開するので、再開後と起動後の最初の Δt には、制御側の 2 つの操作の間隔に加えて、`on_resume` の実行時間とノードのスレッドが起きるまでの遅れも入る。通常のループ 1 周の手間が Δt に入るのと同じ性質で、正しさの問題ではない。`docs/execution-model.md` の「Δt に混ざるのは制御側の 2 つの操作の間隔だけ」は D が実装の節を書くときに合わせる
- algorithm nit 4（`run_epoch` の回帰テスト）は入れていない。Rust の hook に Barrier を挟めば決定的にできるが、今回は見送った

### 測定（開発機の macOS、`hw.ncpu` = 14。Rust は release ビルド）

自由実行・実時計で、何もしない Rust のノード（ノードごとのカウンタ）の、1 秒あたりの更新回数の合計:

| ノード数 | fast path あり | 毎回 lock（従来） |
| --- | --- | --- |
| 1 | 16.3M | 15.0M |
| 2 | 11.8M | 12.0M |
| 4 | 6.0M | 6.3M |
| 8 | 7.1M | 6.7M |
| 16 | 7.9M | 7.4M |

fast path の効果は数 % にとどまる。合計はノード数によらず 600 万〜800 万回/秒で頭打ちになり、ボトルネックはタイマーの読み取りの `Mutex`（`time_ns`）に移った。`docs/open-decisions.md` の「時刻の読み取りが Mutex を取ることによる競合」がこれに当たる（seqlock などで解く。今回の範囲外）。

何もしない Python の `on_update`（自由実行・実時計）の 1 秒あたりの回数:

| ビルド | ノード 1 | ノード 2 | ノード 4 |
| --- | --- | --- | --- |
| 3.14t（free-threaded） | 8.70M（0.11 µs/回） | 4.81M | 2.50M |
| 3.14t、`PYTHON_GIL=1` | 7.66M（0.13 µs/回） | 3.37M | 1.29M |

レビュー時（M1 の前）は約 1.8 万回/秒（約 55 µs/回）だった。同じメソッドを Python のスレッドから直接呼ぶと 0.025 µs/回。M1 と N7 の効果を分けては測っていない。ノードを増やすと合計が落ちるのは、タイマーの `Mutex` と、hook ごとの attach/detach の競合だと考えている（未検証）。
