# algorithm-reviewer: M:N PR 3（ベンチマーク）のスケジューリングの変更

対象: `84a3102`（timekeeper を起こすのは、どのワーカーもキューを見ていないときだけ）、`b1dca76`（周期の `Option` を消す）、`d5177c7`（GIL ありの既定のワーカー数）。worktree `/Users/geson/Projects/mujou-mn3`、ブランチ `feature/2026-10-19/mn-benchmark`（`76d6b2b` まで）。観点: 実行モデルの不変条件（期限の来たノードの取り残し、周期の規則と Δt、過負荷での飢え、ノードが互いを待たないこと、停止中に時間が進まないこと）。更新式そのものは diff に無い。

## verdict

**approve**。確信度: 高。

起こし方を減らしたことで期限の来たノードが取り残される経路は見つからなかった。ワーカーが戻した期限は、そのワーカー自身が同じ lock の保持のまま次のループで必ず見る（期限が来ていれば自分で取り、来ていなければ timekeeper の期限と比べて起こす）。制御の側から置いた仕事は、従来どおり起こす（評価し直し 1 回につき 1 度にまとめただけ）。周期の規則・Δt・期限順の取り出し・停止とタイマーの凍結のコードは変わっていない。`b1dca76` は振る舞いを変えない。`d5177c7` はワーカーの本数を変えるだけで、スケジューリングの規則には触れない。

## must-fix

なし。

## should-fix

なし。

## nit

### N1. 新しいテストが「ワーカー 2 本以上」を保証していない（確信度: 高）

- `period::a_target_a_worker_puts_back_before_the_timekeepers_is_kept`（`crates/mujou-core/tests/runtime.rs:960-987`）のコメントは「Needs a second worker」と書くが、`Mode::Pool` は `workers` を既定（`min(プールのノード数, available_parallelism)`）にし、渡した `config.workers` も上書きする（`crates/mujou-core/tests/runtime.rs:121-125`）。CPU が 1 つの環境ではワーカーが 1 本になり、このテストは新しい規則（`crates/mujou-core/src/runtime/pool.rs:160-163`）を通らずに通る
- 規則を通るかどうかは `thread::sleep(QUIET)`（`runtime.rs:977`, `:980`）の間にもう 1 本が timekeeper になるかにも依る。結果（合否）は眠る長さに依らないので決定性の問題ではないが、「規則を外すと落ちる」ことは確率的にしか保証されない
- この規則は、ワーカーが戻した期限を timekeeper に渡す唯一の経路で、無いと仮想時計では止まる（実装者の記録どおり）。GitHub の runner は 2 コア以上なので CI では通るはずだが、`Mode` を通さずに `workers: NonZeroUsize::new(2)` を明示して始めるほうが、コメントの前提をテストが守る

### N2. 文書の「GIL ありのビルド」と実装の「今 GIL が有効か」がずれる（確信度: 高）

- 判定は bindings の `default_workers` が `sys._is_gil_enabled()` で行う（`crates/mujou-python/src/runtime.rs:451-466`）。free-threaded のビルドでも `PYTHON_GIL=1` や対応しない拡張の import で真になる。`specs/native-core.md` と bindings の docstring は「GIL が有効なとき」と正しく書いている
- 一方 `src/mujou/_core/run.py:444`（"on a build of Python with the GIL"）、`specs/nodes.md:46`（「GIL ありのビルドでは」）、`docs/en/user-guide/running.md` / `docs/ja/user-guide/running.md` の説明はビルドの話として書いている。スケジューリングの不変条件には影響しない（本数が変わるだけ）

### N3. Python のテストが実時間と CPU 数の近似に依る（確信度: 中）

- `TestWorkers.test_the_default_number_of_workers_all_take_updates`（`tests/mujou/_core/test_run.py:620-628`）は実時間 0.5 s で全ワーカーが 1 回以上更新を取ることを確かめる。負荷の高い CI で 0.5 s のうちに全ワーカーが起きる前提は、普通は満たされるが時間依存
- 期待値は `os.process_cpu_count()`（affinity だけを見る）から求め、コアの `available_parallelism()`（Linux では cgroup の CPU quota も見る）の代わりにしている（docstring に明記あり）。quota のあるコンテナでは一致せず落ちる。今の CI と dev container には quota が無いので現状は問題にならない

### N4. 「起こさない」側の回帰テストは無い（確信度: 高、任意）

- 正しさ（取り残さない）のテストは足されたが、性能上の性質（ワーカーが戻すたびに `interrupt_waiters` を呼ばない）を固定するテストは無い。数はベンチマークの一時パッチで測っただけ。必要なら `Timer` の waiter が起きた回数を数える小さなテストで固定できるが、正しさには関わらないので任意

## 検証結果

- `just run`: 成功（exit 0）。Rust のテストはすべて ok（`tests/runtime.rs` 48 件ほか）、pytest 595 passed / 20 skipped、pyright 0 errors、docs-build 成功。実行前後で作業ツリーは clean のまま
- `cargo test -p mujou-core --release --test runtime` を 30 回連続: 失敗なし
- 同じテストバイナリを 8 本同時に 3 巡（CPU に負荷をかけた状態）: 失敗なし
- `uv run pytest tests/mujou/_core/test_run.py tests/mujou/_core/test_node.py -k "Workers or period or Period"`: 22 passed（`test_a_period_of_zero_updates_as_soon_as_the_time_moves` を含む）

## 照合メモ

### 84a3102: 期限の来たノードが取り残されないか

`place` の呼び出し元は 3 つ（`schedule`・`take`・`run_work`）だけで、`schedule` を呼ぶのは制御の側の `wake_pool` / `wake_node` だけ（`pool.rs:175-190`、`control.rs:66,86,163`、`mod.rs:246,339,734`）。

- **ワーカーが戻す場合**（`take` の `pool.rs:258-262`、`run_work` の `pool.rs:413-419`）: どちらも同じ guard を `run_worker` のループに返し、lock を離さずにループの先頭へ戻る（`pool.rs:136-165`）。`left > 0` の間は必ず次のどれかになる
  - ready が空でなければ、それを先に取る。このとき ready にある仕事は、(a) このワーカー自身が今積んだ hook（自分で取る）か、(b) 制御の側が積んだもの。(b) なら積んだときに timekeeper をすでに起こしている（`place` が `true` → `wake_pool` / `wake_node` が `interrupt_timekeeper`）。ワーカーが積む hook は自分のノードの 1 つだけで、積んだ直後に同じ lock の保持のまま 1 つ取るので、ワーカーだけで ready が伸びることはない。timekeeper は ready が空のときにしかならない（`pool.rs:136-141` が先）。よって「ワーカーが早い期限を戻した後、ほかの hook を長く回している間、timekeeper が遠い期限で眠ったまま」という状態は、ready の hook が更新より先に回るという既存の優先順位（`pool.rs:7-9`）の範囲に収まる。従来も、起こされた timekeeper は ready を先に片づけてから期限を見ていた
  - ready が空なら `earliest` を見る。`advance` が `Next::Heap` を返すのは `!manual && command == Run` のときだけで（`pool.rs:333-353`）、同じ lock の中なので `earliest` も `None` にならない（`pool.rs:195-197`）。先頭の期限が来ていれば自分で取り、来ていなければ timekeeper がいないなら自分がなり、いて期限が早いなら起こす（`pool.rs:155-164`）
- **起こした後**: `keep_time` は `sched_epoch` を制御の lock の下で読んでから待ち（`pool.rs:215-223`）、`interrupt_timekeeper` も制御の lock の下で epoch を進める（`pool.rs:230-235`）。`Timer::wait_until_ns_or` は条件を timer の lock の下で見てから眠り、`interrupt_waiters` はその lock を取る（`time.rs:339-371`）。取りこぼしは無い。起きた timekeeper は `timekeeper = None` にしてからループに戻り、新しい先頭の期限で待ち直す
- **timekeeper の値が古い窓**（起きたが lock をまだ取っていない間は `Some(W)` のまま）: その間に別のワーカーが起こしても冗長な割り込みになるだけ。起こさずに `sched_changed` で待っても、timekeeper が lock を取った後に先頭を見直す
- **制御の側から置いた更新**: `wake_pool` は全ノードを評価し直した後に 1 度、`wake_node` は 1 度起こす。`look |= !dedicated[i] && schedule(i, c)` の右辺は毎回評価されるので、評価し忘れるノードは無い（`pool.rs:177-179`）。`place` が `Next::Heap` で timekeeper がいないときは `notify_one` を呼ぶ（従来どおり）
- **手動ステップ**: `earliest` が `manual` で `None` を返すので、手動モードには timekeeper がいない。`step` → `wake_node` → `place(Hook)` の `notify_one` で空いたワーカーが取る（`control.rs:78-96`、`pool.rs:278-286`）。変更の影響なし
- **停止からの再開**: 停止の要求は制御語を変えて timekeeper を起こし（`mod.rs:236-247`）、停止中は `earliest` が `None` なので timekeeper はいない。再開は `set_command(Run)` → `wake_pool` で `Resume` を ready に積み、`on_resume` を終えたワーカーが自分のノードを heap に戻して上の規則で見る。期限は停止の前のままで、タイマーが凍っていたので停止中の時間は入らない（`a_capture_does_not_move_the_target` が通る）
- **例外・終了の直後**: `fail` はワーカーのスレッドからでも制御の lock を取って `set_command(Stop)` → `interrupt_waiters` と `wake_pool` を呼ぶ（`mod.rs:326-342`）。失敗したノード自身は `Running` なので `wake_pool` が飛ばし、ワーカーが `exception` を ready の先頭に積んで自分で取る（`pool.rs:413-418`）。終了は `finish` が `stop_allowed` を立てて `wake_pool`（`control.rs:153-164`）

### 周期の規則と Δt

- `next_target_ns`（`node.rs:31-38`）は変わっていない（コメントだけ）。目標 = 前回の目標 + P、`now >= 目標 + P` なら `now + P` に数え直す
- Δt はプールでは `(now_ns - task.last_ns) / 1e9`（`pool.rs:151`）、手動では `pool.rs:349`。どちらも変更なし
- 0.0 秒: Python の `Node.__init__` が有限かつ >= 0 を確かめ（`node.py:183-186`）、`checked_period` が float のまま渡し、bindings の `period_to_ns` が `seconds_to_ns(0.0).max(1)` = 1 ns にする（`crates/mujou-python/src/runtime.rs:446-448`）。コアの `check_start` は従来の `None → 1`、`Some(0) → エラー`、`Some(ns) → ns` を、外側の `None → vec![1; n]`、0 を含めばエラー、それ以外はそのまま、に置き換えた（`mod.rs:711-721`）。Python は常に `periods` を渡すので、0.0 秒のノードの周期は従来どおり 1 ns
- 過負荷での飢え: heap は期限の早い順・同点は番号順（変更なし）。過負荷ではすべての期限が来ているので timekeeper はおらず、変更した起こし方は効かない。戻したワーカーが次に取るのも最も古い期限
- ノードが互いを待たないこと・停止中に時間が進まないこと: スロット、`Freeze`、`set_command` には触れていない。起こし方が減っただけで、待つ対象は増えていない

### d5177c7

- bindings が `workers` を省かれたときだけ `sys._is_gil_enabled()` を見て `max(1, available_parallelism / 2)` を渡す。コアの `min(プールのノード数)` はそのまま。本数が減っても、期限順の取り出しは本数に依らないので飢えない（`OneWorker` のモードのテストが同じ性質を見ている）
