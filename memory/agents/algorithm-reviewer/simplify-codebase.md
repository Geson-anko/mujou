# algorithm-reviewer: コードベース全体の簡素化のうち実行基盤に触れる変更

対象: `git diff 85e2ee7..HEAD -- crates/ src/mujou/_core/`（worktree `/Users/geson/Projects/mujou-simplify`、ブランチ `refactor/2026-10-19/simplify-codebase`、`ca25ad4` まで）。担当の記録は `memory/agents/code-simplifier/simplify-a.md`（Rust コア）・`simplify-b.md`（bindings）・`simplify-c.md`（Python）。観点は実行モデルの不変条件（ノードは互いを待たない、停止中は時間が進まない、Δt は実際の経過、同じノードを 2 つのワーカーで同時に走らせない、勾配は hook の前に毎回切る、Port は所有ノードだけが読み書きする）。更新式（PC-ALM など）は diff に無い。

## verdict

**approve**。確信度: 高。

どの変更も振る舞いを変えていない。不変条件を弱める経路は見つからなかった。指摘は should-fix 1 件（一本化で暗黙になった不変条件を doc に書く）と nit 3 件だけ。

## must-fix

なし。

## should-fix

### S1. `Shared::last_update_ns` に、プールのノードの読み書きが制御の lock の下に限られることを書く（確信度: 中）

- `Task.last_ns` は `Control` の中にあったので、「プールが Δt の起点として読む値は制御の lock の下でしか変わらない」ことが型の配置から保証されていた。一本化した後のフィールド `last_update_ns: Vec<AtomicU64>`（`crates/mujou-core/src/runtime/mod.rs:185`）には doc が無い。プールが lock の下で読む箇所（`crates/mujou-core/src/runtime/pool.rs:151`, `:330`, `:345`）が前と同じ値を見る根拠は、次の 3 点がそろっていることだけになった
  1. プールのノード `i` の書き手は、制御の lock を持った `record`（`pool.rs:438`）だけ
  2. 初期値はスレッドを起こす前に書く（`mod.rs:521-528`）
  3. lock を取らずに書く専用スレッド（`crates/mujou-core/src/runtime/node.rs:137`）は自分の添字にしか書かない
- 今のコードはこの 3 点を満たしているので、等価性の判定は変わらない（照合メモ）。ただ、将来たとえば restore や `capture` から lock の外で `last_update_ns` を書く変更を入れると、プールの Δt の起点と `target_ns` がずれても型からは気づけない。フィールドに「各ノードの次の Δt の起点。プールのノードの分は制御の lock の下でだけ書く（専用スレッドは自分の分だけを lock なしで書く）」と 1〜2 行書いておくとよい

## nit

### N1. `PyNode::method` のキャッシュ参照が名前の線形探索になった（確信度: 高、任意）

- `crates/mujou-python/src/runtime.rs:233-236` は hook を呼ぶたびに `CACHED_HOOKS.iter().position(..)` で 9 個の文字列を比べる。`call_as` は 1 回の hook で `_before_hook` と本体の 2 回引く。以前の `Methods::get` の `match` も文字列の比較で、GIL の取得や `Context.run` に比べれば無視できる。正しさには関わらない

### N2. 制御スレッドから自分のランタイムを呼ぶと `Reentrant` になることのテストが Rust 側に無い（確信度: 中、任意）

- `spawn`（`crates/mujou-core/src/runtime/mod.rs:696-710`）に寄せたことで、制御スレッドの印は `Controller` を組み立てる前に付くようになった（以前は `Controller::run` の先頭）。どちらも `started.send` と最初の job より前なので、観測できる違いは無い
- Rust の結合テストが印を確かめるのはワーカーの場合だけ（`crates/mujou-core/tests/runtime.rs:1619-1646` の `a_hook_on_a_worker_cannot_wait_for_its_own_runtime`）。専用スレッドと制御スレッド（`state_dict` は制御スレッドで呼ぶ）の場合は Python 側のテストに頼っている。helper に寄せたことで付け忘れは構造上起こらなくなったので任意

### N3. `Run.__exit__` の再送出で traceback に 1 フレーム増える（確信度: 高、任意）

- `src/mujou/_core/run.py:354-358` の `except _native.NetworkError: if exc is None: raise` は、以前の「try の外で `shutdown()` を呼んで素通しする」書き方と比べ、traceback に `__exit__` の except 節が 1 行増える。例外の型・`__cause__`・`__context__` は変わらない（素の `raise` は `__context__` を付けない）。表示だけの差なので任意

## 検証結果

- `just run`（worktree）: exit 0。cargo test は lib 33 / runtime 49 / time 37 が全て pass、pytest 595 passed / 20 skipped、pyright 0 errors、mkdocs strict 成功。format 後の差分なし（`git status` clean）
- 不変条件ごとの判定
  - ノードは互いを待たない: スレッドの起こし方を helper に寄せただけで、lock の取り方・起こし方・順序は変わっていない（`pool.rs`・`control.rs`・`node.rs` の diff は import と印の移動と `last_ns` の読み先だけ）
  - 停止中は時間が進まない: 凍結と再開のコードは diff に無い。`time::remaining`（`crates/mujou-core/src/time.rs:108-113`）は実時間の締め切りまでの残りで、4 か所の元の式と同じ（`map_or(Duration::MAX, saturating_duration_since)`）。タイマーの時刻には触れない
  - Δt は実際の経過: プールの Δt は `now_ns - last_update_ns[i]`（`pool.rs:151-152`, `:345-351`）。以前の `now_ns - task.last_ns` と同じ値（照合メモ）。`a_restored_node_counts_its_period_from_its_last_update`（`tests/runtime.rs:973-991`、Pool・OneWorker を含む each_mode）と `a_step_without_time_passing_does_not_update`（each_mode）が、復元した起点と Δt = 0 の分岐を確かめている
  - 同じノードを 2 つのワーカーで同時に走らせない: `take`・`run_work`・`place` の `At::Running` の遷移は不変。`c.sched.tasks[i].at = At::Running`（`pool.rs:149`）は借用の書き方を変えただけ
  - 勾配は hook の前に毎回切る: `call_as` の `_before_hook` を飛ばす条件は `hook`（呼ぶ名前）で判定する `!matches!(hook, "_on_enter" | "_after_update")`（`crates/mujou-python/src/runtime.rs:262`）のままで、報告名 `reported` を変えても影響しない。`on_update`・`on_pause`・`on_resume`・`on_exception`・`on_stop`・`on_start`・`load_state_dict` の前に今までどおり呼ばれる
  - Port は所有ノードだけが読み書きする: `Binding._receiving`（`src/mujou/_core/binding.py:106-110`）は `_check_owner("receives on")`（`current_node() is self._owner`）を先に、recv spec の検査を後に行う。順序・判定・文言とも以前の `recv`・`has_received` と同じで、受信の経路は 2 つともこれを通る

## 照合メモ

- `Task.last_ns` と `last_update_ns[i]`（プールのノード `i`）
  - 旧: `last_ns` の書き込みは `WaitGo → Loop` の遷移（`last_update_ns[i]` を読んで代入）と `record` の `Work::Update`（`last_update_ns[i].store(now_ns)` と同時に `task.last_ns = now_ns`）の 2 か所。どちらも制御の lock の下
  - `last_ns` を読むのは `run_worker` の Δt と `advance` の手動モードの分岐だけで、どちらも `Phase::Loop` のときに限る。`Loop` に入る経路は `WaitGo` からの遷移と `Paused → Resume`（`Paused` は `Loop` からしか来ない）だけなので、読む時点では必ず 1 回は `WaitGo` の代入を経ている
  - `last_update_ns[i]` の書き手は、初期化（スレッドを起こす前）・プールの `record`・専用スレッドの `node_loop` の 3 つで、専用スレッドはプールのノードの添字に書かない。`Loop` の間に `last_update_ns[i]` を変えるのは `record` だけで、そこでは `last_ns` も同じ値にしていた。よって読む時点で常に `task.last_ns == last_update_ns[i]`
  - 順序: どちらも制御の lock の下で読み書きするので、lock による happens-before が以前と同じに効く。`SeqCst` の atomic を lock の下で読むのは冗長なだけで、意味は変わらない
- `spawn` helper: 専用スレッド・ワーカー・制御スレッドのいずれも、印（`RUNTIME_OF_THREAD.set`）は `N::on_thread` の closure の中で、本体（`run_node`・`run_worker`・`Controller::run`）より前に付く（`mod.rs:704-707`）。以前もこの 3 つの本体の先頭で付けていたので、`on_thread` の内側である点は同じ。制御スレッドの起動に失敗したとき、ノードのスレッドの `JoinHandle` は `control` closure と一緒に drop される（`std` は起動に失敗した closure を drop する）。以前の `Controller` の drop と同じ
- `fail(i, e)`: `NodeFailure { node: i, hook: e.hook, error: e.source }` の組み立てを中に移しただけ。最初の失敗だけを記録して停止を頼む処理は不変
- `check_start`: 検査の順（names → dedicated → last update times → periods の長さ → 0 の周期）と文面 `got {n} nodes but {len} {what}` は以前と同じ
- `CACHED_HOOKS`: 名前と getattr の順は旧 `Methods::of` と同じ（失敗したときに報告する最初の属性も同じ）。キーに無い名前（`on_start`・`load_state_dict`）は以前どおり名前で引く。探索に失敗したときは `methods` を `None` のまま `on_start` のエラーとして返す。`_after_update` の報告名は旧 `public()` の代わりに `update` で `call_as("_after_update", "on_update", ..)` と明示した。`call` を `_after_update` で呼ぶ箇所は他に無い（`git show 85e2ee7:crates/mujou-python/src/runtime.rs` で確認）。`_on_enter` は以前から `enter` が `call_as` で次の hook の名前を報告している
- start の attach の統合: 探索とエラーの記録（`self.error` → `log_error`）を 1 回の `Python::attach` の中で行う。このスレッドは `on_thread` で Python のスレッド状態を持ったままなので、attach の回数が減っても GIL の扱いは同じ
- `Run.__exit__`: body の例外が無いときは `shutdown()` の全ての例外が素通しになる（`NetworkError` は素の `raise`）。body の例外があるときは `NetworkError` だけを握りつぶし、他は素通しする。以前と同じ
- `node.py`: `_forget_state_name` は以前の `elif name in names` / `__delattr__` の削除と同じ。`_check_reserved` は `inspect.getattr_static(type(node), name) is not vars(Node)[name]` で以前と同じ判定。文言は `period` の側だけ "pass period to Node.__init__ instead" に変わった（テストは "reserved" で照合）
- `component.py`: `ports_of`・`children_of` を 2 回呼ぶようになったが、間に入る `is_data_descriptor`・`_check_not_built`・`assign_owner` はこれらを変えない
