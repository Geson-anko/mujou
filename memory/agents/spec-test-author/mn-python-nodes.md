# M:N PR 2（Python のノードをワーカーに載せる）の仕様テスト

計画: `memory/agents/implementation-planner/mn-scheduler.md`（「PR 2」「テスト観点 / Python」「ユーザーの決定（2026-10-18）」）。以下「計画 L<n>」はその行番号。

## 書いたテストと仕様根拠

### `tests/mujou/_core/test_native_runtime.py`（bindings）

- `TestStart`
  - `test_hooks_run_in_order_on_a_dedicated_thread`（既存を改名し `dedicated=[True, True]` に限定）: 専用スレッドは今までどおり（計画 L19、L180）
  - `test_hooks_on_the_workers_run_in_order_without_on_thread_start`: 既定は全ノードがワーカー、`on_thread_start` は専用スレッドだけ（決定 Q1、計画 L31）
  - `test_on_thread_start_is_called_only_for_the_dedicated_nodes`: 混在で dedicated 側だけ呼ぶ（計画 L137）
  - `test_an_error_while_starting_stops_before_any_update`: `on_thread_start` の場合は dedicated に限定（pool では呼ばれないため）
  - `test_the_hooks_of_a_dedicated_node_share_one_python_thread_state`（既存の `test_hooks_share_one_python_thread_state` を専用スレッドに限定。決定 Q3、計画 L180）
  - `test_invalid_arguments_start_nothing[workers|dedicated]`: `workers=0` と `dedicated` の長さ違いは `ValueError`、何も起動しない（計画 L31、L168）
- `TestWorkers`
  - `test_each_node_keeps_its_own_contextvars_on_a_shared_worker`: workers=1 で 2 ノードが各自の ContextVar を保つ。ホストには漏れない。`state_dict` は文脈の外なので除外（計画 L109、L179、決定 Q3）
  - `test_the_hooks_run_on_at_most_the_given_number_of_threads`: 32 ノード、workers=2 で `on_update` のスレッドが 2 種以下（計画 L181）
- `TestFrameworkHooks`（`_on_thread_start` → `_on_enter`）
  - `test_the_framework_enters_a_node_on_a_worker_before_its_hooks` / `..._dedicated_node_before_on_thread_start`（専用スレッドは `_on_enter` 1 回だけ）: 計画 L32
  - `test_a_worker_enters_a_node_only_when_it_switches_to_it`: workers=1、b,a,a,b の手動ステップで a の `_on_enter` は「切り替え時だけ」（計画 L111、L162）
  - `test_an_error_in_the_first_enter_is_reported_as_on_start` / `..._later_enter_is_reported_as_on_update`: 計画 L32、L182。旧テスト「on_thread_start の名前で報告」は置き換えた（コアの `enter` は次の hook 名で報告するので、専用スレッドでも `on_start`）

### `tests/mujou/_core/test_node.py`

- `Probe` に `dedicated_thread` 引数。`thread_ident` の import を削除（src から消えるため）
- `TestThreadSetup.test_gradients_stay_off_in_every_hook_of_a_node_sharing_a_worker`: workers=1、a が毎 hook の最後に `set_grad_enabled(True)` しても b の全 hook で偽、ホストは真のまま（計画 L104、L184、決定 Q3）
- `TestDedicatedThread`: 既定 False（`super().__init__` を呼ばない場合も）、bool 以外は `TypeError`、クラス属性での置き換えは `mujou.run` で `TypeError`（reserved）、専用ノードは全 hook が 1 本の自分のスレッド（計画 L20-22、L186）
- `TestOnThreadStart`
  - `test_runs_on_the_dedicated_thread_after_the_settings_and_before_on_start`: 既存を dedicated に。on_start より前、勾配は記録しない、generator の続き、on_update も同じスレッド
  - `test_overriding_it_on_the_workers_is_refused_before_anything_starts`: `TypeError`（"dedicated_thread=True" を案内）、どのノードの on_start も呼ばれない、override を消せば同じネットワークで再び run できる（決定 Q1、計画 L23、L187）
- `TestCuda.test_nodes_sharing_a_worker_keep_their_own_device_and_stream`（cuda マーカー、ローカルでは skip）: workers=1 で 2 ノードが毎回自分の device と自分の stream（相手が既定の stream に戻して去っても）（計画 L105-106、L185）

### `tests/mujou/_core/test_port.py`

- `TestOwnerOnAWorker.test_only_the_owning_node_reads_and_writes_on_a_shared_worker`: workers=1 で別ノードの Port の recv / send は `RuntimeError`「only the node that owns」、所有ノードは読める（計画 L47、L107、L189）。ホストからの検査は既存の `test_only_the_owning_node_*` が既定（pool）のまま担う

### `tests/mujou/_core/test_run.py`

- `TestWorkers`
  - `test_rejects_what_is_not_a_positive_int_before_starting`: `True`・`1.0` は `TypeError`、0・-1 は `ValueError`、何も起動せず再試行できる（計画 L17、L192）
  - `test_more_nodes_than_workers_all_update_exactly`: 8 ノード・workers=2・仮想タイマーで 3 周期。各ノードの状態がホストで計算した参照（自分の seed の乱数と dt=周期の Euler）とビット単位で一致、スレッドは 2 種以下
  - `test_a_manual_run_gives_the_same_bits_on_workers_and_dedicated_threads[False|True]`: 論理 CPU 数 + 2 個の鎖を手動ステップ（鎖の順）で 4 回。ワーカーでも専用スレッドでも、ホストで同じ順に計算した参照とビット一致（= 両者がビット一致）。計画 L194「手動ステップは pool でも同じ」
  - `test_the_default_shares_threads_among_more_nodes_than_cpus`: 既定のワーカー数は min(ノード数, 論理 CPU 数) なので、CPU 数を超えるノードはスレッドを共有する（計画 L17）
- hook の中の判定（pool のノードから `mujou.run` は `RuntimeError`）は既存の `TestHostMethodsFromANode.test_raise` が既定（pool）のまま担うので足していない

### その他

- `tests/mujou/conftest.py`: `start` fixture に `workers` を通す（常に `mujou.run(workers=...)` を渡す）
- `tests/mujou/test_testing.py`: `test_run_takes_the_arguments_of_mujou_run_but_workers`（`mujou.testing.run` には `workers` を足さない。計画 L19。plan-implementer からの依頼で更新）

## red の確認

HEAD d1ca90f（PR 2 の実装前の src/crates）の別 worktree にこのテストを置いて `pytest tests/mujou -m "not slow"` を実行: 64 failed / 473 passed。失敗の内訳は
- `Runtime.start()` に `workers` / `dedicated` が無い（TypeError）、pool では `on_thread_start` が先頭に来る、`_on_enter` が呼ばれない（DID NOT RAISE）
- `Node.__init__` に `dedicated_thread` が無い、`mujou.run` に `workers` が無い（conftest の `start` を使う既存テストもここで赤になる。実装が入れば緑）
- 予約名のメッセージ（"reserved"）が出ない、`mujou.run` の引数に `workers` が無い

実装中の worktree（plan-implementer の途中の src）では 537 passed / 14 skipped（slow 除外）。新しいテストを 15 回繰り返して全部通った（揺れなし）。

## 実装側への注意

- `on_thread_start` の override の検査は、クラスから override を `del` すれば同じネットワークを再び run できること（検査は起動前、ネットワークを使い切らない）をテストが要求する
- `_on_enter` で戻す CUDA の stream は、最初に作ったものを保持して毎回同じものにすること（`len(set(streams)) == 1`）
