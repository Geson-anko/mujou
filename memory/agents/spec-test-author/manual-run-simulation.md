# manual-run-simulation のテスト（spec-test-author）

計画書: `memory/agents/implementation-planner/manual-run-simulation.md`。tests/ 配下だけを変えた。

## 状態（2026-11-16）

書き終えた時点で plan-implementer の実装が先に入っており、`pytest tests`（slow 含む）は全部緑（927 + slow 23 passed、CUDA は skip）。`ruff check tests` と `pyright tests` もエラー 0。下の「期待される失敗」は、実装が無い状態で赤くなるはずのもの（仕様の根拠との対応）として残す。

## 既存テストの移行

- `mujou.testing.run(` → `mujou.run_manually(`、`from mujou.testing import ManualRun` → `from mujou import ManualRun`（conftest、test_execution、test_state、test_driver・test_port・test_state のサブプロセスの script、test_pyright、pc/test_input・test_layer・test_output、test_node のコメント）
- `tests/mujou/test_testing.py` を削除し、`test_execution.py` の `TestRunManually` / `TestManualRunStep` に移した。旧 `TestManualRun` も 2 つに分けて吸収
- conftest の `StartManual` / `start_manual` に `workers`
- `test_port.py` `TestSend.test_reaches_every_peer`: `run.step(H, source, *nodes.values())`
- 旧 `test_testing.py:48`: 後半を `sorted(order[2:]) == ["a", "b"]` に
- `test_init.py`: `ManualRun` / `run_manually` を `__all__` に、サブモジュール一覧から `mujou.testing` を外し、`import mujou.testing` が `ModuleNotFoundError` のテストを追加
- `tests/test_layers.py`: 層と `PUBLIC` から `mujou.testing` を外し、docstring を直した
- `native/test_runtime.py`: fixture の `manual: bool` は残し、`start(..., staging=Staging() if manual else None)` に（15 か所の呼び出しは不変）
- `wire(` は `staging` の既定が `None` なので変更不要

## 新しいテストと期待される失敗（実装が無い場合）

| テスト（test_execution.py） | 実装前の失敗 | 根拠（計画書） |
| --- | --- | --- |
| `TestRunManually::test_takes_the_arguments_of_mujou_run` | `AttributeError: run_manually` / 引数に `workers` が無い | 公開 IF: 引数名と順は `mujou.run` と同じ |
| `TestRunManually::test_rejects_less_than_one_worker_before_starting`、`test_runs_on_the_workers_given` | `TypeError`（workers 無し） | `workers` の意味と既定は `mujou.run` と同じ |
| `TestManualRunStep::test_advances_the_timer_by_dt_then_updates_in_the_given_order` | 通る（順を問わなくした） | step(dt) は全ノード 1 回ずつ |
| `test_every_node_of_a_step_sees_dt`、`test_every_message_of_a_step_carries_the_time_of_the_step` | workers の TypeError | 並列ステップのノードは dt == h、メッセージの時刻はステップの時刻 |
| `test_updates_a_node_of_an_infinite_period` | 通る見込み | 周期は無視 |
| `test_a_negative_or_infinite_dt_moves_no_time`（nan / inf 追加） | 通る見込み | Raises: dt が負か非有限 |
| `test_a_chain_steps_as_its_nodes_from_the_output_back[1,4]` | GS だと b, c が a の今の値を読み不一致 | 数値オラクル: chain の一致 |
| `test_a_loop_steps_from_the_values_at_the_start_of_the_step` | GS だと (1, 11) | 閉路の Jacobi（手計算 (1,10)(11,11)(12,21)） |
| `test_no_order_of_a_loop_is_a_step_of_every_node` | GS の木の順と一致して失敗（b_first=False） | どの逐次の順とも一致しない |
| `test_gives_the_same_bits_on_any_workers_and_threads` | GS の値で Jacobi の参照と不一致 | workers=1/4・dedicated_thread で同じ bit |
| `test_gives_the_same_bits_every_time` | 同上 | 再現性（workers=4 を 10 回） |
| `test_one_node_steps_as_it_does_by_name` | 通る | ノード 1 つなら step(dt) == step(dt, node) |
| `test_updates_the_nodes_at_the_same_time[pool/dedicated]` | 逐次では Barrier が 5 秒で BrokenBarrierError → NetworkError | 全ノードを同時に更新 |
| `test_the_host_reads_the_values_before_the_step_until_it_ends` | 保留が無いと止まっている間に 2.0 が見える | ステップ中のホストの読み出しはステップ前の値 |
| `test_a_subscription_gets_the_value_once_the_step_ends` | 保留が無いと Gate が止まっている間に届く | subscribe は commit の publish で起きる |
| `test_a_restored_run_goes_on_with_the_same_bits` | Jacobi の参照と不一致 | 復元して続けた結果が bit 一致 |
| `test_an_exception_in_a_step_stops_every_node` | 通る見込み（並列でない逐次でも成り立つ形） | 並列ステップの例外 → NetworkError、全ノード exception → stop |
| `TestCuda::test_a_cuda_chain_steps_as_its_nodes_from_the_output_back` | CUDA 無しで skip（未実行） | GPU: 並列ステップと逆順の逐次が一致 |

pc/test_chain.py: `TestChain::test_steps_to_the_same_bits_on_one_worker_and_on_several`（workers=1 と 4 の bit 一致）。

## 計画書と食い違った点

1. **pc.Chain で `step(h)` と `step(h, *from_next)` は一致しない（理論的に）**。計画書の「数値的オラクル / pc」と「chain なら出力側からの逆順が step(dt) と一致」は、メッセージが一方向に流れる chain でだけ成り立つ。PC の Layer は `prev` に `g_in` を送り返す（双方向）ので、出力側からの逆順では層 i が i+1 の**このステップの** `g_in` を読む（`g_in` 自体は i+1 の更新前の δ・θ と前ステップの h から作るので、TestJacobi の `Reference` とは一致する）。並列ステップでは前ステップの `g_in` を読むので、後ろ向きの信号が 1 ステップ遅れる。実測で 5 ステップ目から r・λ が 1e-2 のオーダーでずれた。`test_layer.py` の TestJacobi を `step(H)` に置き換えても失敗する。
   - 対応: そのテストは書かず、代わりに workers=1 と 4 の bit 一致を置いた。
   - 実装者・ドキュメントへ: `ManualRun.step` の docstring、docs（simulation.md、tutorial 3、concepts）、specs で「chain なら逆順と一致」と書くなら「メッセージが前にだけ流れる chain なら」に限定する必要がある。`test_layer.py` の `HandChain.from_next` の docstring「each reads what the others sent in the previous step (a Jacobi update)」も厳密には誤り（後ろ向きは同じステップの値）。`Reference` の「Jacobi」の意味もこれに合わせて確認が要る
2. `native/test_runtime.py` は呼び出し 15 か所を書き換えず、fixture の中で `manual` を `staging` に変換した（diff を小さくするため）
3. `TestManualRunStep::test_an_exception_in_a_step_stops_every_node`: ほかのノードの `update` は「走った」か「始まる前に止まった」のどちらでもよいとした（workers が少ないとき、まだ取られていないノードの update が走るかは計画書から決まらない）
