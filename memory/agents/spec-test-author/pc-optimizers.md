# pc-optimizers（spec-test-author のメモ）

計画書 `memory/agents/implementation-planner/pc-optimizers.md` のテスト観点をテストにした。変更したのは tests/ だけ。

## 状態（2026-10-22 時点）

並列で進んでいる src の実装に対して実行した。`tests/mujou/pc` 全体・`tests/mujou/test_init.py`・`tests/test_layers.py` は 213 passed（slow を含む）。赤のテストは無い。tests の該当ファイルに対する `ruff format` / `ruff check` / `pyright` も通った。`just format` は全ファイルに走るので、作業中の src に触れないよう実行していない（ruff だけを tests に当てた）。

## 書いた / 変えたテスト

| ファイル | 内容 | 仕様根拠 |
| --- | --- | --- |
| `tests/mujou/pc/test_optim.py`（新規） | TestSGD、TestAdamW、TestBound、TestState、TestChecks、TestTorchAdamW（slow、subprocess）、TestFreeThreaded（3.14t だけ、subprocess） | 計画書「テスト観点 / test_optim.py」「離散化」「例外」、K31 |
| `tests/mujou/pc/_reference.py` | `TAU = TimeConstants(h, lam)`、`TAU_THETA = 0.05`、`SGD`、`ADAMW`（tau_m=0.02、tau_v=0.5、w=0.1）。`NO_LEARNING` を削除。`_learn` は `TAU_THETA` を使う手書きの式のまま | 計画書「変更するファイル / tests」 |
| `tests/mujou/pc/test_layer.py` | `HandChain` / `Learning` に `optimizer` を追加。`NO_LEARNING` → `learn_interval=math.inf`。`layer_update` を `layer_gradients`（g_theta / B）と手書きの SGD に分け、`adamw_first_update` を追加。`test_without_learning_the_interval_changes_nothing` → `test_an_infinite_interval_never_learns`。`TestState` を SGD/AdamW で parametrize し、キーに `"optimizer"` を追加。新規 `TestOptimizer`（AdamW の初回の閉じた形、frozen な param） | 計画書「既存テストの書き換え」「既存のテストファイルへの追加」 |
| `tests/mujou/pc/test_output.py` | `optimizer=SGD`、`TAU_THETA`。新規 `TestOptimizer`（AdamW の初回の閉じた形） | 同上 |
| `tests/mujou/pc/test_chain.py` | `optimizer=` を渡す。新規 `test_each_node_binds_the_optimizer_to_its_own_parameters` | 同上 |
| `tests/mujou/pc/test_show.py`、`test_input.py` | `optimizer=` を渡し、theta を消す | 同上 |
| `tests/mujou/pc/test_functional.py` | `TestLearn` と `learn_` の import を削除。`TimeConstants(h, lam)` にし、mse のコメントを直す | 同上 |
| `tests/mujou/pc/test_time_constants.py` | theta を消し、lam の検査と「lam は inf 可」にする | ステップ 2 |
| `tests/mujou/pc/test_clock.py` | `TimeConstants(h, lam)` にし、`test_never_learns_with_an_infinite_theta` を削除 | ステップ 2 |
| `tests/mujou/test_init.py` | pc の `__all__` に `AdamW`、`BoundOptimizer`、`Optimizer`、`SGD` を追加し、functional から `learn_` を消す | ステップ 1 |
| `tests/test_layers.py` | `mujou.pc.optim` を `{"mujou.pc.messages", "mujou.pc.time_constants"}` の段に追加 | ステップ 1 |

## src が未実装のときに期待される失敗（対応表）

| テスト | 失敗 | 仕様根拠 |
| --- | --- | --- |
| test_optim.py 全体と、_reference.py を import するすべてのテスト | 収集時に `AttributeError: module 'mujou.pc' has no attribute 'SGD'` | 公開 IF（`pc.SGD` / `pc.AdamW`） |
| test_time_constants / test_functional / test_clock / test_input など | `TypeError`（theta が無い、または余計） | `TimeConstants(*, h, lam)` |
| `Layer` / `Output` / `Chain` を作るすべてのテスト | `TypeError: unexpected keyword argument 'optimizer'` | `optimizer: Optimizer` を必須にする |
| test_init | `__all__` の不一致 | `mujou.pc.__all__` に 4 つの名前を足し、functional から `learn_` を消す |
| test_layers | `mujou.pc.optim` の段が合わない | ステップ 1 |

## テストが前提にする実装側の契約（計画書では明記が薄いもの）

1. `BoundOptimizer.state_dict()` の形。AdamW は `{"m": list[Tensor], "v": list[Tensor], "c_m": float, "c_v": float}`（m・v は params と同じ順のリスト。c は束縛全体で 1 つの Python の float）。SGD は `{}`。test_layer / test_chain は `state["optimizer"]["m"]` の `len` を param の数と比べる
2. ノードの状態の木のキーは `"optimizer"`（`Layer.optimizer` / `Output.optimizer` を属性に置く）。`Input` には無い
3. 例外メッセージは `"SGD.tau"`（regex の match）と `f"AdamW.{field} must"`（substring）で検証する。計画書の頭「`AdamW.<name> must be finite and > 0`」「`AdamW.eps must be finite and >= 0`」に合わせた
4. `SGD.step_` の丸めは `p.sub_(g, alpha=elapsed / tau)` と同じ（B=4 のビット一致のテストは `p0.sub(g_sum, alpha=elapsed / tau / 4)` と `torch.equal` で比べる）
5. `grads` の数が違うときは `ValueError`（型だけを見て、メッセージは見ない）
6. `load_state_dict` は m・v を既存のテンソルへコピーする（渡した木を後で書き換えても、束縛は変わらない）
7. `pc.AdamW(tau=..., tau_m=..., tau_v=...)` は eps・weight_decay を既定値（1e-8、1e-2）にして作れる（TestFreeThreaded が使う）
