# M:N スケジューリング PR 2（Python のノードをワーカーに載せる）の判断ログ

計画書: `memory/agents/implementation-planner/mn-scheduler.md` の PR 2 と「ユーザーの決定（2026-10-18）」。仕様テストは spec-test-author（`memory/agents/spec-test-author/mn-python-nodes.md`、b7f30b9）。

## 実装したもの

- bindings（`crates/mujou-python/src/runtime.rs`）: `Runtime.start(workers=, dedicated=)`（`dedicated` 省略で全ノードがワーカー、`workers=0` は `ValueError`）。`PyNode` に `context`（ノードごとの `contextvars.Context`）と `dedicated`。`enter` で `_on_enter` を呼び、エラーは次の公開 hook の名前で報告。`start` は `dedicated` のときだけ `on_thread_start`。`public()` から `_on_thread_start` を消した
- `node.py`: `_on_thread_start` → `_on_enter`。`dedicated_thread` 引数・property・`checked_dedicated`。`thread_ident` / `end_thread` / `_thread_ident` を削除
- `hooks.py`: `enter_node(node)` / `current_node()`。`binding.py` は `current_node() is self._owner` で所有を判定
- `session.py` / `run.py`: `workers` と `dedicated` を native へ。`mujou.run(workers=None)` の検査
- docs（en/ja）、examples のコメント、memory、skill `locality-and-async`。specs はサブエージェントに委譲して確認した

## 計画にない判断（理由つき）

1. **ノードの `contextvars.Context` は空の `Context()` にした**（ホストの文脈の `copy_context()` ではない）。今まで Rust が立てたスレッドの文脈も空だったので、振る舞いを変えない。ホストの `decimal` の文脈などを持ち込まない
2. **`state_dict`（`_capture_state`）は bindings で文脈の外から直接呼ぶ**。計画どおり。`call()` は必ず `Context.run` を通すので、`state_dict` だけ別の経路にした
3. **`_on_enter` も `Methods` で 1 回だけ引く**。切り替えのたびに `getattr` しないため。`start` が終わるまでは名前で引く（ほかの hook と同じ）
4. **ノード専用の CUDA ストリームは最初の `_on_enter` で作る**（`self._stream is None` のとき）。ノードは 1 回しか走らないので、実行ごとに作り直す必要は無い
5. **`super().__init__` を呼ばないサブクラス向けの generator の生成は `_on_enter` に残した**（今まで `_on_thread_start` にあった）。`is None` の判定 1 回で済む
6. **`checked_dedicated` が予約名の検査と `on_thread_start` の override の検査を兼ねる**。計画は別の関数も挙げていたが、どちらも「そのノードが専用スレッドか」を決めるときの検査なので 1 つにした。override はクラスで判定する（`inspect.getattr_static(type(node), ...)`）。インスタンス属性での置き換えは見ない（計画も「クラスが override していたら」）
7. **`workers` の検査は `start_run` の `in_hook` の検査の直後、構築の前**。何も起動せず、木も作らない。`mujou.testing.run` は `workers=None` を渡す（`start_run` の引数は必須にした）
8. **`current_node` の束縛は外さない**。ワーカーも専用スレッドも実行の終わりに抜け、スレッドの Python の状態ごと消える。ホストと制御スレッドは `enter_node` を呼ばないので、`current_node()` は常に `None`
9. **`docs/*/index.md` の「自分のスレッドで更新」は「自分の周期で更新」に変えた**。ja の index はユーザーが直した文体の基準なので、語を 1 つ替えるだけにした
10. GIL ありの 3.14（Homebrew）でも `tests/mujou` と `tests/test_examples.py` が通ることを手元で確かめた（551 passed）

## 確認事項

なし（ユーザーの決定の範囲で実装した）。

## レビュー対応（2026-10-18）

記録: `memory/agents/code-reviewer/mn-python-nodes.md`（request-changes、must-fix 1）、`memory/agents/algorithm-reviewer/mn-python-nodes.md`（approve、should-fix 3）。orchestrator の裁定に従った。

### 対応したもの

1. code M1・algo N1: `_on_enter` の先頭で `enter_node(self)` を呼ぶようにした。CUDA や generator の設定で失敗しても、コアが続けて呼ぶ `on_exception` / `on_stop` は自分として走る。テスト `TestThreadSetup.test_a_node_whose_setup_failed_cleans_up_as_itself`（generator の作成を失敗させ、別のノードの Port を読むと `RuntimeError`、勾配の記録は無効）。直す前の順序に戻すと落ちることを確かめた
   - 裁定は「先頭で勾配の記録も切る」だったが、2 の `_before_hook` が `on_exception` / `on_stop` を含む全公開 hook の前に毎回切るので、`_on_enter` からは消した（二重になるだけ。`enter` が失敗しても bindings は hook の前に `_before_hook` を呼ぶ）
2. S1（両者）: 勾配の記録を各 hook の前に毎回切る。bindings に内部の hook `_before_hook` を足し、`state_dict` 以外の公開の hook（`on_thread_start`・`on_start`・`load_state_dict`・`on_update`・`on_pause`・`on_resume`・`on_exception`・`on_stop`）の直前に毎回呼ぶ（`Context.run` の外、同じ attach の中）。エラーはその hook の名前で報告する。`mujou.Node._before_hook` が `torch.set_grad_enabled(False)`。bindings は torch を知らないので、Python 側の内部の hook にした。テスト `test_gradients_are_off_at_the_start_of_every_hook_of_the_same_node`（workers=1、ノード 1 つ。`_before_hook` を空にすると落ちる）。docstring・specs（torch.md・native-core.md・nodes.md）・skill・memory を「勾配は毎回、CUDA の device と stream は切り替えのたび」に揃えた
3. algo S2・code S3: スレッドごとの設定は hook を抜ける前に戻す（戻さないと同じワーカーの次のノードに漏れる。フレームワークが戻すのは勾配と CUDA の device・stream だけ）を、Node の docstring、docs（en/ja の writing-a-node・torch）、skill の局所性の不変条件に書いた。「次の hook に持ち越されない」の言い過ぎを「スケジュール次第」に直した。autocast をフレームワークで切るかはユーザーに確認中なので、「フレームワークが戻すのは〜だけ」と今の実装の事実だけを書いた
4. code S2: `Run` の docstring、`Port.recv` / `send` の Raises、`slot.rs` の `Outlet` の doc から「ノードのスレッド」を消した
5. algo S3: skill の既定のワーカー数を `min(ワーカーに載るノード数, 論理 CPU 数)` に、`_on_enter` / `_before_hook` は「フレームワーク側の設定を書く内部の hook（利用者は override しない）」と読めるようにした
6. nit
   - code N1: docs（running.md en/ja）と `mujou.run` の docstring に、既定はプロセスが使える CPU の数（affinity や割り当ての上限で論理コア数より少ないことがある）と書いた
   - code N3: `StreamRecorder` の docstring を直した（替えるのはストリームだけ）
   - code N4・algo N3: docs の torch.md（en/ja）と specs/torch.md に、CPU のノードは CUDA の device と stream を設定しないので直前の CUDA のノードのストリームが残る、CPU のノードは CUDA を使わない、と書いた
   - code N5: `workers > sys.maxsize` も `ValueError` にした（コアは `usize` で数える）。`TestWorkers` に `2**70` の場合を足した
   - code N6: ja の 3 か所（running.md の体言止め、torch.md の tip の「の和」、writing-a-node.md の読点）
   - code N7・algo N4: docs の writing-a-node（en/ja）に、ノードの `Context` は空から始まりホストの文脈を引き継がないこと、`state_dict` はその文脈の外で呼ばれることを書いた。Node の docstring にも
   - algo N2: docs の torch.md（en/ja）と Node の docstring に、CUDA のノードは `_after_update` の同期の間ワーカーを占有すると書いた

### 対応しなかったもの

- code N2（既定のワーカー数のテストが `os.cpu_count()` で近似している）: bindings にも `Run` にも、実際のワーカー数を読む口が無い。テストのためだけに公開の口を足すのは要求外なので直さない。`PYTHON_CPU_COUNT` などで `os.cpu_count()` を小さくした環境でしか落ちない
- 原文書にあった `Runtime` の docstring「The hooks other than ... are looked up once」は不正確だった（`_on_enter` も start の後は 1 回だけ引く）ので、`_before_hook` を足すついでに「`on_start`（と `load_state_dict`）が戻るまでは名前で、その後は 1 回だけ引く」に直した
