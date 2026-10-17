# run-handle（実行の API の作り直し）レビュー

対象: `feature/2026-10-16/run-handle`、`git diff 942f5ca..HEAD`（9eedc29..4c73e0d）。

## verdict: request-changes

決定との一致はおおむね取れている。build は同じ Graph を返す。`Network.run` は無い。`mujou.run(*networks, timer, inputs, state, pause_timeout)` は `Run` を返し、with で使える。Compose のパスは `"0.l1"` / `"enc.l1"`。スロットとランタイムは run の時点でその実行のタイマーで作る。`Timestamp` を返す。`mujou.testing` は mujou 自身のテストの口でもあり、公開シグネチャに `_` の付いた引数は無い。

ただし、次の 2 点で docstring が約束する「1 回だけ走る」「断られた run はやり直せる」が崩れる。

- 複数のネットワークを渡した run が断られると、そのネットワークは二度と走れない
- 同じネットワークを別スレッドから同時に run すると 2 回走る

## must-fix

### M1. `mujou.run(a, b)` が断られると、`a` と `b` は二度と走れない

- 対象: `src/mujou/_core/run.py:313-314`、`src/mujou/_core/network.py:272-274`
- 問題: `start` は呼ぶたびに新しい `Compose(*networks)` を作って build する。build は `a` と `b` の Port を凍結し、Graph のキャッシュは捨てられる Compose の側に残る。`inputs` の shape 違いや `state` の不一致で `ValueError` / `TypeError` になったあと、同じ `mujou.run(a, b)` を呼び直すと `RuntimeError: the port '0.l.prev' belongs to a network that is already built` になる。`mujou.run(a)` 単独でも同じエラーで走れない。`mujou.testing.run` も同じ
- 根拠: `run` の docstring（run.py:282-286）は「Nothing started: the networks may run once the cause is gone.」と約束している。`test_a_refused_run_can_be_retried` が確かめているのは 1 つのネットワークの場合だけ
- 再現:
  ```python
  a, b = S(), S()
  mujou.testing.run(a, b, timer=VirtualTimer(), inputs={a.prev: torch.zeros(3, dtype=torch.float64)})  # ValueError
  mujou.testing.run(a, b, timer=VirtualTimer())  # RuntimeError: ... already built
  ```
- 確信度: 高（再現済み）／深刻度: 中〜高

### M2. 同じネットワークを 2 つのスレッドから同時に `mujou.run` すると、両方が走ることがある

- 対象: `src/mujou/_core/network.py:205-206, 274`（`build` の `_graph` の確認と代入）、`network.py:319-339`（`start_once`）
- 問題: `build` は lock を取らずに check-then-act する。2 つのスレッドがどちらも `_graph is None` を見ると、それぞれ別の Graph を作る。凍結の確認（`is_frozen`）も相手の `freeze` より先に走りうる。`start_once` の lock は Graph ごとなので、別々の Graph は止められない。2 つの Session が同じ Port を結び付け直し、2 つの Runtime が同じ Node を動かす。`thread_ident`・`_timer`・`_binding` は互いに上書きされる
- 根拠: docstring は「A network runs once.」（run.py:253）、`Graph._lock` のコメントは「makes the check and the start one step for concurrent callers」（network.py:94-95）と書いている。以前は `build()` を利用者が明示的に呼んでいたが、今は `mujou.run` の中で暗黙に呼ぶので、この経路が新しく開いた
- 再現: 30 ノードのネットワークに対して、Barrier で揃えた 2 スレッドから `mujou.testing.run(net, timer=VirtualTimer())` を呼ぶ。3.14t で 200 回中 6 回、両方の run が返った（scratchpad の `probe/p2.py`）
- 確信度: 高（再現済み）／深刻度: 中（並行に呼んだときだけ起きる。起きると「1 回だけ走る」が崩れ、状態が壊れる）

## should-fix

### S1. ノードを直接 `mujou.run` に渡すと `AttributeError` になる

- 対象: `run.py:313-314`
- 問題: `mujou.run(Hidden())` は `'Hidden' object has no attribute 'build'` で落ちる。2 つ以上渡すと Compose が `TypeError` を投げるので、渡した数で例外の型が変わる
- 根拠: Compose は `isinstance(network, Network)` を確かめて `TypeError` にしている（network.py:309-315）。1 つのときだけその確認を通らない
- 確信度: 高／深刻度: 低

### S2. 正常に終了した後の `ManualRun.step()` が投げる `RuntimeError` が docstring に無い

- 対象: `run.py:223-235`
- 問題: shutdown の後に `step()` を呼ぶと `RuntimeError: the runtime has finished` になる。docstring の Raises には `ValueError` と `NetworkError` しか書かれていない
- 確信度: 高／深刻度: 低

### S3. `mujou.run` の Raises が、木の一部が別の構築済みネットワークに属する場合を書いていない

- 対象: `run.py:279-287`
- 問題: Compose の中のネットワークを先に単独で run した場合や、`mujou.run(a, b)` を 2 回呼んだ場合に出るのは、`Network.build` の `RuntimeError: ... belongs to a network that is already built` である。docstring は「A network has already run」とだけ書いている。「As `Network.build`」の参照は `ValueError, TypeError` の項にしか無い
- 根拠: 動作は意図どおり。単独 → Compose、Compose → 単独のどちらも `RuntimeError` で止まることを確認した。足りないのは文書だけ
- 確信度: 中／深刻度: 低

### S4. `Timestamp` は pickle も deepcopy もできないので、受信した時刻を状態に持つノードで `state_dict` が `TypeError` になる

- 対象: `src/mujou/_core/port.py:227-238`（`recv_with_time` の戻り値が `int` から `Timestamp` に変わった）、`session.py:557`（`copy.deepcopy`）
- 問題: 受け取った時刻をノードの状態に残す書き方は自然だが、今はその `state_dict` が状態の取得の途中で `TypeError` になる。テストも `(... ).ns` で int に直して避けている（test_run.py:247-252）。以前は int だったので、ここは後退である
- 根拠: `pickle.dumps(VirtualTimer().now())` は `TypeError: cannot pickle 'mujou._core._native.Timestamp' object` になることを確かめた。計画書の確認事項 2 にも挙がっている。この PR で直すか、`recv_with_time` / `Node.state_dict` の docstring に「`Timestamp` は木に入れられない。`.ns` で入れる」と注記するかを決めてほしい
- 確信度: 高／深刻度: 中

### S5. 正常に終了した後の `state_dict` の中では `Node.timer` が `RuntimeError` になる

- 対象: `src/mujou/_core/node.py:154-172`、`session.py:435-447`（`_settle()` の後に `capture_state`）
- 問題: 実行中の `state_dict` の中では `self.timer` が読めるのに、正常に終了した後の `state_dict` の中では読めない。決定の主な流れは「作る → 走らせる → 保存する」で、保存の主な経路は終了の後である。終了の後の木のタイマーは停止を頼んだ時刻に凍結しているので、`timer` を返しても意味は保てる
- 根拠: 計画書には意図した挙動として書かれ、docstring にもある。ただし利用者が引っかかりやすいので、orchestrator かユーザーの判断に回したい
- 確信度: 中／深刻度: 低〜中

### S6. `timer` が Node の予約名になったのに、Node の docstring に書いていない

- 対象: `node.py:154`（property）、Node のクラス docstring
- 問題: サブクラスの `__init__` で `self.timer = ...` と書くと `AttributeError: property 'timer' of 'N' object has no setter` になる。`timer` は利用者が属性名にしやすい名前である。クラス docstring は `device` / `generator` には触れているが、`timer` には触れていない
- 確信度: 高／深刻度: 低

### S7. M1 と M2 の経路にテストが無い

- 対象: `tests/mujou/_core/test_run.py`（`TestState.test_a_refused_run_can_be_retried` は 1 つのネットワークだけ）、`tests/mujou/_core/test_network.py`（`TestCompose`）
- 問題: 次の 3 つのテストが無い
  - 断られた `mujou.run(a, b)` を呼び直せること
  - Compose の中のネットワークを単独で run したとき（前後どちらでも）に `RuntimeError` になること
  - 同じネットワークを 2 回並行に run しても 1 回しか走らないこと
- 確信度: 高／深刻度: 中

### S8. 並列ブランチ `feature/2026-10-16/node-state`（c4cb9ad）と merge すると衝突する

`git merge-tree` で試した（読み取りだけ）。

- テキストの衝突:
  - `src/mujou/_core/node.py`（クラス docstring の `load_state_dict` / `on_update(dt)` の行）
  - `specs/nodes.md` の L119 付近（「定義と実行」の段落と、node-state の「状態の自動収集」の節）
  - `specs/execution-model.md` の L217-222（`mujou.run` / `Run.state_dict` と `run()` / `Network.state_dict()`。後者は `on_thread_start` に改名している）
  - `tests/mujou/_core/test_node.py`（import）
- 意味の衝突:
  - node-state が足したテストは、このブランチで消した口を使っている。merge 後の test_node.py の L197 付近は `run_network(Solo(node), _manual=True)`、L370-397 は `steps(net, virtual_timer, n)` と `net.state_dict()` / `net.shutdown()` / `restored.load_state_dict` / `run_network(restored, _manual=True)` を使う。テキストとしては衝突せずに入るが、壊れる。`start_manual` / `virtual` / `Run` / `state=` で書き直す必要がある
  - `Node.timer` の docstring と `specs/nodes.md:73` の「on_start と、その後の hook で読める」は、公開される `on_thread_start` に触れていない。タイマーは `runtime.start` の前に付けるので、`on_thread_start` でも読める
  - `set_timer`（`node._timer = ...`）と `end_thread` は、node-state の `__setattr__` を通る。Tensor ではないので登録から外す（pop）だけで、害は無い
- 確信度: 高（衝突は merge-tree で確かめた。意味の衝突は merge 後の木を grep して確かめた）

## nit

- `node.py:29`: docstring の 1 行が 96 文字ある（`on_update(dt)` を編集したあと折り返していない）
- `specs/nodes.md:75` の「停止している間、グローバル時間は進まない」と、`specs/network.md:134,149-153` の「グローバル時間で提示」は、実行ごとのタイマーの話に揃っていない（「その実行のタイマー」）
- `mujou.run()` を引数なしで呼ぶと `ValueError: the network has no node` になり、原因が分かりにくい
- `Compose(**{"": net})` や `Compose(**{"a.b": net})` は拒否されず、曖昧なパスができる（`network.py:303`。弾くのは `_`・Network の属性名・位置だけ）
- `runtime.start` の最中に `BaseException`（KeyboardInterrupt）が起きると、後始末がどれも動かない。`session.py:258` の `except Exception` も `network.py:335` の `except NetworkError` も捕まえないので、Graph は走った印が付かず、`Node.timer` も付いたまま残る
- `Graph` の docstring の「which `mujou.run` runs」は、`mujou.run(graph)` を呼べるように読める。実際に受け付けるのは Network だけである（決定どおり）
- 同じ Node（Port を持たない）を 2 つのネットワークに入れると、どちらも build できて走る。凍結は Port でしか見ていないためで、以前からある。今回 `_timer` の上書きが 1 つ増えた

## 検証結果

- just run: pass
  - format: pass
  - cargo test: 28 + 14 + 37 passed
  - pytest: 405 passed, 14 skipped
  - pyright strict: 0 errors
  - mkdocs build --strict: pass
- 繰り返し: `tests/mujou` を free-threaded（`sys._is_gil_enabled()` が False）で 5 回、`python -X gil=1` で 5 回回し、どれも 384 passed, 13 skipped
- 成果物の汚染（`</content>` など）: 無し
- 消したテストを確かめた。どれも新しい口のテストに移ったか、新しい API では表せなくなったものである（`test_before_run_gives_the_initial_value` → `TestInputs.test_are_the_initial_values`、`test_a_broken_tree_changes_nothing` → `TestState.test_a_refused_run_can_be_retried`、`test_after_load_state_dict_takes_precedence` → `TestInputs.test_after_a_state_take_precedence_and_carry_its_time`、`test_needs_a_build` / `test_graph_before_build` / `test_rejects_a_second_build` は API が無くなった）。足りないのは S7 だけ
- テストの決定性: manual のテストはどれも `VirtualTimer` と `step` で決定的に回る。`test_a_pause_stops_only_the_timer_of_its_run` と `test_sleeps_on_its_own_time` は実時間を使うが、assert は不等式だけで余裕がある
