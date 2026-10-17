# run-handle（実行の API の作り直し）の監査

対象: worktree `.claude/worktrees/run-handle`、ブランチ `feature/2026-10-16/run-handle`。`git diff 942f5ca..HEAD`（`9eedc29` / `986af93` / `4c73e0d`）。基準は skill `locality-and-async`、`specs/execution-model.md`、`specs/network.md`。

## verdict: request-changes

実行の骨格は不変条件を守っている。1 つの実行が 1 本のタイマーを使うこと、停止中の時刻が T 以下になること、復元の順序、`Compose` が非同期の境界を変えないこと、手動ステップが自由実行と同じループを使うこと、の 5 点は確かめた。学習則の焼き付けも無い。

ただし、新しく足した `Node.timer` が、ホストと同じ `Timer` オブジェクトをノードに渡している。そのため、ノードが実行の時間軸を書き換えられる。他のノードの Δt を変える、テストの時計を進める、停止や手動ステップを止める、の 3 つを probe で再現した。must-fix 1 件。

## must-fix

1. **`Node.timer` がノードに実行のタイマーの制御を渡している** — `src/mujou/_core/node.py:155`（`timer` property）、`src/mujou/_core/session.py:247`（`set_timer(node, self._timer)` で利用者の `Timer` をそのまま渡す）、`src/mujou/_core/timer.py:43,56,91`（`sleep` / `set_time_scale` / `advance`）
   - 問題: ノードが受け取るのは、ホストが `timer=` に渡したオブジェクトそのもので、テスト `tests/mujou/_core/test_node.py:125` も `timer is virtual` で同一性を固定している。ノードの hook から次の 3 つを呼べる
     - `self.timer.set_time_scale(s)`: 同じ実行の全ノードの Δt が変わる。ノード A がスケールに値を載せれば、ノード B は自分の Δt からそれを読めるので、スロットを通らない通信路になる（局所性）。タイマーの書き手はホストのはずが、ノードも書き手になる（状態の所有）
     - `VirtualTimer.advance(dt)`（`self.timer` が `VirtualTimer` のとき）: 手動ステップの途中で時計が進み、後のノードの Δt が `timer.advance(H)` の値からずれる。`mujou.testing` が約束する「Δt はテストが進めた分そのもの」が崩れ、離散アルゴリズムのオラクルが成り立たなくなる
     - `self.timer.sleep(x)` を `on_update` の中で呼ぶ: 停止はタイマーを先に凍結するので sleep が戻らず、`state_dict` が `PauseTimeoutError` になる。手動実行の `VirtualTimer` では、ホストが `step()` の中で待っているため誰も時計を進めず、`step()` が永久に戻らない（`Runtime::step` に時間制限は無い）
   - 再現（worktree の venv、scratchpad の probe）:
     - `set_time_scale(1000)` をノードから呼ぶと、実行の後もタイマーのスケールは 1000 のまま。隣のノードの Δt の最大値は 0.18 s だった（実時間の 1 周は ms 程度）
     - `mujou.testing.run` + `VirtualTimer` で、ノード a が `advance(5.0)` を呼んでから `timer.advance(0.001); run.step()` すると、ノード w の Δt は `[5.001]` になった
     - `self.timer.sleep(0.5)` を `on_update` で呼ぶノードに `state_dict(pause_timeout=2.0)` を呼ぶと、2.0 s で `PauseTimeoutError` になった。手動 + `VirtualTimer` で `sleep(0.001)` を呼ぶと、`step()` は 3 s たっても戻らなかった
   - 範囲: グローバルタイマーについては、`mujou.time.set_time_scale` / `sleep` を hook から呼べる穴が以前からある（今も hook からの `mujou.time.set_time_scale(1.0)` は例外にならないことを確認した）。この差分で新しく増えたのは 2 点。実行ごとのタイマーにも同じ穴が開いたこと、そして `Node.timer` の docstring と `specs/nodes.md:73` が、この口をノードに使わせる正規の経路として案内していることである
   - 直し方の案（小さい順）:
     1. ノードには読み取り専用の view（`now()` と、必要なら `get_time_scale()` だけ）を渡す。`Node.timer` の型を `Timer` 以外にする
     2. または `Timer.sleep` / `set_time_scale` / `VirtualTimer.advance` が `in_hook()` で `RuntimeError` を出すようにする（`Run` のホスト用メソッドと同じ扱い）。同じ検査を `mujou.time.sleep` / `set_time_scale` にも入れると、以前からある穴も塞がる
     - どちらにしても `test_node.py:125` の `timer is virtual` は書き換える。公開型が `Timer` のまま出荷すると、後から狭めるのは破壊的変更になる。そのため、merge の前に決めることを勧める
   - 確信度: 高（振る舞いは再現済み）。深刻度の判断（must-fix とすること）は中〜高。手動ステップのオラクルと局所性という mujou の存在理由に直接かかわり、しかも新しい公開 API の形の問題なので must-fix にした

## should-fix

1. **`Compose` で束ねても結果が変わらないことのオラクルテストが無い** — `tests/mujou/_core/test_network.py:268-318`
   - 問題: `TestCompose` はパスと検査しか見ていない。束ねたネットワーク同士が 1 本の時間軸で動くこと、ネットワークの境界が非同期の境界を変えないことを、数値で確かめるテストが無い。skill `locality-and-async` の「分け方を変えても同じ結果になる」の、ネットワーク版にあたる
   - 実装は正しい（probe で確認）。`Leaky` 1 個のネットワーク 2 つを connect して `Compose` したものを、`Chain` と同じ順序（Gauss-Seidel / Jacobi）で 20 ステップ回した。両者の `a.x` / `b.x` は `torch.equal` で一致し、木の `timer` と `last_update_ns` も一致した
   - 案: `TestEndToEnd.test_manual_steps_match_the_discrete_algorithm` を `Compose(a, b)` 版でもパラメタライズする。あわせて `state_dict()["timer"]` が 1 つであることも確かめる
   - 確信度: 高

2. **ポートを持たないノードは 2 つの実行に同時に入れる** — `src/mujou/_core/network.py:258-262`（構築済みの判定はポートの `is_frozen` だけ）
   - 問題: `a = Solo(Lone())` を `mujou.run(a)` で走らせたまま `mujou.run(Compose(a))` も走らせると、どちらも通る。1 つのノードオブジェクトが 2 つのランタイムの 2 本のスレッドで同時に `on_update` され、`_timer` と `_thread_ident` は後の実行の値で上書きされる。状態の書き手が 1 人であるという不変条件（状態の所有）に反する。`Network.build` の docstring は「Part of the tree belongs to another built network」で `RuntimeError` にすると書いているが、ポートの無いノードはこの判定をすり抜ける
   - 再現: probe で、`threads == 2` を確認した
   - 範囲: 判定は以前からポート頼みなので、既存の穴である。ただ、`Compose` と `mujou.run(a, b)` によって「既にあるネットワークを束ね直す」書き方が自然になり、踏みやすくなった
   - 案: `build` でノード（とネットワーク）にも構築済みの印を付け、ポートと同じように検査する
   - 確信度: 高（再現済み）。深刻度: 中（ポートの無いノードは実用上まれ）

## nit

1. **`mujou.testing.run` の `timer` の既定がグローバルタイマー（実時計）** — `src/mujou/testing.py`、`specs/execution-model.md:247`。テスト用の口の既定で Δt が実時間に依存し、決定的にならない。`mujou.run` と意味をそろえる判断（確認事項 4）は理解できる。それでも、`VirtualTimer` を渡し忘れたテストが黙ってフレーキーになる。docstring の先頭で強く促すか、既定を `VirtualTimer()` にするかを、もう一度検討してほしい。確信度: 中（設計判断）
2. **`Timestamp` はどのタイマーの時刻かを持たない**。別のタイマーで走る実行の `read` の時刻と `mujou.time.time_ns()` を比べても、型の上では何も言われない。今は文書で足りるが、`Run.read` の docstring に「`mujou.time` の時刻とは比べられない（その実行の `timer` のときだけ同じ軸）」と一言あるとよい。あわせて、`Node.timer.now()` を案内すると、ノードが `Timestamp` を自分の状態に入れて、`state_dict` が pickle / deepcopy で落ちる経路も増える（確認事項 2）。確信度: 中
3. **`Node.timer` は正常に終了した後の `state_dict` の中で `RuntimeError` になる**。走っている間の `state_dict` では読めるので非対称になる。仕様に明記されているので nit にとどめる。must-fix 1 を読み取り専用の view で直すなら、終了後も停止時刻 $T_\text{req}$ を返す view にできる。確信度: 低

## 確かめたこと（問題なし）

- **1 つの実行が 1 本のタイマーを使う**: `Session.__init__`（`session.py:150-153,182,240`）が、native の `Arc<Timer>` を 1 つ（`self._clock`）だけ持ち、`InputGroup`・全 `Outlet`・`Runtime` に `timer=self._clock` で渡す。停止の時刻の記録（`:288`）と、走っている間の木のタイマー（`:424`）も同じ `self._clock` から読む。`session.py` に `_native.timer` を直接読む箇所は残っていない（以前は `:268` と `:411` で読んでいた）。`GLOBAL` は `PyTimer(Arc::clone(core::global()))` を包んでいるので、`timer=None` でも `mujou.time` と同じ軸になる。スロットは `mujou.run` の中（`start_once` の lock の下）で作るので、利用者の操作で食い違うことは無い
- **停止中の時刻 ≤ T**: ランタイムの区切りの論理（`runtime.rs` の `boundary` / `capture` は先にフラグを立て、続けて凍結する）は変わっていない。スロットもホストの入力も同じタイマーで時刻を付けるので、前提が満たされる。正常に終了した後の木は、`_stop_timer` と全ノードの最後の更新・全スロットの時刻の max を取る（`_ended_tree`）
- **復元の順序**: `Session.start` は `_restore`（スロットをその場で戻す）→ `inputs` を `restored["timer_state"]["time_ns"]`（木の時刻 T）で `slot.restore` → `Runtime.start(timer_state=...)`（凍結してから T を読み込み、ノードを起動する）の順に進む。`test_after_a_state_take_precedence_and_carry_its_time` が T の付与を、`test_cannot_set_back_a_timer_another_run_uses` が復元の排他を確かめている
- **`Compose`**: `Compose` も `Network` で、`build` が全ノードを平らな `Graph.nodes` にする。1 つの `Session`・1 つの `Runtime`・1 つの `InputGroup` で動き、ネットワークごとの同期点は無い。各ノードは自分のスレッドのままで、ネットワークの境界は非同期の境界に影響しない（数値は should-fix 1 の probe で一致を確認）
- **手動ステップ**: `ManualRun.step` → `Runtime::step` は `grant` を 1 ノードずつ与える。ノードは自由実行と同じ `node_loop` / `boundary` を回し、違いは区切りで許可を待つことだけである。同期専用の別経路は無い
- **オラクル**: `TestEndToEnd` の Gauss-Seidel / Jacobi の `torch.equal` 一致、走っている間に取った木からの継続、終了後に取った木からの継続（`virtual.advance(100.0)` の後でも同じ）は、新しい口でも保たれている
- **学習則の焼き付け**: `src/` の差分に、PC-ALM などの更新手法の量は無い（`teacher` は docstring の例だけ）。ランタイムの契約も `NodeHooks` だけのまま
- **ノード間の autograd**: 差分に `.backward()` / `autograd.grad` は無い

## 検証結果

- `just run`: 成功。pytest 405 passed / 14 skipped（CUDA）、pyright 0 errors、mkdocs `--strict` 成功。実行後も作業ツリーは clean
- probe（scratchpad の `rh/probe.py`・`probe2.py`・`probe3.py`）: must-fix 1 の 3 つの再現、should-fix 2 の再現、`Compose` と `Chain` の bit 一致

## 照合メモ

- 停止中の時刻 ≤ T の前提は「スロット・入力・ランタイムが同じ `Arc<Timer>` を使う」こと。`Session` がこれを構造で保証する（タイマーを受け取る口は `Session.__init__` だけで、`native(timer)` を 1 回呼ぶ）
- 復元した入力の時刻 T と、`Runtime.start` が読み込むタイマーの時刻は、同じ `restored["timer_state"]` から取る。終了後の木では T = max(T_req, 最後の更新, スロット) なので、入力の時刻 T は全スロットの時刻以上で、時刻の順序は崩れない
- 手動実行の Δt は `(now_ns - last_ns)/1e9`。外から `advance(H_NS)` したときだけ H と一致する。ノードが同じタイマーを進められると、この等式がノードの側から崩れる（must-fix 1）
