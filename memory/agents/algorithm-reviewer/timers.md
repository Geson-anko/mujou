# timers（実行ごとのタイマー・Timestamp）の監査

対象: worktree `.claude/worktrees/timers`、ブランチ `feature/2026-10-16/timers`（`edaf433`）、`feature/2026-10-16/cuda-tests..HEAD` の差分。基準は skill `locality-and-async`、`docs/execution-model.md`（グローバルタイマー・停止・復元・複数のネットワーク）、判断ログ `memory/agents/plan-implementer/timers.md`。`slot::tests::readers_see_whole_messages_in_order` の間欠的な失敗は対象外。

## verdict

**request-changes（文書と決定記録だけ）**（確信度: 高）。Rust コアの実装に誤りは見つからなかった。1 つのランタイムの中では、Δt・停止・復元・再開がすべて `Shared::timer` の 1 本の時間軸で行われる。別のタイマーのランタイムは、互いの時間を止めず、時刻も書き換えない。復元の排他もタイマーごとに正しく数えている。直すべきなのは次の 2 点である。

- 決定の記録: 「グローバルタイマーはシングルトン」という記録済みの決定と skill の不変条件を覆したのに、`memory/` と skill を更新していない（must-fix 1）
- 契約の明文化: 「1 つの実行のスロットはランタイムと同じタイマーで作る」という契約が、型でも文書でも示されていない。今の Python 層は全部グローバルタイマーを使うので壊れていない。しかし予定している `mujou.run(..., timer=...)` は、`build()` の時点でスロットを作る今の構造とぶつかる（should-fix 1）

観点ごとの結論:

- **1 つの実行の中で同じタイマーか**: ランタイムの中では満たす。`node_loop` / `boundary` / `wait_until_ns_or` / `interrupt_waiters` / `freeze.hold` / 起動時の `now_ns` / `last_update_ns` は、すべて `self.timer` を使う。`time::global()` の参照は bindings の既定値（`timer_or_global`）と `GlobalTimer` の生成にしか残っていない。ただしスロット（`Outlet` / `InputGroup`）のタイマーは別の引数で、ランタイムのタイマーと一致することを誰も確かめていない（should-fix 1）
- **別のタイマーの実行が互いを止めない・時刻を書き換えない**: 満たす。`PauseHold` が `Arc<Timer>` を持ち、凍結は自分のタイマーにだけ掛かる。`load_state` を呼ぶのは `freeze_for_startup` の自分のタイマーに対してだけ。Rust の `separate_timers` 4 本と pytest の `TestOwnTimer` 3 本が、Δt・capture・例外・復元のそれぞれで確かめている
- **停止中の時刻 ≤ T**: ランタイムの側（最後の更新の時刻）はタイマーごとに保たれる。`boundary` の「word → time → word」の論証は、制御側の凍結と読み取りが同じタイマーの lock で順序付けられることだけに依存する。各ランタイムは 1 本のタイマーだけを使うので、論証はそのまま成り立つ。スロットの時刻の側は、スロットがランタイムと同じタイマーにあるときだけ成り立つ（should-fix 1 で実際に破れることを確認した）
- **復元の排他**: 正しい。`LIVE` のキーは `Arc::as_ptr`。数えている間は `Live(Arc<Timer>)` が割り当てを生かしているので、別のタイマーが同じアドレスを再利用することはない。確認・凍結・`load_state`・計上は `LIVE` の lock の中で行うので、同じタイマーで 2 つの復元が同時に通ることはない。別のタイマーでは通る（テストあり）
- **docs の記述**: 「複数のネットワーク」「復元」の段落は新しい意味に直っている。しかし停止・再開・例外・状態の木の段落には「グローバルタイマーを凍結する」などの記述が残っている。`mujou.time.sleep` が延びるという記述も、グローバルタイマーで動くネットワークにしか当てはまらない（should-fix 2）
- **学習則の焼き付け**: 無い。`Timer` / `Timestamp` / `Message::timestamp` は時間の量だけを扱い、PC-ALM の量（λ、$W^\top\varepsilon$）や教師の概念は入っていない。`NodeHooks` の契約も変わっていない

## must-fix

1. **「グローバルタイマーはシングルトン」という決定を覆したのに、`memory/` と skill を更新していない**（確信度: 高）
    - `memory/project_async_node_execution.md` 15 行目には「グローバルタイマーはシングルトンにする。システム全体で時間は 1 つだけ。テストでは、外から別のタイマーを渡すのではなく、シングルトンの時刻の源を仮想時計に切り替える」とある。27・29 行目も同じ前提で書かれている。`memory/project_native_core_rust.md` 17 行目にも「グローバルタイマー（シングルトン）」とある
    - skill `locality-and-async` 34 行目の「グローバルタイマーはシングルトン（システム全体で時間は 1 つ）」と 112 行目のテスト規約も、今の実装と食い違う。CLAUDE.md の不変の原則の「グローバルタイマーごと止める」も、「その実行のタイマーごと止める」と読み替える必要がある
    - `memory/project_open_decisions.md` 16 行目と `docs/open-decisions.md` 41 行目には、未決として「1 つのネットワークを停止するとプロセスのグローバル時間が止まり、同時に走る他のネットワークも止まる（この意味でよいかユーザー確認待ち）」が残っている。この変更はその問いへの答え（別のタイマーで動かせば止まらない）なので、そこを更新する
    - CLAUDE.md は「破る必要があるときはユーザーに確認し、理由を `memory/` に残す」と定めている。要求がユーザー由来であれば、決定として 1 行足せば済む。そうでなければ、メインエージェントがユーザーに確認する。`.claude/skills/` の編集は skill `edit-dot-claude` の手順で行う
    - 新しい不変条件の書き方の案: 「時間軸は実行ごとに 1 本。1 つの実行の全ノード・全スロット・ホストの入力はその実行のタイマーを使う。既定は全実行が共有するグローバルタイマー」

## should-fix

1. **スロットのタイマーとランタイムのタイマーが一致する保証が無い。予定している `run(timer=...)` は、ビルド時にスロットを作る構造とぶつかる**（確信度: 高）
    - コアの IF では、`Outlet::new(.., timer)`、`InputGroup::new(timer)`、`Runtime::start(.., timer, ..)` がそれぞれ別にタイマーを受け取る。ランタイムはスロットを知らないので、一致を確かめる場所が無い
    - 実際に破れることを確かめた。ランタイムをグローバルタイマー（仮想時計）で動かし、`Outlet` を別の `Timer()` で作った。capture の `build` の中で別のタイマーを 1 秒進めて `send` すると、凍結時刻が $T$ = 917667 ns のところに、スロットの時刻 1000000000 ns が入った。状態の木の「全スロットの時刻 ≤ T」が崩れ、メッセージの時刻とノードの Δt が別の時間軸に乗る。今の `src/mujou/_session.py` はすべて既定値（グローバル）を使うので、公開 API からはこの状態にならない
    - ぶつかる構造: `Session.__init__`（= `Network.build()`）は、`InputGroup()` と `Outlet(...)` をビルドの時点で作る。`write` と `load_state_dict` は `run()` より前にそのスロットへ書く。`mujou.run(*networks, timer=...)` で実行のときにタイマーを渡す形にすると、ビルド済みのネットワークのスロットは既にグローバルタイマーで時刻を付けている
    - 案（ホスト側の作業へ申し送り）: タイマーはビルドの時点で決める（`build(timer=...)`、またはビルドしていなければ `mujou.run` がビルドする）。`Session` は `Arc<Timer>` を 1 つだけ持ち、`InputGroup` / `Outlet` / `Runtime` に同じものを渡す。ビルド済みのネットワークに別のタイマーを渡したら `ValueError` にする。あわせて、グローバルタイマーを直接読んでいる次の箇所もそのタイマーに替える
        - `_session.py` の `_tree` の `"timer": _core.timer.state_dict()`
        - `_capture_on_exception` の `_core.timer.time_ns()`
        - ノードが時刻を読む手段（`docs/nodes.md` 73 行目の「グローバルタイマーから読める」）
        - ホストが提示時間を待つ手段（`docs/network.md` 150 行目の `mujou.time.sleep`）
    - 申し送りの 2 点目: 復元の排他はタイマーごとに数える。そのため `mujou.run(a, b, timer=t)` で、復元した 2 つのネットワークを別々のランタイムとして順に起動すると、2 つ目の起動で `ValueError` になる。1 つのランタイムにまとめて起動するか、時刻を戻すのを 1 回にする
    - このタスクで足すとよいもの: `Outlet::new` / `InputGroup::new` / `Runtime::start` の doc comment（とスタブの docstring）、`docs/execution-model.md` の実装節に、次の契約を 1 文で書く。「1 つの実行のスロットとホストの入力は、そのランタイムと同じタイマーで作る。違うと、停止中の時刻 ≤ T の保証とメッセージの時刻の意味が崩れる」

2. **`docs/execution-model.md` の停止・再開・例外・状態の木の記述が「グローバルタイマー」のまま**（確信度: 高、重大度: 中）
    - 91 行目（停止の 1）、102 行目（再開）、134 行目（状態の木の表）、146・148 行目（復元）、199 行目（例外で凍結）、233 行目（例外の後で保持を解く理由に `mujou.time.sleep` を挙げている）。新しい意味では、どれも「その実行のタイマー」になる
    - 106 行目の「ホストの `mujou.time.sleep` も、止まっていた実時間だけ延びる」は、グローバルタイマーで動くネットワークにしか当てはまらない。別のタイマーで動くネットワークが capture しても、`mujou.time.sleep`（グローバルタイマー）は延びない。ホストがそのネットワークへの提示時間をグローバル時間で測ると、停止の分だけ提示が短くなる。このネットワークの時間で待つにはそのタイマーの `sleep` を使う、と書く。公開 API はまだ無いので、ホスト側の作業で書く、と注記するだけでもよい
    - 98 行目の「全スロットの値の時刻も T 以下になる」には、should-fix 1 の契約（スロットが同じタイマーにあること）が前提として必要

## nit

1. **別のタイマーのランタイムが capture 中も動き続けることを、ランタイムの単位では確かめていない**（確信度: 中、優先度: 低）。`a_capture_freezes_only_its_own_timer` と `test_a_capture_does_not_pause_another_timer` は、もう一方のタイマーの `is_paused()` だけを見ている。案: 手動モードで A（`ta`）と B（`tb`）を走らせる。A の capture の `build` の中で `tb.advance_ns(H)` と `b.step(None).wait()` を呼び、B のノードが Δt = H で更新されたことを確かめる。build は A の制御スレッドで走り、`b.step` は B の制御スレッドを待つだけなので、デッドロックはしない
2. **`Timestamp` はどのタイマーの時刻かを持たない**（確信度: 高、優先度: 低）。別のタイマーの時刻同士を比べても、黙って意味の無い結果が返る。docstring に「1 つのタイマーの時刻だけを比べる」とあり、should-fix 1 の契約が守られれば、1 つの実行の中では起きない。今のままでよい
3. pytest の `test_a_new_timer_starts_at_zero_and_runs` の上限 `< 1_000_000_000` は緩い。Rust 側の `a_new_timer_starts_at_zero` は「タイマーの時刻 ≤ 作る前から測った実経過」で厳密に確かめているので、実害は無い

## 検証結果

- `just run`: 成功（exit 0）。pre-commit は全 hook が Passed（`just stub` を含み、生成物は最新）。`cargo test --workspace` は全件 ok（lib 28、runtime 12、time 35 ほか）。pytest は 335 passed / 11 skipped。pyright は 0 errors。実行の前後で作業ツリーに差分は出ていない
- 不一致の再現スクリプト（スクラッチパッドの `mismatch.py`）の出力は `T(global) = 917667 1000000000 <- slot time`。ランタイムが凍結している間に、別のタイマーのスロットに $T$ より後の時刻が入る（should-fix 1）

## 照合メモ

- `boundary` の ≤ T の論証: 制御側は control lock の中で word を store し、続けて `freeze.hold(&s.timer)` で凍結する（timer の lock）。ノードは word → `self.timer.time_ns()`（timer の lock）→ word の順に読む。凍結が時刻の読みより先なら、timer の lock を介して word の store がノードの 2 回目の load より先に順序付けられ、ノードはその時刻を捨てる。この論証に要るのは「凍結するタイマー = ノードが読むタイマー」だけで、`Shared::timer` が 1 本なので成り立つ。別のタイマーのランタイムは、別の lock・別の word を使うので互いに干渉しない
- `interrupt_waiters`: 通知は `self.timer` の condvar に送られ、待つのは同じランタイムのノードが `self.timer` で待つ `wait_until_ns_or` なので、取りこぼさない。同じタイマーの別のランタイムのノードも起きるが、自分の word が変わっていないのでまた待つ。これは以前のグローバルタイマーのときと同じ挙動
- lock の順序: `LIVE` → control → timer（`freeze_for_startup`）。`Live::drop` は `LIVE` だけを取り、`mark_finished` は control lock を離してから drop する。変更の前と同じで、循環は無い
- `LIVE` のアドレスの再利用: `Live` が `Arc<Timer>` を持ち、エントリは計数が正の間しか無い。計数が正の間は割り当てが生きているので、同じアドレスの別のタイマーは存在できない
- `PauseHold` は `Arc<Timer>` を持つので、hold がタイマーより長く生きることはない。`Timer` はランタイムを参照しないので、参照の循環も無い
- Python の `_core.timer` は declarative module の const で、モジュールの初期化時に 1 回だけ作られる。中身は `global()` と同じ `Arc` なので、`Runtime(timer=_core.timer)` と `Runtime()` は LIVE の同じエントリに数えられる
