# M:N スケジューリング PR 1（コアのワーカープール）の監査

対象: worktree `/Users/geson/Projects/mujou-mn`、ブランチ `feature/2026-10-18/mn-scheduler`（`8d7b414`）、`git diff origin/main..HEAD`。基準: skill `locality-and-async`、`specs/execution-model.md`、計画書 `memory/agents/implementation-planner/mn-scheduler.md`、実装者の記録 `memory/agents/plan-implementer/mn-scheduler.md`。

## verdict

**approve**（確信度: 高）。must-fix は無い。依頼の 5 観点はどれもコードの上で成り立つ。should-fix は 4 件で、どれも不変条件を破るものではない。内訳は、M:N にしたことで仕様の約束が弱まる箇所（例外・再開・タイムアウト）の明文化、timekeeper の過剰な起こし（M:N の目的に反するスレッドの起こし方）、`enter` が失敗したときに `on_stop` が呼ばれなくなる点、T 以下の保証を直接確かめるテストの不足。

- ノードは互いを待たない: 満たす。ノードの更新を取れるかどうかは、そのノードの期限とタイマーと制御の状態だけで決まる（`pool.rs:184-195`、`:286-351`）。ほかのノードの更新の終わりを条件にしていない。待つのは空いたワーカーだけで、N ≥ M なら更新中のノードがほかのノードの開始を遅らせることはない（timekeeper は仕事を取ると役を譲る。`pool.rs:362-365`）。ワーカーが 1 つのノードを待つ箇所も無い（`advance` は `Park` を返すだけ）
- 停止中は時間が進まず、時刻は T 以下: 満たす。プールの更新は、制御の lock の下で `c.command == Run` を確かめ（`pool.rs:185`）、同じ lock の下で時刻を読む（`pool.rs:144`）。`capture` は同じ lock の下で `set_command(Pause)` と `freeze.hold` を行う（`control.rs:110-112`）。凍結より後に始まる `on_update` は無く、使う時刻は T 以下になる。凍結は Timer の性質なので、ワーカーかどうかに依らない。専用スレッドの「語 → 時刻 → 語」の証明（`node.rs:149-157`）より単純な、lock による証明になっている（照合メモ 1）
- Δt と周期: 満たす。Δt は `now_ns - task.last_ns` の ns 差（`pool.rs:150`、手動は `:320`）。`last_ns` の起点は専用スレッドと同じ `last_update_ns[i]`（`:298-301`）。次の目標は同じ純関数 `next_target_ns`（`:403`、`node.rs:403-410`）に、ヒープに入れた目標そのものを渡して求める。ワーカーの割り当てを待った時間は Δt に実経過として入る。目標の格子は動かない
- 過負荷での飢え: 起きない。EDF で、待っているノードの期限は固定、走ったノードの次の期限は必ず「前の目標 + P」か「開始時刻 + P」以上に進む。どのノードも有限回のうちに先頭に来る（照合メモ 3）。周期の無いノードは開始時刻順の round robin になる。テスト `an_overloaded_worker_serves_the_nodes_in_turn` のオラクルとも一致する
- 学習則の焼き付け: 無い。`pool.rs` と `NodeHooks::enter(next_hook)` が知るのは hook の名前だけ。`grep -i "lambda|dual|pc_alm|epsilon"` は `runtime/` で 0 件

## must-fix

なし。

## should-fix

1. **M:N で弱まる仕様の約束が `specs/execution-model.md` に書かれていない**（確信度: 高。挙動はコードで確かめた。重大度は中。PR 2 で Python のノードが載る前に直しておきたい）
    - 例外の手順 3「ほかのノードを待たずに自分の `on_exception()` を呼ぶ。例外を投げたノードも、捕まえた直後に呼ぶ」（`specs/execution-model.md:196`）。プールでは `on_exception` もワーカーの空きを待つ。N 本のワーカーがすべて長い `on_update` を走らせていれば、それが終わるまで呼ばれない。さらに、失敗したノードの `on_exception` は列の後ろに回る。`call` の中の `fail`（`mod.rs:295-300`）が `wake_pool` でほかのノードの `Exception` を先に積み、失敗したノード自身は `record` → `schedule`（`pool.rs:376-379`）で最後に積まれるからである。workers=1 なら、失敗したノードの `on_exception` は全ノードの中で最後になる
    - 再開（`:107`）: ready の列が更新より優先なので（`pool.rs:135`）、ワーカーが少ないと、再開後の最初の更新は全ノードの `on_resume` が済むまで始まらない。Δt に入るのは「スレッドが起きる遅れ」ではなく「ほかのノードの `on_resume` とワーカーの割り当て」になる。止めている間の話なので不変条件には触れないが、ワーカーが 1 本のときは事実上「全員の再開を待つ」順序になる
    - タイムアウト（`:177`、`:242`）: `PauseTimeout` の「止まらなかったノード」に、ワーカーの空きを待っていただけのノードも入る（`control.rs:117-120`）。遅いノードを特定する手がかりとしては弱くなる。ワーカーが走らせていたノード（`At::Running`）を分けて出すか、仕様にそう書く
    - 停止の手順 3（`:98`、main の時点からある文）の「ワーカーが実行を割り当てなくなった時点で止まる」は実装と違う。プールのノードも区切りで `on_pause` を呼ばれ、`paused_at` を報告する（`pool.rs:306`、`:408-416`）
    - 案: 計画どおり PR 2 で直すなら、計画書の「specs で直す箇所」に例外の手順 3 とタイムアウトの名前の件を足しておく

2. **timekeeper を起こす回数が多く、同じタイマーで待つ全スレッドを巻き込む**（確信度: 中〜高。正しさには影響しない。M:N の目的である「コンテキストスイッチを減らす」に反する）
    - `place` は、新しい期限が timekeeper の待つ期限より早いと、`interrupt_timekeeper` → `timer.interrupt_waiters()` を呼ぶ（`pool.rs:274-278`、`:220-225`）。`interrupt_waiters` はタイマーの condvar の `notify_all` なので、同じタイマー（既定はグローバル）で待つ全員が起きる。この実行の timekeeper のほか、ほかの実行の専用スレッドや timekeeper、ホストの `sleep` も起きる
    - 典型例: 周期の無いノード A と周期 10 ms のノード B、workers=2。W2 が B の timekeeper のとき、W1 が A の更新を終えるたびに、A の次の期限（開始時刻 + 1 ns ≤ 今）が B の期限より早いので割り込む。ところが W1 は lock を持ったままループに戻って自分で A を取る。W2 は起きて lock を取り、何もせずに timekeeper に戻る。A の更新 1 回ごとに、無駄な起床が「この実行の 1 本 + タイマーで待つ全スレッド」ぶん生じる
    - 案: 戻す側のワーカー（`run_work` → `schedule`）が置いた期限が今の時刻以下なら、割り込まない。そのワーカー自身が次の周回で取る。今より後で、timekeeper の期限より早いときだけ割り込む。`place(Next::Hook)` の割り込み（`:262`）も、`wake_pool` の O(M) 回の `place` のたびに `interrupt_waiters` を呼ぶので、`wake_pool` の最後に 1 回にまとめられる。PR 3 のベンチマークで、混在する周期について非自発的スイッチを測る項目を足すとよい

3. **`enter` が失敗し続けると、`on_exception` と `on_stop` が呼ばれない**（確信度: 中。PR 1 では `enter` が既定の no-op なので顕在化しない。PR 2 で `_on_enter` を実装した時点で効く）
    - `invoke_as` は `enter` の失敗で `?` により `f` を呼ばずに返す（`mod.rs:272-275`）。逸脱 4 で `LAST_NODE` を更新しないので、次の `on_exception`（`pool.rs:373`）と `on_stop`（`:374`）でも `enter` を呼び直す。同じ理由（CUDA のデバイスが無い、など）でまた失敗すれば、どちらも呼ばれない。専用スレッドでも同じ（`node.rs:234`、`:243`）
    - `specs/execution-model.md` の例外の手順 5 は「例外を投げたノードの `on_stop()` も呼び、I/O などの資源を解放する」と約束している。PR 2 では、`on_exception` / `on_stop` の前の `enter` の失敗はログに出すだけにして hook を呼ぶか、呼ばないことを仕様に書くかを決める（判断ログの申し送りに 1 行足す）

4. **T 以下の保証をプールで直接確かめるテストが無い**（確信度: 中）
    - `builds_from_every_state_while_the_time_is_frozen` は手動モードで、`a_capture_does_not_move_the_target` は目標が動かないことしか見ていない。自由実行のプールで「更新の途中に capture が来ると、その更新は最後まで走り、`on_pause` はその後に来る。凍結の後に始まる `on_update` は無い。最後の更新の時刻 ≤ T」を確かめるテストが無い
    - 案: `Probe` の `gate` を使う。workers=1 で 2 ノード、ノード 0 の更新をゲートで止めた状態で `capture` を出す。ゲートを開けた後のイベント列が `Update(0)` → `Pause` × 2 → `StateDict` …… となること、ノード 1 の `Update` が凍結の後に 1 回も来ないこと、`last_update_ns` がすべて凍結した時刻以下であることを確かめる

## nit

1. `an_overloaded_worker_serves_the_nodes_in_turn`（`tests/runtime.rs:1302-1321`）は順序だけを確かめていて、Δt を捨てている（`.map(|(id, _)| id)`）。過負荷で「Δt が伸びるだけ」を確かめるには、各 Δt が実経過（前回の開始からの ns 差。ノード 0 なら 1 ns, 3 ms, 3 ms ……）に一致することも確かめるとよい（確信度: 高、優先度: 低）
2. `specs/native-core.md` の「ワーカープールの実装」に、「ノードはほかのノードの更新を待たない。待つのは空いたワーカーだけで、過負荷では Δt が伸びる。ブロッキングする hook はワーカーを占有する」を 1 行入れておくと、不変条件との関係が節の中で読める（計画の PR 2 の docs の項と重なるので、PR 2 でもよい）（確信度: 高、優先度: 低）
3. 制御の lock がプールの更新の hot path（1 回の更新で 2 回取る）に乗り、その下で `timer.time_ns()` がタイマーの Mutex を取る（`pool.rs:144`）。専用スレッドの速い経路が避けていた共有 lock なので、PR 3 のベンチマークでは M が大きいときの lock の待ちも見る（逸脱 1 の判断は妥当で、正しさには影響しない）（確信度: 中、優先度: 低）

## 検証結果

- `cargo test -p mujou-core`: 全件 pass（lib 34、`tests/runtime.rs` 42、`tests/time.rs` 37）。`tests/runtime.rs` を 5 回繰り返して 5 回とも pass（各 3.0 s）
- `just run` は回していない。先頭の `just format`（pre-commit を全ファイルに実行）が worktree を書き換えうるため、「コードは修正しない」の指示を優先した。pytest・pyright・docs-build は未実行（PR 1 の bindings は全ノードを専用スレッドで渡すので、Python から見た振る舞いは変わらない）
- EDF のテスト `updates_the_earliest_target_first_and_by_index_on_a_tie` のオラクルを手で展開し、期待の列と一致することを確かめた（照合メモ 2）

## 照合メモ

1. **T 以下の証明（プール）**: ワーカーが更新を始める条件は、lock L の下での (a) `command == Run` と (b) 時刻 `now ≤ ...` の読み取り。capture は L の下で (c) `command := Pause` と (d) 凍結（時刻 T）を行う。L で直列化されるので、(a)(b) が (c)(d) より前なら now < T、後なら (a) が偽。手動モードの経路（`advance` の `:313`）も L の下。`take` は取り出すときに `advance` をやり直すので、列に積んだ時点の判断が古くても、凍結や停止の後に更新が始まることは無い。進行中の更新が書くスロットの時刻も、凍結後は T のまま
2. **EDF のオラクル**: 周期 (3, 2, 1) h、起点 0。5h で全員が期限切れ → 期限順 2, 1, 0、Δt はどれも 5h。`next_target_ns`: n0 は (3, 5, 3) → 6、n1 は (2, 5, 2) → 4 ≤ 5 なので 7、n2 は (1, 5, 1) → 6。6h: n0 と n2 が同じ期限 → 添字順で 0, 2。Δt はどちらも h、次の期限は 9 と 7。7h: 1, 2（Δt は 2h と h）、次は 9 と 8。8h: 2。9h: 0, 1, 2（Δt は 3h, 2h, h）。テストの期待と一致する
3. **飢えの上界**: ノード j の次の期限 ≥ max(前の目標 + P_j, 開始時刻 + P_j) で、厳密に増える（P ≥ 1 ns）。待つノード i の期限 t_i は固定。i より先に取られるのは期限 < t_i（同じなら添字が小さい）の項目だけで、各 j につき高々 ⌈(t_i − 期限_j)/P_j⌉ + 1 回。今の時刻が t_i を超えて遅れたノードは「開始時刻 + P」で数え直すので、すぐ t_i より後ろに回る。よって i の待ちは有限で、増えるのは Δt だけ。ready の列（ライフサイクル）は制御の変化でしか積まれないので、自由実行の更新を飢えさせない
4. **Δt の起点**: 専用スレッドは `node_loop` に入るときに `last_update_ns[i]` を読む（`node.rs:104`）。プールは `WaitGo → Loop`（go の後）に同じ値を読む（`pool.rs:298`）。起動中はタイマーが凍結しているので、どちらも同じ時刻になる。再開後も `target_ns` と `last_ns` は保たれるので、`a_capture_does_not_move_the_target` を 3 つの形で満たす
