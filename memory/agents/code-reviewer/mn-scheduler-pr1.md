# M:N スケジューリング PR 1（コアのワーカープール）レビュー

対象: `git diff origin/main..HEAD`（3091a6f・7ce19fa・8d7b414）。計画書は `memory/agents/implementation-planner/mn-scheduler.md`（末尾のユーザーの決定を優先）、実装者の記録は `memory/agents/plan-implementer/mn-scheduler.md`。

## verdict: approve

must-fix はなし。下の should-fix を PR 1 の中で直すか、PR 2 に回すかは orchestrator が裁定する。

### 構造として確かめたこと（根拠）

- 排他: `At::Running` に移るのは `take`（ready から pop、pool.rs:131 付近の `run_worker` → `take`）と、ヒープの先頭を `earliest` が世代番号で検証した直後の pop だけで、どちらも制御の lock の下。`schedule` は `Ready | Running` を素通しし（pool.rs `schedule`）、`Running` から戻すのは `run_work` の末尾だけ。`at == Ready` と ready キューの要素が常に 1 対 1 で、ヒープの古い項目は `at != Heap(gen)` で捨てる。破れる経路は見つからなかった
- 待つ条件を変える箇所の網羅: `advance` が読む制御の状態（`command`・`epoch`・`run_epoch`・`go`・`grant`・`stop_allowed`・`failed`）を書く箇所を grep し、全部が `wake_pool` / `wake_node` を呼ぶことを確認した（`set_command`・`fail`・`startup`・`step`・`finish`・`abandon`）。`grant = None`（`end_step`）は待つ側を進めない方向なので不要
- timekeeper: 役は `timekeeper: Option` で 1 本に限られ、`keep_time` は lock の下で `sched_epoch` と `word` を読んでから待つ。早い期限・ready の追加は lock の下で epoch を増やして `interrupt_waiters`（timer の lock を取る）を呼ぶので、`wait_until_ns_or` の「条件はタイマーの lock の下で確かめる」契約により取りこぼさない。仕事を取ったワーカーは `run_work` で timekeeper 不在かつ仕事が残れば `notify_one` で役を渡し、待つワーカーが居なくても、戻ったワーカーが待つ前にキューを見直す。lock の順序は control → timer の一方向
- 凍結の保証: プールの更新は制御の lock の下で `command` と時刻を読む。`capture` は同じ lock の下で `set_command(Pause)` → `freeze.hold` をするので、凍結後に始まる `on_update` は無い
- 専用スレッドの経路: `node.rs` の変更は `next_target_ns` の切り出し（式は同値）と `end_step(&mut Control)` だけ。既存テストのアサーションは 1 行も変えずに `each_mode` で包んだだけ（`git diff -w` で確認）。bindings は `dedicated: Some(vec![true; n])` なので `workers = min(.., 0) = 0` でワーカーは立たない
- 追加の検証（worktree は変えず、scratchpad にコアを複製して実施）: 実時間で 12 ノード × workers 1/2/3 × 専用スレッド混在あり/なしで capture 100 回、10 ノードで capture と失敗の競合を 60 通り、手動モードで乱数（seed 固定）の step/capture/時刻の進めを 300 回、workers=2 で片方のワーカーを update に閉じ込めたまま他方のノードが期限どおり更新されること。6 並列 × 5 回すべて通った。既存の `tests/runtime.rs` も 6 並列 × 3 回、pool 関連のテストを 30 回回して失敗なし

## must-fix

なし

## should-fix

1. **`enter` が失敗し続けると、そのノードの `on_exception` と `on_stop` が呼ばれない。trait の doc と specs はこれと食い違う**（確信度: 高 / 深刻度: 中）
   - 対象: `crates/mujou-core/src/runtime/mod.rs:87-89`（"A dedicated thread calls it once, before `start`"）、mod.rs:272-275（失敗時に `LAST_NODE` を更新しない）、`specs/native-core.md:56`（「専用スレッドでは `on_start` の前の 1 回だけになる」）
   - 問題: 実装者の判断 4 により、失敗後の `on_exception`・`on_stop` の前にも `enter` を呼び直す。`enter` が失敗し続けると `invoke` は hook を呼ばずにエラーを返し、そのエラーは無視されるため、後始末の hook が黙って飛ぶ。scratchpad で `enter` が常に失敗するノードを作ると、専用スレッド・プールのどちらでも `enter` は 3 回呼ばれ、`start`・`exception`・`stop` は 1 回も呼ばれなかった
   - 根拠: doc と specs は「1 回だけ」と書いている。PR 2 で `_on_enter`（CUDA の device や stream の設定）が失敗したとき、利用者の `on_stop` が呼ばれないのは、今の専用スレッドの振る舞い（`on_start` が失敗しても `on_exception` → `on_stop` は呼ぶ）から外れる。どちらの振る舞いにするかを決め、doc・specs・テストを揃える必要がある
2. **`enter` の失敗の経路にテストが無い**（確信度: 高 / 深刻度: 中）
   - 対象: `crates/mujou-core/tests/runtime.rs` の `mod pool`
   - 問題: 「`enter` のエラーは次の hook の名前で報告する」「失敗したら `LAST_NODE` を更新せず、次の hook の前にまた呼ぶ」（判断 4）はどちらもテストで固定されていない。`Probe::enter` は失敗できない
   - 根拠: 計画書のテスト観点は Python 側で `_on_enter` の例外の名前を確かめるとしているが、名前の付け方の半分（panic のとき `hook` を使う）はコアの責務で、Rust の意味論のテストに置くべき（skill `testing-strategy` の 3 層の分担）
3. **`enter` が `Err` を返したとき、報告される hook の名前は実装側任せ**（確信度: 中 / 深刻度: 低）
   - 対象: mod.rs:273（`node.enter(hook)?`）、mod.rs:89
   - 問題: panic のときは `hook`（次の hook の名前）で報告するのに、`Err` のときは実装が返した `HookError::hook` をそのまま使う。doc の "An error stands for `next_hook`'s" は、実装が `hook: next_hook` を入れるべきという意味にも、コアが付け替えるという意味にも読める
   - 根拠: 計画書の「ここで出た例外は、次に呼ぶ公開の hook の名前で報告する」。コアで `e.hook = hook` に揃えるか、doc に「実装は `next_hook` を `HookError::hook` に入れる」と明記する
4. **ワーカーの間で timekeeper の役を受け渡す経路を直接確かめるテストが無い**（確信度: 高 / 深刻度: 中）
   - 対象: tests/runtime.rs の `mod pool`
   - 問題: 1 本のワーカーが hook に閉じ込められている間に、別のワーカーが timekeeper になって期限の来たノードを回すこと（`run_work` の `notify_one`、pool.rs:364）を確かめるテストが無い。既存テストの `Mode::Pool` はノードが 1 つなら `OneWorker` と同じになり、複数ワーカーで自由実行するのは `each_node_keeps_its_own_period` だけ。`an_exception_after_a_normal_stop_still_comes_before_stop` はノード 0 を閉じ込めるが、ノード 1 の更新は確かめていない
   - 根拠: レビュー依頼の lost wakeup の観点そのもの。`notify_one` を消してもスイートは通るはず（ノード 1 の更新は、閉じ込められたワーカーが戻ってから遅れて起きるだけ）。scratchpad の `another_worker_keeps_time_while_one_is_held`（workers=2、仮想タイマー、ノード 0 を gate で止めたまま時刻を 1 ms ずつ進め、ノード 1 の tick を毎回受け取る）の形で決定的に書ける
5. **specs の追記の文体が `memory/feedback_japanese_style.md` から外れる**（確信度: 中 / 深刻度: 低）
   - 対象: `specs/native-core.md:51-56`、`specs/execution-model.md:229`
   - 問題: 「〜デッドロックするからである」（理由を「からである」で付け足す）、「なった順に取る」（何が「なった」のか曖昧。「積まれた順」「期限が来た順」など）、「待たずに置いておき」（口語）、「lock を放して」（直訳調。「lock を解放して」）、読点が細かい（「コアは次の hook の直前に、同じ呼び出しの中で」「今の bindings は、Python のノードを」）
   - 根拠: memory/feedback_japanese_style.md の「読点で細かく区切らない」「理由を『〜だからである』で付け足さない」「英語の直訳を避け」

## nit

1. 排他の smoke test（tests/runtime.rs:1668-1686）は、アサーションでは重なりを捉えられない。ノードは `Mutex<N>` の中にあるので `busy` の swap は構造上 true にならず、仮に二重に取られたら `debug_assert`（mod.rs:266）が `catch_unwind` の外でワーカーを panic させ、ノードが `Running` のまま残って `finish` がハングする形で現れる。テストの失敗が「ハング」になる点を doc comment に書くか、計画書の「hook ごとの通し番号が単調に増える」を足すかのどちらか（確信度: 高）
2. `spawns_no_worker_without_nodes_in_the_pool`（tests/runtime.rs:1360）は、hook を走らせたスレッドの名前しか見ないので、何もしないワーカーが立っていても通る（プールのノードが 0 なら、ワーカーは `left == 0` ですぐ抜け、hook を呼ばない）。`spawns_no_more_workers_than_nodes_in_the_pool`（:1346）も、余分に立ったワーカーが偶然 hook を取ったときだけ落ちる確率的な検出。スレッドの列挙は移植性が無いので、ここは doc comment に限界を書くだけでもよい（確信度: 高）
3. `an_exception_after_a_normal_stop_still_comes_before_stop`（tests/runtime.rs:1547）は 50 ms の実時間の sleep で「ノード 1 が先にループを抜ける」順を作る。実装者の記録どおり結果は sleep の長さに依らないので flaky にはならないが、遅い runner では意図した経路を通らずに通る可能性がある（確信度: 中）
4. 先頭の期限が早まるたびに `interrupt_timekeeper` → `Timer::interrupt_waiters`（pool.rs:262, 276）を呼ぶ。周期の短いノードと長いノードが混ざると、短いノードの更新のたびに同じタイマーの全 waiter（ほかの実行の timekeeper・専用スレッド・ホストの sleep）を起こす。計画書の想定リスクに挙がっている件で、PR 3 のベンチマークで「周期の違うノードの混在」を条件に入れたい（確信度: 中）
5. `specs/native-core.md:52` は「起動・一時停止・再開・`on_exception`・`on_stop` の仕事を先に」とあるが、手動モードのステップの更新も同じ ready キューに入る（`advance` の `Command::Run if self.manual` が `Next::Hook`）。一言足すと実装と一致する（確信度: 高）
6. `specs/execution-model.md:229` のリンクの文言が「M:N スケジューリング」なのに、行き先は `#ワーカープールの実装`（確信度: 高）
7. コメントの折り返し: mod.rs:696 が 128 桁、`NodeHooks::on_thread` の doc（"or the control thread). The bindings of a language with per-thread state wrap it here,"）とモジュール doc の "The nodes never wait for each other; ..." が他より長い、control.rs:132-134 は "holding its lock, and" の後で不自然に改行している（確信度: 高）
8. `invoke_as(i, hook, enter: bool, f)` の bool 引数は呼び出し側で意味が読めない（`state_dict` 以外は `true`）。`state_dict` の中で直接書くか、名前付きの小さな関数に分けると読みやすい（好みの範囲）
9. merge の注意: `origin/docs/2026-10-18/specs-sync` の 0520a5a（`NodeHooks::state_dict` の doc から "(or has stopped after an exception)" を消した修正）と mod.rs で衝突する（`git merge-tree` で確認）。この PR は同じ段落を書き換えて古い文言を残しているので、解消時に 0520a5a の訂正を採る（確信度: 高）
10. 計画書 `memory/agents/implementation-planner/mn-scheduler.md` がどのブランチにもコミットされていない（worktree では untracked）。実装者の記録が参照しているので、この PR か別の chore で入れる（確信度: 高）
11. 実時間のテスト 2 本（`on_the_real_clock_it_updates_about_once_per_period`・`the_time_scale_scales_the_updates_per_real_second`）も `each_mode` で 3 回回るので、合計 6 秒になった。許容範囲だが、ノード 1 つのテストでは `Pool` と `OneWorker` が同じ構成になる（既定のワーカー数は `min(並列度, 1) = 1`）ので、1 ノードのテストは 2 通りで足りる（確信度: 高）

## 検証結果

- just run: pass（pre-commit 全項目、cargo test 34 + 42 + 37、pytest 558 passed / 18 skipped、pyright 0 errors、mkdocs build --strict）
- 追加: 既存の runtime テストの 6 並列 × 3 回、pool 関連の 30 回連続、scratchpad の負荷テスト 5 本の 6 並列 × 5 回、いずれも失敗なし
- 成果物汚染（`</content>` など）: なし

---

# 再レビュー（レビュー対応、`git diff 8d7b414..HEAD`、5 コミット）

## verdict: approve

must-fix はなし。前回の should-fix 1〜5 はすべて解消した（4 は変異を入れて落ちることも確かめた）。残るのは下の nit だけ。

### 確かめたこと

- `Entry` と `clean_up`: `on_exception`・`on_stop` を呼ぶ箇所は 3 つ（`node.rs` の `node_finish` の 2 か所と `pool.rs` の `run_work`）で、すべて `clean_up`（`BestEffort`）に置き換わった。`invoke(i, "on_exception" | "on_stop", ..)` は grep でもう残っていない。`Required` は `call` 経由の `on_start`・`on_update`・`on_pause`・`on_resume`、`Skip` は制御スレッドの `state_dict` だけ。呼ぶ回数を決める状態（`ExceptionHook`、`Phase::Finish` → `Done`、専用スレッドの `stop_allowed` ループ）は前回から変わっていないので、二重呼び出しも抜けも増えていない
- 名前の付け替え: `Required` で `enter` が失敗すると `HookError { hook, source }` に組み直す（mod.rs:122-127）。panic は `guarded` が `hook` の名前で包み、`NodeHooks::panicked` を呼ぶ。`BestEffort` では失敗を捨てて本体を呼び、`LAST_NODE` は更新しないので、次の hook の前にまた `enter` を呼ぶ
- 終了の経路: scratchpad の `enter_failure_is_reported_and_retried` を新しいコアで回すと、専用スレッドでもプールでも、常に失敗するノードは `on_start` を呼ばれず `exception` → `stop` を 1 回ずつ呼ばれる（前回は 0 回）。起動失敗（`on_start` の前の `enter`）・切り替え時の失敗（`on_update` の前）・停止中の例外（前回の負荷テスト）のどの経路でも、`exception` と `stop` は 1 回ずつだった
- 足した 4 本のテストが意図した経路を通るか
  - `a_failed_enter_fails_the_next_hook_and_still_lets_the_node_clean_up`: `Probe` は `boom("enter")` を返すので、`"on_start"` の断定は付け替えを確かめている。`[Exception, Stop]` の断定は `BestEffort` を確かめている（`Required` のままなら空になる）。Err と panic、専用スレッドとワーカー 1 本の 4 通り
  - `a_failed_enter_on_a_switch_fails_the_update`: ワーカー 1 本で最後に起動したのはノード 1 なので、ステップでノード 0 に切り替わって 2 回目の `enter` が失敗する。意図どおり
  - `another_worker_keeps_the_time_while_one_is_held`: scratchpad の複製で `run_work` の `notify_one`（pool.rs:364）を消すと、`recv_timeout` で 5 秒後に落ちた。実装者の記録どおり
  - `a_capture_freezes_the_time_before_any_later_update_in_the_pool`: ワーカー 1 本をノード 0 の update に閉じ込め、期限の来たノード 1 を待たせたまま capture する。凍結を待つ 1 ms おきの `is_paused()` の確認は順序を作るだけで、待つ長さは結果を変えない。ノード 1 が更新せずに `on_pause` に進むことと `last_update_ns ≤ T` を断定しており、凍結の保証を確かめている
  - `an_overloaded_worker_serves_the_nodes_in_turn` の Δt の期待値 9 個を手で追い、`next_target_ns` と update 中に時刻を進める動きから導けることを確かめた
- specs: `specs/native-core.md` の追記（`enter` の失敗、ready キューに手動ステップの更新も入ること、過負荷で $\Delta t$ が伸びること）は実装と一致する。`specs/execution-model.md` の停止の手順 3・再開・停止のタイムアウト・例外の手順 3 の追記も実装と一致する。ただし例外の手順 3 は下の nit 1 を参照。リンク先 `execution-model.md#例外による終了` は見出しがある。文体は前回の指摘（「からである」・「なった順」・「置いておき」・「放して」・細かい読点）がすべて直り、`memory/feedback_japanese_style.md` に沿っている

## must-fix

なし

## should-fix

なし

## nit

1. `specs/execution-model.md` の例外の手順 3 の追記「例外を投げたノード自身の `on_exception()` は、ほかのノードの分より後に回る」は言いすぎ（確信度: 高）。`fail` の `wake_pool` の時点で待っていたノードの `on_exception` は先に積まれるが、そのとき別のワーカーで hook を実行中だったノードは、例外を投げたノードより後に積まれる。「その時点で待っていたノードの分より後」とすると正確になる。また、同じ手順の前半「例外を投げたノードも、捕まえた直後に呼ぶ。安全停止などをなるべく早く行うため」と、プールでは逆になっている。`record` の後で例外のノードの `Exception` を ready の先頭に積めば（`push_front`）前半の約束を保てる。直すか今の記述のまま残すかは orchestrator の裁定に任せる（確信度: 中）
2. `a_failed_enter_on_a_switch_fails_the_update` は `Enter` の行を除いて比べるので、「失敗した後の hook の前に `enter` をもう一度呼ぶ」（trait の doc に書いた約束）は断定していない。ノード 0 のログに `Enter("on_exception")` と `Enter("on_stop")` があることを足すと、doc の約束がテストで固定される（確信度: 高）
3. `specs/execution-model.md` の再開の追記「その場合 $\Delta t$ に入るのは、ほかのノードの `on_resume()` とワーカーの割り当てを待つ時間になる」は主語と述語がねじれている（「入るのは〜時間である」か「$\Delta t$ には〜を待つ時間が入る」）（確信度: 中）
4. 前回の nit 3（50 ms の sleep）と nit 9（specs-sync の 0520a5a との衝突）は、対応しない理由が記録にある。妥当と判断する

## 検証結果（再レビュー）

- just run: pass（pre-commit 全項目、cargo test 34 + 46 + 37、pytest 558 passed / 18 skipped、pyright 0 errors、mkdocs build --strict）
- 追加: scratchpad の複製に新しい `src/` と `tests/runtime.rs` を写し、負荷テスト 5 本と runtime テスト全体を 6 並列 × 4 回ずつ回した。失敗なし。`notify_one` を消す変異では `another_worker_keeps_the_time_while_one_is_held` が落ちた（確かめた後、複製は元に戻した）
- 成果物汚染: なし
