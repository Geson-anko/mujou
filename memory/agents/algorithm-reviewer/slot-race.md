# slot-race（arc-swap をスロットごとの mutex に替えた修正）の監査

対象: worktree `.claude/worktrees/slot-race`、ブランチ `fix/2026-10-16/slot-race`（`3e0f677`）、`edaf433..HEAD` の差分。基準は skill `locality-and-async`、`docs/execution-model.md`・`docs/messages.md`・`docs/native-core.md`・`docs/network.md`、`memory/project_async_node_execution.md`・`memory/project_native_core_rust.md`。

## verdict

**request-changes**（確信度: 高）。コードの修正は要らない。止めている理由は 2 つで、どちらも文書と手続きの問題である。1 つは設計の決定を変える修正なので、ユーザーの確認が要ること（must-fix 1）。もう 1 つは、「待たない」と書いた文書が 3 か所残っていること（should-fix 1）。技術的な中身は正しく、不変条件も満たしている。ユーザーが mutex 方式を認めて文書を直せば approve でよい。

- 原因の分析: 正しい（確信度: 高）。arc-swap 1.9.2 のソースで確かめた。`Debt::pay_all`（`src/debt/mod.rs`）は、どの storage の書き手でも、ポインタの値が一致する fast slot の debt を払う。`HybridProtection::attempt`（`src/strategy/hybrid.rs`）の最後の分岐は、debt が払われていたら、払ったのは自分の storage の書き手だとみなして `ptr` を返す。storage A で解放されたアドレスが storage B で再利用されると、A の読み手が B の値を受け取る。型が違えば型の取り違えになる
- ノードが互いを待たない: 満たす。lock を持つ区間は、すべての経路で `Arc` の複製・差し替え・`ptr_eq`・版の読み出しだけである。バッファを埋める処理と古いメッセージの drop（プールへの返却）は lock の外にある（照合メモ 1）。ノードの更新が、隣のノードの進み具合を待つことはない。待つのは、相手が数命令の区間を抜けるまでの間だけである
- 版番号による一斉切り替え: 保たれる。以前より強くなった。読み手は pair の lock を持ったまま版を読むので、各 `load` の結果は読んだ版の値だけで決まる（照合メモ 2）
- 停止中の時刻 ≤ T: 影響しない。時刻は `publish` の前に `timer.time_ns()` で取る。凍結の後に取った時刻は T である。ノードがループの区切りで止まる前に `publish` を終えるので、その後ホストが読めば、書かれた値が見える（mutex の happens-before が成り立つ）
- `wait_newer`: 呼び出し元は `src/mujou/_session.py` の `subscribe`（`_follow`）だけである。bindings は GIL を外して呼び、doc comment で「Port には出さない」と書いている。待ち手は Condvar の中で lock を手放して眠るので、書き手が待つのは、待ち手が値を確かめる区間だけである。通知の取りこぼしもない（照合メモ 3）
- 学習則の焼き付け・局所性・計算グラフ: この修正は関係しない。スロットの型に新しい量は入っていない

## must-fix

1. **「atomic なポインタ差し替え」という設計の決定を、ユーザーの確認なしに書き換えている**（確信度: 高）
    - 2026-10-11 の大元の決定（`memory/project_async_node_execution.md`、skill `locality-and-async` の「排他制御は値の読み書きの lock だけ」）には反しない。変わるのは、そのあとに決めた実装の方式である。`docs/native-core.md` と `memory/project_native_core_rust.md` には「atomic なポインタ差し替え（`arc-swap` など）… OS の mutex で長く待たせない」とあった。差分では、実装者がこの 2 つを自分で書き換えている。CLAUDE.md の「破る必要があるときはユーザーに確認し、理由を `memory/` に残す」に当たるので、merge の前にメインからユーザーに確認する
    - 性質の変化はこうである。以前の読み手は lock-free で、`load` は書き手が何をしていても進めた。今は lock を持つスレッドが OS に横取りされると、その間、相手は待つ。待つ時間に上限は無い（優先度の逆転）。将来の M:N のタスクプールでは、プールのワーカーがこれで止まると、そのワーカーに載ったタスクがすべて止まる
    - ユーザーに判断してもらう材料。いずれも監査役の見立て
        - (a) 現在の mutex 方式を採る（推奨）。正しさを示しやすく、Miri・TSan・ASan も通った。臨界区間は数命令しかない。チャンネルのスロットは書き手 1・読み手 1 なので、ほとんど競合しない。さらに `Outlet::send_with` は毎回 `timer.time_ns()` を呼び、これがプロセスに 1 つの `Mutex<State>` を取る（`crates/mujou-core/src/time.rs:193`）。横取りで待つという性質は、もっと競合する形ですでにある。スロットの mutex で新しい種類の待ちが増えるわけではない
        - (b) lock-free を保つ別の実装（epoch や hazard pointer 系の crate、または自前の実装）に移る。読み手が書き手を待たないことは保てる。その代わり、依存を新しく監査する手間か、unsafe の実装を検証する手間が増える
        - (c) どちらにするかと別に、arc-swap の upstream に報告するかどうか（調査記録にも、ユーザーが決めることだと書いてある）
    - 確認が取れたら、決定と理由（lock-free を手放した理由と、残る横取りの問題）を `memory/project_native_core_rust.md` に決定として残す。今の差分は事実を書き足しただけで、ユーザーの決定にはなっていない。残る問題は `docs/open-decisions.md` のタイマーの Mutex の項と並べて、同じ扱い（性能か遅延が問題になったら見直す）で載せる

## should-fix

1. **「待たない」という記述が 3 か所に残っている**（確信度: 高）
    - skill `locality-and-async` 47 行の「atomic なポインタ差し替えとバッファプール」。`.claude/` の編集になるので、must-fix 1 の確認と一緒に直す（skill `edit-dot-claude`）
    - `docs/network.md` 173 行の「この切り替えは、書き手も読み手も待たずに行う」。今は、各スロットの短い臨界区間だけ待つ
    - `crates/mujou-core/src/slot.rs` の `InputGroup` の doc comment にある「without the writer or any reader waiting」。同じ理由で古い
    - `docs/network.md` 195 行の「書き手は通知するだけで待たない」も、厳密には「待ち手が値を確かめる短い区間だけ待つ」になった。ノードを待たせないという趣旨は変わらないので、直すかどうかは任せる
    - 書き換える案: 「書き手も読み手も、相手の更新の進み具合は待たない。待つのは、スロットごとの lock の下で `Arc` を複製・差し替える短い区間だけである」

## nit

1. 回帰テスト `readers_never_see_a_message_of_another_slot` は、実スレッドを使う確率的なテストである。修正前のコードでも 4 割ほどしか落ちない。skill `testing-strategy` は決定的なテストを優先し、実スレッドの並列の検証は別の層に置くとしている。回帰テストとしては妥当なので残してよい。doc comment に「確率的な smoke test で、修正前のコードでは約 4 割で落ちる」と書いておくと、将来このテストが通ったときに過信されない（確信度: 中）
2. `wait_newer` の `notify_all` で起きた待ち手は、それぞれスロットの mutex を取り直す。`subscribe` が多いと、外向きのスロットの書き手と短く競合する。今の用途（ホストの消費者がいくつか）では問題にならない（確信度: 中、優先度: 低）

## 検証結果

- `just run`: 通過（cargo test 全件、pytest 335 passed / 11 skipped、pyright 0 errors）。ログ: scratchpad の `slot-race-just-run.log`
- `cargo test -p mujou-core --lib slot::` を 40 回繰り返した: 落ちたのは 0 回
- arc-swap 1.9.2 のソース（`~/.cargo/registry/src/*/arc-swap-1.9.2/src/debt/mod.rs` の `pay_all`、`src/strategy/hybrid.rs` の `attempt`）を読み、調査記録の 6 段の手順が実際に起きうることを確かめた
- 呼び出し元の確認: `wait_newer` の呼び出しは `src/mujou/_session.py:340` だけ。`Slot.load` の Python 経路（`crates/mujou-python/src/slot.rs:397`, `:603`）は GIL を持ったまま呼ぶ。臨界区間の中では GIL を取らないので、デッドロックは起きない

## 照合メモ

1. **lock の中で何をしているか**（全経路）
    - `Slot::load`: `Arc::clone(&lock(..))`。guard は文の終わりで落ちる
    - `Slot::publish` / `Slot::restore`: `let old = mem::replace(&mut *lock(..), msg);`。一時値の guard は `let` 文の終わりで落ちる。`notify_all` と `drop(old)` は lock の外。`restore` の `build` も外
    - `Slot::wait_newer`: lock の中では `ptr_eq` と `Arc::clone` だけ。`last` は呼び出し側が持つ参照で、lock の中では drop されない。`last` を持っている間はそのアドレスが再利用されないので、`ptr_eq` で ABA は起きない
    - `InputSlot::load`: lock の中で版を Acquire で読み、`resolve` と `Arc::clone` をする
    - `InputGroup::write`: fill は全部 lock の外で済ませる。pair の lock の中では `resolve`・`clone`・`replace` だけをして、`drop(pair)` のあとに `drop(old)` する。lock の順は write_lock → pair で、pair を 2 つ同時に持つ経路は無い。読み手は pair を 1 つしか取らないので、デッドロックしない
    - `InputSlot::restore`: write_lock → pair の順。`old` は一時値の guard が落ちてから drop する（write_lock はまだ持っているが、これはホスト専用）
2. **版による一斉切り替え**。書き込み W は、版 V から V+1 にする書き込みとする
    - 読み手が Y の pair を取ったのが W の lock より前の場合。読み手は lock の中で版を読む。版の読み → 読み手の unlock → W の lock → W の `version.store(V+1)` の順に happens-before でつながるので、読み手は V+1 を読めない。結果は古い値になる
    - 読み手が W の lock より後に Y の pair を取った場合。pair は新しい（`next=(V+1, new)`、`current` は版 V の値）。版が V なら current（古い値）、V+1 なら new を返す
    - 1 つ後の書き込み W2 が pair を書き換えたあとに読む場合。W の `store(V+1)` → W2 の pair の更新 → 読み手の lock とつながるので、読み手は版 ≥ V+1 を読む。繰り上げた current を、古い版で読み違えることはない
    - したがって、各 `load` の結果は「読んだ版の値」だけで決まる。版は 1 つの atomic 変数で、値は単調に増える。どの瞬間に見ても、外から書けるスロットは同じ write の値で揃う。以前は「pair を先に、版を後に」の順序と Acquire の組で保証していた。今は mutex がそれより強い順序を与える
3. **通知の取りこぼし**（`waiters` が Relaxed で足りる理由）
    - 待ち手が先に lock を取った場合: fetch_add → 待ち手の lock → 値を確かめる → `wait_timeout` で unlock する（Condvar の待ち行列に入るのと同時）→ 書き手の lock → replace → unlock → `waiters.load`。fetch_add はこの load より happens-before なので、load は fetch_add 以降の値を読む。この待ち手の fetch_sub は起きた後にしか来ないので、読む値は 1 以上になる。`notify_all` は、すでに待ち行列にいる待ち手を起こす
    - 書き手が先に lock を取った場合: 待ち手は lock を取った時点で新しい値を見るので、眠らない
    - 偽の起床とタイムアウトは loop で扱っている
