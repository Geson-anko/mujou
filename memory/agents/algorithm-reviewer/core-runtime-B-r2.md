# core-runtime WP-B 再監査（差し戻しへの修正）

対象: worktree `.claude/worktrees/core-runtime`、ブランチ `feature/2026-10-16/core-runtime`、`git diff 8b906ea..HEAD`（`60068e8`・`694d9c9`・`7c585fb`）。前回: `memory/agents/algorithm-reviewer/core-runtime-B.md`。

## verdict

**approve**（確信度: 高）。前回の must-fix（他の停止を解いてしまう）は解消した。fast path は「フラグ → すぐ凍結」に対して T 以下の保証を保つ。Δt に停止中の時間は混ざらない。学習則の焼き付けは無い。

ただし、複数のネットワークが同時に走るときの時間の意味について、前回見落としていた穴が 1 つある（should-fix 1）。1 つのネットワークを `timer_state` 付きで起動すると、走っている別のネットワークの時刻が巻き戻ったり飛んだりする。原因は前回の must-fix と同じで、シングルトンのタイマーを 1 つのランタイムが書き換えると、他のランタイムにも効いてしまう。今回の差分が持ち込んだものではなく、D で復元を公開するまでに決めればよいので should-fix とした。orchestrator の判断で must-fix に上げてもよい。

観点ごとの判定:

- **must-fix 1 の解消**: 解消した。`GlobalTimer` が保持の数（`holds`）とホストのスイッチ（`paused`）を別に持ち、`is_paused = paused || holds > 0` で判定する。`Freeze` は理由の集合が空でない間 `PauseHold` を 1 つ持つだけで、`resume()` を呼ばない。前回の再現 2 件を作り直したテスト（`test_one_capture_does_not_end_the_pause_of_another`、`test_a_pause_by_the_host_outlasts_the_runtime`）と、Rust の `hold` モジュールの 4 件がある。`hold()` と `drop` は、件数を変える前に `reanchor` する。そのため凍結の時刻は、件数が 0 から 1 になった瞬間の時刻になる。0 に戻るときは anchor の実時刻を張り替えるので、凍結中の実時間は時刻に入らない。RAII にしたことで、`Shared` が解放されれば保持も確実に解ける。この点は前より頑健になった
- **fast path の順序の保証**: 保たれる（照合メモ 1）。制御側は control の lock の下で `word.store(SeqCst)` → `interrupt_waiters`（タイマーの lock）→ `timer.hold()`（タイマーの lock の下で reanchor）の順に処理する。ノードは `w1` → `time_ns()`（タイマーの lock）→ `w2` の順に読む。ノードの `time_ns` が hold より先にタイマーの lock を取れば、読んだ時刻は T 以下になる。後に取れば、lock の解放と取得の同期によって `w2` が新しい語を見るので、時刻は捨てられる。epoch は `set_command` のたびに単調に増えるので、語の ABA は起きない。例外のとき（`fail`: Stop → hold）も同じ順序になる。手動モードは従来どおり lock の下で処理する。実測では、実時計で 4 ノードを自由実行し、1000 回取得しても `last_update_ns > T` は 0 件、build の最中に時刻や更新回数が動いたことも 0 件だった
    - 「凍結の後に始まる on_update は無い」という docs の文言は、壁時計の順序としては前回も今回も成り立たない。区切りの判定がフラグより前なら、その `on_update` が凍結の後に呼ばれることはある。ただしその時刻は T 以下で、取得は全ノードの `paused_at` を待つので、木には影響しない（nit 1）
- **Δt に制御の待ち時間が混ざらないか**: 混ざらない
    - 起動: `last_update_ns` を `Runtime::start` で保持を取り、`load_state` した直後、スレッドを作る前に設定する。`startup` は保持を解いてから `go` を出す。起動中の例外で取得した値は 0 ではなく凍結時刻になる（`test_the_start_of_each_dt_is_known_when_starting_fails`）
    - 停止と再開: `resume_after_pause` は保持を解いてから `Run` にする。`test_the_pause_takes_no_global_time` がある（実時計で build の中で 0.3 s 眠り、次の Δt が 0.25 未満）
    - 復元: `test_restores_the_time_and_where_each_dt_starts` が、`on_start` と `load_state_dict` の中でタイマーが停止中かつ時刻が T であること、最初の Δt が `(k + h)/1e9` に完全一致すること、T より先の起点では更新しないことを確かめている。前回の should-fix 1〜3 はすべて解消した
    - 実測: 1000 回の取得をはさんでも、各ノードの Δt の和と、開始時の `last_update_ns` からの差がナノ秒で一致した（ずれ 0）。最大の Δt は 0.11 ms
- **複数のネットワークの時間の意味**: 停止についてはおおむね整合している。A の取得中は B の時刻も止まり、B のノードは `on_pause` を受けずに Δt = 0 の待ちに入る（実測: A の build の 0.3 s の間、B の新しい更新は 0 回、B の `on_pause` は 0 回）。docs の「停止中はグローバル時間が進まないので、ダイナミクスからは見えない」は、B から見ても成り立つ。ただし docs はこのことを明示していない（nit 2）。また復元は整合しない（should-fix 1）
- **学習則の焼き付け**: 無い。`NodeHooks` に足した `on_thread` と `panicked` は、スレッドの寿命とログのための汎用の口である。bindings の `Methods` もライフサイクル hook の名前を持つだけ

## must-fix

なし。

## should-fix

1. **`timer_state` 付きの起動が、走っている別のネットワークの時刻を書き換える**（確信度: 現象は高（再現済み）。重大度は中〜高。今回の差分以前からある）
    - `Runtime::start` は保持を取ってから `timer.load_state(state)` を呼ぶ（`runtime.rs` の `start`）。`load_state` はシングルトンの `anchor_ns` を上書きするので、すでに走っている別のランタイムの時刻も T に飛ぶ
    - 再現（照合メモ 2）: A を実時計で自由実行し、0.1 s 後に B を `timer_state={"time_ns": now + 3600e9}` で起動した。A の次の Δt は 3599.99998 s になった。陽的 Euler のノードは発散する。`time_ns: 0`（巻き戻し）で起動すると、A は時刻が自分の `last_update_ns` に追いつくまで約 0.1 s 更新を止め、A のスロットの書き込み時刻も単調でなくなる。A の取得の最中に B がこの起動をすると、A の木の「時刻 T でそろう」も崩れる（`Shared::boundary` のコメントの論証は、タイマーの時刻が単調であることを暗に前提にしている）
    - docs「復元」節の「グローバルタイマーも戻し、木を取得した時刻 T から実行を始める」は、ネットワークが 1 つであることを前提にしている。計画の D は `WeakSet` で複数の Session を扱うので、`mujou.run` を 2 回呼べば公開 API から届く
    - 直し方の案（どれにするかは orchestrator／ユーザーの判断。D の設計にかかわる）:
        - (a) 他のランタイムが動いている間は、`timer_state` 付きの `start` を `RuntimeError`（または `ValueError`）にする。生きているランタイムの数をコアで数える必要がある。単純で、docs に制約を 1 行足すだけで済む
        - (b) 他のランタイムが動いているときはタイマーを戻さず、各ノードの `last_update_ns` を `now − T` だけずらして読み込む（スロットの時刻も同じだけずらす）。Δt の起点の相対関係が保たれ、他のネットワークの時刻も動かない。ただし「木の時刻 T から始める」という意味は、絶対時刻としては失われる
        - (c) docs に「タイマーの状態の復元はプロセス全体の時刻を変える。他のネットワークが動いているときに使ってはならない」と書くだけ。検出しないので事故は残る
    - テスト: (a) なら「A が走っている間の `start(timer_state=...)` がエラーになり、A の Δt が乱れない」。(b) なら「A の最大 Δt が通常の範囲に収まり、B の最初の Δt が `(k + h)/1e9` になる」

## nit

1. **docs「停止」節の「フラグを凍結より先に立てるので、凍結の後に始まる `on_update` は無い」は、壁時計の順序としては強すぎる**（確信度: 高、優先度: 低）。区切りでフラグを見る前に時刻を読んだノードの `on_update` は、凍結の直後に呼ばれることがある。そのときの時刻は T 以下で、ノードは次の区切りで止まる。成り立つのは「凍結時刻 T より後の時刻で始まる `on_update` は無い」のほうである。前回の nit 1（「Δt に混ざるのは制御側の 2 つの操作の間隔だけ」）と合わせて、D が実装の節を書くときに文面を直す
2. **1 つのネットワークの停止・起動がプロセス全体の時間を止めることを、docs が明示していない**（確信度: 高、優先度: 低〜中）。今回足された項目は「別のランタイムの停止を解いてしまう」理由として保持を説明しているが、帰結は書いていない。次の 3 点を一言書いておくとよい: A の取得中は B のノードも時刻が止まり、`on_pause` を受けずに Δt = 0 で待つ。A の起動（全ノードの `on_start`・`load_state_dict`、GPU へのモデルの読み込みなども含む）の間も B は止まる。ホストの `sleep` も延びる。リアルタイム用途で、2 つ目のネットワークの起動が 1 つ目を壁時計で止めることは、利用者が知っておくべき性質である
3. **fast path の 2 回目の読み（`w2`）を外す退行を検出するテストが無い**（確信度: 中、優先度: 低）。競合の窓が狭く、私の 1000 回のストレス実験も、再読みが無くても通る可能性が高い。決定的に確かめるには、Rust のテストでタイマーの読みの直前に割り込む口が要る（loom など）。手間に見合わなければ、`Shared::boundary` のコメント（論証は正しい）で足りる
4. **前回の nit 4（`run_epoch` の回帰テスト）は見送りのまま**。実装者の記録どおりで、異論は無い

## 検証結果

- `just run`（format → cargo test（28 + 7 + 29 件）→ pytest 123 件 → pyright）: すべて通過。実行後も worktree に差分は無い
- 追加の実験（scratchpad の `exp_r2.py`。コードは変えていない）: 照合メモ 2

## 照合メモ

1. **fast path の T 以下の論証**
    - 制御側（`capture`）: lock(control) → `set_command(Pause)` で `epoch += 1`、`word.store(SeqCst)`、`interrupt_waiters`（lock(timer)）→ `freeze.hold` → `timer.hold()`: lock(timer)、reanchor（`anchor_ns = T`）、`holds += 1`、unlock
    - ノード: `w1 = load` → `is_run` → lock(timer)、`now = ns_at(Instant::now())`、unlock → `w2 = load` → `w1 == w2` なら `now` を使う
    - タイマーの lock の全順序で、ノードの読みが hold より前なら、`Instant` の単調性と `ns_at` の単調性（`anchor_ns + (Δinstant as f64 * scale) as u64` は Δinstant について非減少）によって、`now ≤ T` になる。hold より後なら、`word.store` が hold の unlock より前に sequenced-before にあり、unlock が lock と同期するので、`w2` は新しい語以降を見る。epoch は単調に増えるので `w2 ≠ w1` となり、`Again` に進む。すでに凍結していた場合（この Runtime の別の理由、他の Runtime の保持、ホストのスイッチ）は、ノードが読む時刻が凍結値なので T 以下になる
    - 前提: 2 つの読みの間に、タイマーの時刻を非単調に動かす操作（`load_state`）が入らないこと。これを満たさないのが should-fix 1
    - `Freeze::release` は理由が無ければ何もせず、空になったときだけ保持を drop する。`release_all` は保持を drop する。`abandon` もそうする。保持の二重取得は `hold.is_none()` で防いでいる
2. **実験**（実時計、3.14t）
    1. 4 ノードで自由実行し、1000 回取得した。build の中で `last_update_ns > timer.time_ns()` は 0 件。build の最中に時刻・更新回数・`is_paused` が変わったのも 0 件。各ノードの Δt の和と、開始時からの `last_update_ns` の差のずれは 0 ns。最大 Δt は 0.000109 s
    2. A・B を同時に走らせ、A の build の中で 0.3 s 眠った。その間の B の新しい更新は 0 回で、B の `on_pause` は 0 回。B の最大 Δt は 0.0002 s
    3. A が走っている間に、B を `timer_state={"time_ns": now + 3600e9}` で起動した。A の最大 Δt は 3599.999977 s（should-fix 1）
    4. 同じく `time_ns: 0` で起動した。A は時刻が `last_update_ns`（約 0.1 s）に追いつくまで更新しない（should-fix 1）
