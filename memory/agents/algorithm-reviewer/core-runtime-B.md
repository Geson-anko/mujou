# core-runtime WP-B（ノードの実行と制御）の監査

対象: worktree `.claude/worktrees/core-runtime`、ブランチ `feature/2026-10-16/core-runtime`（`03e4c26`）、`chore/2026-10-15/core-scaffold..HEAD` の差分。基準は skill `locality-and-async`、`docs/execution-model.md`、`docs/nodes.md`、計画書の WP-B。

## verdict

**request-changes**（確信度: 高）。must-fix 1 件。ノードのループ・停止・再開・例外の流れは docs のとおりに書けている。ただし凍結の管理が「タイマーを止めるのはこのランタイムだけ」という前提に立っており、その前提が崩れると停止中の時刻の一貫性が壊れる。実験で再現した。

監査観点ごとの判定:

- **ノードが互いを待たないか**: 満たす。自由実行のループで取る lock は、制御の `Mutex<Control>`（区切りで指示を確かめ、時刻を読む短い臨界区間）、自分の `hooks[i]`、タイマーの lock だけ。他のノードの報告や完了を待つ箇所は無い。ノードを待たせるのは制御の操作に限られる: 起動の `go`（全ノードの `start` の完了待ち。docs の「全ノードの準備ができてから再開」）、停止（`Pause`）、手動の許可（`grant`）、終了（`stop_allowed`）
- **停止中の時刻が T 以下でそろうか**: ランタイムが 1 つでホストがタイマーに触らない限り満たす。区切りで指示を確かめるのと `timer.time_ns()` を読むのを同じ制御の lock の下で行い（`runtime.rs:429-452`）、`capture` は同じ lock の下でフラグを立ててすぐ凍結する（`runtime.rs:648-654`）。そのため、要求より前に読んだ時刻は T 以下になり、要求の後は必ず停止を見る。`last_update_ns` の store は `paused_at` の報告より前に lock を通して見える。例外のときの凍結（`fail`、`runtime.rs:381-388`）も同じ順序。ただし must-fix 1 の状況では崩れる
- **Δt に制御の待ち時間が混ざらないか**
    - 起動: タイマーを凍結 → `load_state` → スレッドを作る → `start`（`_on_thread_start` → `on_start` → `load_state_dict`）→ 凍結時刻で `last_update_ns` を設定 → 凍結を解く → `go`。満たす（`test_on_start_takes_no_global_time`）
    - 停止: フラグ → すぐ凍結。満たす
    - 再開: `resume_after_pause` が凍結を解いてから `Run` にする（`runtime.rs:752-756`）。満たす。実時計で build の中で 0.3 s 眠らせても、再開後の最初の Δt は 0.0005 s だった（照合メモ 2）。ただしこの確認はテストに無い（should-fix 2）
    - 復元: 「停止 → load → 起動 → 再開」の順。満たす。テストは無い（should-fix 3）
    - 手動: 許可を受けた時点の時刻で Δt を作る。仮想時計で `advance(h)` → `step` なら全ノードで `dt == h`（テストで完全一致を確認済み）
    - Δt はナノ秒の差を 1e9 で割って作る（`runtime.rs:469`）。`now <= last` なら更新しない
- **手動モードが同じループか**: 満たす。`node_loop` は 1 つだけで、手動モードの違いは区切りで `grant == Some(i)` を待つことと、1 回ごとに `end_step` を報告することだけ。Δt を作る処理、停止、例外の経路も共通。`Controller::step` は許可を順に出すスケジュールで、同期専用の更新の経路ではない。Gauss–Seidel と Jacobi の参照とビット単位で一致するテストがある
- **例外時の凍結と状態の取得**: docs の 1〜7 のとおり。最初の例外を捕まえたスレッドがその場で `Stop` にし、取得する設定ならすぐ凍結する。例外を投げたノードも、区切りで止まったノードも、停止中のノードも、他を待たずに `on_exception` を呼ぶ。全ノードの `on_exception` が終わってから制御スレッドで `state_dict` を呼び（失敗したノードは除く）、そのあと `on_stop`、join、凍結の解除の順。停止中に終わったノードは `on_resume` を呼ばれない。ただし起動中の例外では `last_update_ns` が 0 のまま取得される（should-fix 1）
- **学習則の焼き付け**: 無い。`NodeHooks` は `State` を型引数に取るライフサイクル hook だけで、ランタイムはメッセージの中身も Δt の使い方も知らない。PC-ALM の量は出てこない

## must-fix

1. **凍結の理由をランタイムごとに持っているが、タイマーの停止はシングルトン上の bool。他の誰かの停止を解いてしまう**（確信度: 現象は高（再現済み）、重大度は高）
    - `Freeze`（`runtime.rs:201-228`）は、自分の理由の集合が空から空でなくなると `timer.pause()`、空に戻ると `timer.resume()` を呼ぶ。一方 `GlobalTimer::pause/resume` は数を数えない bool（`time.rs:179-192`）で、タイマーはプロセスに 1 つしかない
    - 再現 1（照合メモ 3 の 2）: Runtime A と B を同時に自由実行し、A の `capture` の build の途中で B が `capture` を終えると、B の `release` がタイマーを再開する。A の build が終わるまでに時刻が約 0.198 s 進み、build の最後には `is_paused() == False` になった。A が「凍結時刻 T でそろった状態」として受け取る木は、そろっていない（D がこの build でスロットの値と時刻、`last_update_ns`、`timer.state_dict()` を読むので、T 以下の保証が崩れる）
    - 再現 2（同 3）: ホストが `timer.pause()` してから `Runtime.start()` すると、起動の終わりにタイマーが動き出す。`Freeze::hold` はタイマーがすでに止まっていたかを記録しないので、`release` が他人の停止を解く。`release_all_resumes_only_a_timer_it_froze` のテストは「hold していなければ resume しない」ことしか見ておらず、この場合を覆っていない
    - 計画の D は `atexit` で実行中の Session の集合（`WeakSet`）を止める設計で、複数のネットワークが同時に走ることを想定している。`mujou.run` を 2 回呼べば公開 API から届く
    - 直し方は 2 案。どちらにするかは orchestrator の判断:
        - (a) 推奨。停止の保持数（または理由ごとのトークン）を `GlobalTimer` 側に持たせ、保持がすべて解けたときだけ再開する。`Freeze` はその保持の 1 つになる。ホストの `_core.timer.pause()` も 1 つの保持として数えるかは決める（数えるなら `release_all` の意味も揃う）。タイマーの凍結はノードのダイナミクスから見えないという docs の主張は、複数ネットワークでもそのまま成り立つ（片方の取得中は、もう片方も時刻が止まって Δt=0 の待ちに入るだけ）
        - (b) 同時に走る Runtime を 1 つに制限し、2 つ目の `start` と、タイマーが止まっているときの `start` をエラーにする。単純だが、docs・計画のどこにもこの制約は書かれていないので、docs に足す必要がある
    - テスト: (a) なら Rust で「A が hold、B が hold、B が release してもタイマーは止まったまま」、Python で上の再現 1 を決定的にした版（スレッド 2 本、Event で順序を固定）

## should-fix

1. **`last_update_ns` を起動の完了まで 0 のままにしている。起動中の例外で取得した木に 0 が入る**（確信度: 高）
    - `Shared::last_update_ns` は `AtomicU64::new(0)` で作り（`runtime.rs:861`）、`Controller::startup` が全ノードの `start` が終わってから初めて設定する（`runtime.rs:596-600`）。`on_start`・`load_state_dict` が投げると `startup` はここに来ずに `finish` へ進むので、例外のときの取得の callback で `last_update_ns()` を読むと `[0, ...]` になる（再現: `[0]`、そのときの T は 2.03e8 ns）
    - D がこの値をそのまま例外のときの木に入れ、利用者がその木から復元すると、最初の Δt が T 秒ぶん（何時間でもありうる）になり、陽的 Euler が発散する。木の時刻は T 以下という約束は守られているが、「次の Δt の起点」という意味が壊れる
    - 直し方: `Runtime::start` で `load_state` の後、スレッドを作る前に、`config.last_update_ns` か凍結した時刻で設定する（タイマーはすでに凍結されているので値は変わらない）。`startup` の設定は不要になる。こうすると `start` の返事を待つ前に別スレッドから `last_update_ns()` を読んだ場合も正しい値になる

2. **再開後の Δt に停止中の時間が混ざらないことをテストしていない**（確信度: 高）
    - 「再開はタイマーを先に再開してからフラグを下ろす」「停止中は時刻が進まない」が守られていることは、実験では確かめた（照合メモ 2）が、テストに無い。起動については `test_on_start_takes_no_global_time` がある
    - 同じ形のテストを足す: 実時計・スケール 1 で、build の中で実時間 0.3 s 眠ってから戻すと、再開後の各ノードの最初の Δt は 0.25 未満（上界だけを見る）。順序を逆にする（フラグを下ろしてから凍結を解く／全ノードが止まってから凍結する）退行を検出できる

3. **復元の経路（`timer_state`・`last_update_ns`）を B のテストで確かめていない**（確信度: 高）
    - 計画では D の E2E が復元を確かめるが、「停止 → load → 起動 → 再開」の順序と Δt の起点は B が持つ不変条件で、B の単体テストで決定的に確かめられる
    - 案: 仮想時計で `start(manual=True, timer_state={"time_ns": T, "scale": 1.0}, last_update_ns=[T - k])` とする。`on_start` と `load_state_dict` の中で、`timer.is_paused()` が真で `timer.time_ns() == T` であることを記録する。`advance(h)` → `step()` で `dt == (k + h) / 1e9` に完全一致することを確かめる。異常系（長さの不一致、不正なスケール）で何も起動せずタイマーが止まったまま残らないことも確かめる

## nit

1. **docs の「Δt に混ざるのは制御側の 2 つの操作の間隔だけ」は、実装（と docs の手順そのもの）より強い**（確信度: 高、優先度: 低）。タイマーを先に再開するので、再開後の最初の Δt には `on_resume` の実行時間とスレッドが起きるまでの遅れも入る。起動時の `go` の後も同じ。通常のループの 1 周の手間が Δt に入るのと同じ性質で、正しさの問題ではない。D が `docs/execution-model.md` の実装の節を書くときに文面を合わせる
2. **自由実行の全ノードが、ループ 1 周ごとに 1 つの制御の `Mutex` を取り、その下でタイマーの `Mutex` も取る**（確信度: 中、優先度: 低）。意味の上のバリアではないが、ノード数が増えるとここで直列化する。専用スレッドが数本の今は問題ない。タスク実行クラスを入れるときに見直す。ただし T 以下の保証は「指示の確認と時刻の読みが、フラグ → 凍結に対して原子的」であることに依存している。置き換えるなら（epoch の atomic を読む → 時刻を読む → epoch を読み直す、など）この順序を保つこと
3. **Rust の API で `build` や `ExceptionCapture` が panic すると、制御スレッドが凍結を持ったまま死ぬ**（確信度: 中、優先度: 低）。タイマーは止まったまま、ノードは停止中のまま残る。bindings は `PyResult` を返すので Python からは起きない。hook と同じく `catch_unwind` で包むか、doc comment に「panic しないこと」と書く。同様に `Runtime::start` の `spawn(..).expect(..)` も、凍結を持ったまま panic する
4. **判断ログの `run_epoch` の競合（再開の直後の終了で `on_resume` が飛ぶ）に回帰テストが無い**（確信度: 中、優先度: 低）。非決定的なので決定的なテストは難しい。Rust で hook に Barrier を挟めば順序を固定できるので、手間に見合えば足す

## 検証結果

- `just run`（format → cargo test + pytest 61 件 → pyright）: すべて通過。実行後も worktree に差分は無い
- 追加の実験（`uv run python`、scratchpad の `exp.py`。コードは変えていない）: 照合メモ 3

## 照合メモ

1. **T 以下の保証の筋道**: ノードは `lock(control)` → command を見る → `time_ns()` → unlock → `update` → `last_update_ns.store(now)`。`capture` は `lock(control)` → `Pause` → `timer.pause()` → unlock。同じ lock の下にあるので、時刻を読む処理は凍結の前か後のどちらかに順序付けられる。前なら読んだ時刻は T 以下、後ならノードは `Pause` を見て時刻を読まない。スロットへの書き込みは `update` の中で、凍結後なら時刻は T。`paused_at` の報告は store の後で、`capture` は同じ lock の下でそれを読むので、build が読む値は確定している。例外のとき（`fail`）も同じ形
2. **再開の Δt**（実時計、ノード 1 つ）: build の中で `time.sleep(0.3)` → 再開後の最初の dt = 0.000497584 s。それまでの最大の dt も同じ値で、停止中の時間は入っていない
3. **実験**
    1. 上の 2
    2. Runtime A・B を同時に自由実行。A の capture の build の中で 0.3 s 眠り、0.1 s 後に別スレッドで B が capture → A の build の終わりで `is_paused = False`、A の凍結中に進んだ時刻 = 198,162,166 ns（must-fix 1）
    3. `timer.pause()` → `Runtime.start()` → `is_paused = False`（must-fix 1）
    4. `on_start` が投げるノード 1 つ、`on_exception_capture` の中で `last_update_ns()` = `[0]`、`timer.time_ns()` = 203,405,668（should-fix 1）
4. **例外の流れの照合**（docs「例外による終了」1〜7）: 1 は `PyNode::error` がログを出し、`fail` が記録する。2 は `fail` が `set_command(Stop)` する。3 は `node_finish` が `failed` を見て、他を待たずに `exception` を呼ぶ。停止中のノードは `pause_node` が epoch の変化で起き、`run_epoch <= epoch` なので `resume` を呼ばずに抜ける。4 は `finish` の `ready_to_stop`。5 は `exception_capture` の取得（`invoke`。失敗はログだけ）。6 は `stop_allowed` → `stop` → join → `release_all`。7 は `OnceLock` で最初の例外だけを記録する。いずれも一致した
5. **lock の順序**: 制御の lock → タイマーの lock（`Freeze`、`interrupt_waiters`、区切りでの時刻の読み）で一貫している。タイマーの lock の下で呼ぶ `interrupted` は atomic の epoch だけを読む。GIL は hook・build・capture の callback の中でだけ取り、そのとき制御の lock は持っていない。`hooks[i]` の lock を制御スレッドが取るのは、ノードが停止済みか終了処理に入った後だけ。逆順の取得は見つからなかった
