# core-runtime WP-B（Rust のノード実行・制御と PyO3 bindings）レビュー

対象: worktree `.claude/worktrees/core-runtime`、`git diff chore/2026-10-15/core-scaffold..HEAD`（36f1ad2, 03e4c26）

## verdict: request-changes

must-fix は 1 件（M1）。デッドロック・Condvar の取りこぼし・lock 順序・hook の順序は問題が見つからなかった。lock の順序は常に control → timer で、hook は control lock の外で呼ばれる。hooks[i] の lock は node と control が同時に取らない設計になっている。

## must-fix

### M1. hook を呼ぶたびに Python のスレッド状態が作り直され、同じノードの hook 同士でスレッドローカルな Python の状態が引き継がれない（1 hook 約 55 µs のコストもかかる）

- 対象: `crates/mujou-python/src/runtime.rs:133`（hook ごとの `Python::attach`）と `crates/mujou-core/src/runtime.rs:876-884`（ノードのスレッドは寿命の間 Python に attach したままにならない）
- 問題: Python が作っていないスレッドで `Python::attach` を呼ぶと `PyGILState_Ensure` が PyThreadState を作り、抜けるときの `Release` でカウンタが 0 に戻ってそれを破棄する。その結果、hook ごとに別のスレッド状態になる。実際に試すと、`on_start` で設定した `threading.local` と `contextvars` が `on_update` から見えず、`threading.current_thread()` も呼ぶたびに別の `_DummyThread` になった（free-threaded と `-X gil=1` の両方で再現。scratchpad の `probe_b.py`）。`threading.get_ident()` は OS のスレッド ID なので一致する。既存のテスト `test_hooks_run_in_order_on_the_node_thread` が通るのはこのためで、この問題を検出できない
- 性能: 実時計で何もしない `on_update` を自由実行すると約 18,000 回/秒（約 55 µs/回）だった。同じメソッドを Python のスレッドから呼ぶと約 1,700 万回/秒。差のほとんどはスレッド状態を作って壊すコストだと考えている（原因の切り分けは推定）
- 根拠: `docs/nodes.md` の hook 表は `on_start` を「ノードを実行するスレッドの上で」呼ぶと定めている。計画書 WP-B/D の `_on_thread_start` も「スレッドごとの設定」を前提にしている。D が使う torch の grad mode と CUDA stream は C++ の thread_local なので残る（grad mode は確認済み）。一方、Python レベルのスレッドローカル（`threading.local`、`contextvars`、`decimal` の context、torch の一部の Python 側 TLS）は消え、利用者のノードは静かに誤動作する
- 修正の方向（コードは書かない）: ノードのスレッドの寿命の間 PyThreadState を 1 つ保ち、待つときだけ detach する（例: bindings が「スレッド本体を包む」口を `NodeHooks` か `Runtime::start` に持つ）。回帰テストとして「`on_start` で設定した `threading.local` を `on_update`・`on_pause`・`on_stop` で読める」を足す
- 確信度: 現象は高（再現済み）、原因の帰属と性能の数字は中

## should-fix

### S1. 凍結の理由（`Freeze`）が Runtime ごとにしか数えられず、シングルトンのタイマーを使う他の主体の停止を解いてしまう

- 対象: `crates/mujou-core/src/runtime.rs:204-228`、`:868`、`:738`
- 問題 1: Runtime A の `capture` の `build` の間に Runtime B を `start` すると、B の Startup の解除が `timer.resume()` を呼び、A の停止中にタイマーが動き出す。`build` の始めには `is_paused()==True` だったのが、終わりには `False` になった（`probe_b3.py`）。A の停止中の経過が、A のノードの再開後の最初の Δt に入る
- 問題 2: ホスト（またはテスト）が `_core.timer.pause()` した状態で `start` すると、その停止が解かれる（`probe_b.py`: `host-paused timer still paused after start: False`）。`capture` と例外の終了処理でも同じことが起きる
- 根拠: `docs/execution-model.md`「停止」の「凍結した後は…時刻 T の状態として一貫する」と、タイマーはシングルトン（「システム全体で時間は 1 つだけ」）であること。D の計画は実行中の Session を `WeakSet` で持っており、複数のネットワークが同時に走る前提に見える
- 対応案: (a) 同時に走る Runtime を 1 つに制限し、D の `run` でエラーにして docs に書く。(b) 凍結の理由の数を `GlobalTimer` 側に持たせる。どちらにするかは orchestrator の判断事項
- 確信度: 高（再現済み）。深刻度は複数のネットワークを同時に走らせることをサポートするかどうかで変わる

### S2. `Runtime.start` が検証より前にノードを取り出すので、引数の誤りで `ValueError` になった後にやり直せない

- 対象: `crates/mujou-python/src/runtime.rs:358-375`（`pending.take()`）と `crates/mujou-core/src/runtime.rs:819-874`（`last_update_ns` の長さとタイマーのスケールの検証）
- 問題: `start(last_update_ns=[1, 2])`（ノードが 1 つ）は `ValueError` になり、続けて `start()` を呼ぶと、何も起動していないのに `RuntimeError: the runtime has already started` になる（`probe_b.py` で再現）。`timer_state` のスケールが不正な場合も同じ
- 根拠: 判断ログは「不正なら何も起動せずに `ValueError`」を意図としている。状態は変えないのが期待される振る舞い
- 確信度: 高。深刻度は低〜中（D は引数を自分で組むので踏みにくい）

### S3. `start` の復元用の引数（`last_update_ns`・`timer_state`）がテストされていない

- 対象: `tests/mujou/test_core_runtime.py`（この 2 つの引数を使うテストが無い）、`crates/mujou-core/tests/runtime.rs`（同じ）
- 問題: WP-B の公開 IF の一部で、D の復元のビット一致テストはこれに依存する。次を B の段階で確かめておくべき
  - `load_state_dict` の中では `timer.time_ns()` が復元した T に等しく、`is_paused()` が真である
  - 最初の `dt` が `(T + advance - last_update_ns[i]) / 1e9` になる
  - `last_update_ns` が T より先にあるときは更新しない
  - 長さの不一致は `ValueError` になる
- 根拠: skill `testing-strategy`（正常系・異常系）、計画書「テスト観点 WP-B」の起動の順序
- 確信度: 高

### S4. Runtime を参照切れで解放したときの振る舞い（待たずに終了を要求する）がテストされていない

- 対象: `crates/mujou-core/src/runtime.rs:1015-1021`、判断ログの「`Runtime` を参照が切れて解放すると…」
- 問題: 判断ログが振る舞いとして約束しているのにテストが無い。`del runtime; gc.collect()` の後で `on_stop` が呼ばれることは手元で確かめた（`probe_b.py`）
- 確信度: 中

## nit

- N1. `test_waits_without_spinning_until_the_time_moves`（`tests/mujou/test_core_runtime.py:236`）は、名前が言う「空回りしない」ことを検出できない。Δt=0 で空回りしても `on_update` は呼ばれないので、このテストは通ってしまう。名前を実際に確かめていること（制御の要求で起きる・Δt=0 では更新しない）に合わせるか、Rust 側でループの回数を数える。確信度: 高
- N2. 自由実行のループは毎回、共有の control の `Mutex` を取り、その下でタイマーの `Mutex` も取る（`crates/mujou-core/src/runtime.rs:429-452`）。全ノードが更新 1 回ごとに 1 つの lock で直列化される。停止との一貫性のための設計だが、`epoch` の atomic を読む → 時刻を読む → `epoch` を読み直す（seqlock 型）でも同じ保証は作れる。M1 を直すとこちらがボトルネックとして見えてくる。確信度: 中
- N3. あるノードが例外を投げた後に、別のノードの `on_resume` が呼ばれることがある。Run の後、そのノードが起きる前に Stop が来た場合で、`run_epoch` による判断ログの意図どおり。on_exception より前で、`on_update` は無いので docs には反していない。記録として残す。確信度: 高
- N4. panic のときの `hook` 名がコアの名前（`update`）で、Python のノードの名前（`on_update`）とそろっていない。panic はログにも出ない（docs の「パスとトレースバックをログに出す」）。ネイティブのノードは future work なので今は実害なし。確信度: 高
- N5. `thread::Builder::spawn(..).expect(..)`（`runtime.rs:882`, `:897`）が失敗すると、Startup の凍結が解かれないまま panic し、起動済みのスレッドは `go` を待ち続ける。起こる確率は低い。確信度: 中
- N6. `ON_RUNTIME_THREAD` はすべての Runtime で共通の印なので、Runtime A のノードから Runtime B の待つメソッドを呼んでも `Reentrant` になる（デッドロックはしない）。今は困らない。確信度: 高
- N7. `PyNode::call` は毎回、メソッドを名前で引く（`call_method1`）。性能だけの問題。確信度: 高
- N8. `start` の待ちの最中に Ctrl-C すると、ホストには `KeyboardInterrupt` が返るが、起動は裏で続く（`inner` は設定済み）。D の `atexit` で止まるので実害は小さいが、docstring に書いておくとよい。確信度: 中

## WP-D への引き継ぎ（IF の欠けと制約）

- M1 が直るまでは、ノードのスレッドの判定に `threading.get_ident()` を使う（`threading.current_thread()` は hook ごとに変わる）。`_on_thread_start` で行う Python レベルの TLS 設定は残らない（torch の grad mode と CUDA stream は残る）
- S1 の方針（同時に走る Runtime は 1 つに限るか）を D の `run` に反映する
- `capture_state_on_exception=True` で仮想時計のとき、例外の記録から終了処理の完了までタイマーが停止している。その間にホストが `advance` すると、`NetworkError` ではなく `TimerError`（停止中）になりうる
- `Runtime` の pyclass は GC の追跡対象ではない。ノード → Port → Session → Runtime のような循環があると回収されず、明示の `shutdown` か `atexit` でしか止まらない
- ノードの hook と `capture` の `build` から、`Network` の待つメソッド（`state_dict`・`shutdown`・`join`・`_step`）を呼ぶと `RuntimeError`（Reentrant）になる

## 検証結果

- just run: pass（format 全フック Passed、cargo test 全 pass、pytest 61 passed、pyright 0 errors。実行後に worktree の変更なし）
- `tests/mujou/test_core_runtime.py` を free-threaded で 5 回、`PYTHON_GIL=1` で 5 回: 全 pass
- `cargo test --test runtime` を 3 回: 全 pass
- 再現スクリプト（scratchpad）: `probe_b.py`（M1・S1 の問題 2・S2・S4）、`probe_b2.py`（M1 の性能）、`probe_b3.py`（S1 の問題 1）
- 成果物汚染（`</content>` など）: なし
