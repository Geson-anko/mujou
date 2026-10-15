# core-runtime WP-B 差し戻し対応の再レビュー（r2）

対象: worktree `.claude/worktrees/core-runtime`、`git diff 8b906ea..HEAD`（60068e8, 694d9c9, 7c585fb）。前回の指摘は `memory/agents/code-reviewer/core-runtime-B.md` と `memory/agents/algorithm-reviewer/core-runtime-B.md`。

## verdict: approve

must-fix はない。前回の must-fix（M1 のスレッド状態、algorithm の must-fix 1 の停止の保持）と should-fix・nit は、記録どおりに直っていることをコードと実験で確かめた。新たに見つけた指摘は should-fix 2 件と nit 4 件で、どれもマージを止めるものではない。

前回の指摘の確認:

- M1: `on_thread` が `Python::attach(|py| py.detach(body))` でスレッドを包む。PyO3 0.30 の `SuspendAttach` は `ATTACH_COUNT` を 0 に戻してから `PyEval_SaveThread` するので、hook の `Python::attach` は `PyGILState_Ensure` を通り、同じ tstate を使い直す（gilstate のカウンタは 2 → 1 に戻るので、tstate は消えない）。`threading.local`・`contextvars`・`decimal` の context・`threading.current_thread()` が hook 間で同じになることを両ビルドで確認した。ランタイムを 200 回起動・終了しても `threading.enumerate()` は増えなかった（`_DummyThread` は残らない）
- 停止の保持（S1 / algorithm の must-fix 1）: `Freeze` は理由が空でない間 `PauseHold` を 1 つ持つ。lock の順序は常に control → timer で、timer の lock の下で control の lock を取る箇所はない。2 つのランタイムが別々のホストスレッドから 300 回ずつ同時に capture しても、どの `build` の間もタイマーは止まったままで、時刻も動かなかった（下の検証結果）
- fast path（N2）: 制御側は control の lock の下で語を store（SeqCst）してから `timer.hold()`（timer の lock）を取り、ノードは語を load → `time_ns()`（timer の lock の下で `Instant::now()` を読む）→ 語を再 load する。ノードの timer の臨界区間が hold より後なら、mutex の synchronizes-with によって再 load が新しい語を見て、その時刻を捨てる。前なら、読んだ時刻は hold の reanchor 時刻 T 以下になる。`fail` も Stop の store → hold の順。保持がすでにある場合（自分の別の理由、他のランタイム、ホストのスイッチ）は時刻が止まっているので自明に成り立つ。epoch を語に含めるので、Pause → Run の ABA も起きない。証明に穴は見つからなかった
- S2・S3・S4・N1・N4〜N8、algorithm の should-fix 1〜3、nit 3: 記録どおり。spawn の失敗（`abandon`）では、起動済みのノードが `Stop` を見て `on_start` → `on_stop` で終わり、保持は解かれる。`build` の panic は再開してから `Internal`、`ExceptionCapture` の panic は捕まえて終了処理を続ける

## must-fix

なし。

## should-fix

### S1. hook の bound method を `_on_thread_start` の前に 1 度だけ引くので、インスタンス属性で hook を差し替えても効かなくなった

- 対象: `crates/mujou-python/src/runtime.rs:213-218`（`start` の最初の `Methods::of`）、`:174-177`
- 問題: `_on_thread_start` か `on_start` で `self.on_update = wrap(self.on_update)` のようにインスタンス属性で hook を差し替えると、以前（毎回名前で引く）は差し替えた方が呼ばれた。今は差し替える前の bound method が呼ばれ続ける。再現: scratchpad の `r2/methods_probe.py` で `['plain']`（以前なら `['wrapped', 'plain']`）。テストも docs もこの変化に触れていない
- 根拠: 計画書（`memory/agents/implementation-planner/core-runtime.md:166`）は「各 hook で…メソッドを名前で呼ぶ」としている。`src/mujou/_core.pyi:192` も「The runtime calls the hooks by name」と書いている。WP-D は `_on_thread_start` をフレームワークのスレッドごとの設定に使う予定で、ここで hook を包む実装を選ぶと何も言わずに効かなくなる
- 対応案: 引く時点を `start` の最後（`load_state_dict` の後）に移す。そうすれば `_on_thread_start`・`on_start` での差し替えが反映され、性能上の効果も変わらない。あるいは「`start` の前に引く」と `_core.pyi` と `Node` の docstring に書く
- 確信度: 高（現象は再現済み）。深刻度は低〜中

### S2. 他のランタイムが走っている間に `timer_state` 付きで `start` すると、そのランタイムの Δt が飛ぶ

- 対象: `crates/mujou-core/src/runtime.rs:946-951`（`Runtime::start` の `timer.load_state`）
- 問題: 停止の保持で複数のランタイムの同時実行に対応したので、起動時にプロセス共通の時刻を巻き戻す、または進める `load_state` が、走っている別のランタイムの Δt を壊す。再現（`r2/restore_probe.py`）: A を自由実行中に、B を `timer_state={"time_ns": now + 10 s}` で起動すると、A の次の Δt が 10.0014 s になった。過去の時刻に戻すと、A のノードは戻した分だけ更新しない。前回からある挙動だが、今回の修正で複数のランタイムを公式に扱うようになったので顕在化した
- 根拠: `docs/execution-model.md`「システム全体で時間は 1 つだけ」。陽的 Euler で Δt が飛べば発散する（algorithm の前回の should-fix 1 と同じ理由）
- 対応案: orchestrator の判断事項。(a) 他のランタイムが動いている（RUNTIME の数が 0 でない）ときの `timer_state` を `InvalidArgument` にする。(b) D の `run(state=...)` の制約として docs に書く。どちらにしても、WP-D に引き継ぐ
- 確信度: 高（再現済み）。深刻度は中（同時に複数のネットワークを走らせ、かつ片方を状態から復元したときだけ起きる）

## nit

- N1. `PyNode::panicked`（`crates/mujou-python/src/runtime.rs:202-211`）は `catch_unwind` の外（`Shared::invoke` の `unwrap_or_else` の中）で呼ばれる。PyO3 0.30 の `Python::attach` はインタプリタの終了処理中に panic する（`pyo3/src/internal/state.rs:81`）。そのため、終了処理中に hook が panic すると（hook 側の `attach` も同じ理由で panic する）2 回目の panic でノードのスレッドが巻き戻り、`exited` を報告しない。その後の `finish` は永久に待つ。インタプリタが終わるところなので実害はほぼない。`Python::try_attach` を使えば避けられる。実際には、終了処理中に自由実行のノードを走らせても panic の出力は出なかった（`r2/exit4_probe.py`。ノードは CPython の終了処理で止められる）。確信度: 中
- N2. 再入の判定がランタイムごとになったので、ランタイムをまたぐ循環（A の `build` が B の `capture` を呼び、同時に B の `build` が A の `capture` を呼ぶ）は `Reentrant` にならずにデッドロックする。制御スレッドは `check_signals` でシグナルを受け取れないので、Ctrl-C でも抜けられない。利用者が自分で作る循環なので、今は困らない。D の docs に「`build` から他のネットワークの待つメソッドを呼ぶときは循環させない」と書く程度でよい。確信度: 中
- N3. `on_thread` の docstring（`crates/mujou-python/src/runtime.rs:193-197`）は `contextvars` も共有されると書いているが、テスト `test_hooks_share_one_python_thread_state` は `threading.local` と `current_thread()` しか見ていない。制御スレッド（`build` とその呼び出しの間）でスレッド状態が保たれることもテストしていない。手元の実験では、どちらも成り立っていた。確信度: 高、優先度: 低
- N4. worktree の `memory/agents/plan-implementer/core-runtime-B.md` の「`lib.rs` は触らない約束なので、D か後続で直す」は古い。7c585fb で `lib.rs` の `pause`/`resume` の docstring を直している。確信度: 高

## WP-D への引き継ぎ（前回分に追加）

- ランタイムの停止・起動・例外のときの保持は、プロセス共通の時刻を止める。そのため、ネットワーク B の起動（`on_start`・`load_state_dict` の間）と capture の間、同時に走っているネットワーク A のノードは Δt=0 の待ちに入る（意図どおりで、Δt は壊れない）
- S2 の方針（復元と同時実行）
- 前回の「`Runtime` の pyclass は GC の追跡対象ではない」の補足: 起動していなくても、`__main__` のグローバルに `Runtime` を置き、`__main__` で定義した Node のサブクラスを渡すと、`Runtime → node → class → method.__globals__ → __main__ の dict → Runtime` の循環を GC が回収できない。終了時に `__main__` の他のオブジェクトの `__del__` も呼ばれない（`r2/d5.py`）。D で `__traverse__`/`__clear__` を足すか、`atexit` の shutdown で終える前提にする

## 検証結果

- just run: pass（format の全フックが Passed、cargo test 64 件 pass、pytest 123 passed、pyright 0 errors。実行後も worktree に変更なし）
- `tests/mujou/test_core_runtime.py` と `tests/mujou/test_core.py` の繰り返し（43 件）: free-threaded で 25 回、`PYTHON_GIL=1`（`sys._is_gil_enabled()` が True であることを確認）で 25 回、すべて pass。16 プロセス（2 つのビルドを半々）を同時に走らせる負荷試験を 3 回行い、48 回すべて pass（実時計のテストと CPU 時間のテストも含む）
- `cargo test --workspace` を 5 回: すべて pass
- 実験（scratchpad の `r2/`）
  - `exit_probe.py`: 2 つのビルド × 5 通り（del / keep / del-sleep / keep-sleep / shutdown）× 20 回の、ランタイムが走ったままでのインタプリタの終了。200 回すべて rc=0、stderr は空、ハングもなかった
  - `exit4_probe.py`: 終了処理（`is_finalizing()=True`）の間 0.3 s、自由実行のノードを走らせた。rc=0、panic の出力なし
  - `tstate_probe.py`: hook 間の `contextvars`・`decimal`・`current_thread()` の一致、`threading.enumerate()` が増えないこと（両ビルド）
  - `tle_probe.py`: 6 ノードのランタイム 2 つを実時計で走らせ、2 本のホストスレッドから 300 回ずつ同時に capture した。どの `build` でも、各ノードの `last_update_ns` が `build` 中の時刻 T 以下、`is_paused()` が真、`build` の間に時刻が動かない、が成り立った（両ビルドで 3 回ずつ。違反 0）
  - `methods_probe.py`（S1）、`restore_probe.py`（S2）、`d5.py`（引き継ぎの GC の補足）
- 成果物汚染（`</content>` など）: なし
