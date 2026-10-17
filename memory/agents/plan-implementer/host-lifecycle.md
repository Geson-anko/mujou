# host-lifecycle（ホスト側の振る舞いをユーザーの決定に合わせる）の判断ログ

依頼: orchestrator から、ユーザーの決定 1〜8 を skill `solo-dev-cycle` で実装する。worktree `host-lifecycle`（PR #13 の上）。Rust（`crates/`、`_core.pyi`、`test_core*.py`）とツール周り（`.pre-commit-config.yaml`、`pyproject.toml`、`_port.py`）は別の worktree が変更中なので触らない。今のファイル構成と API のまま振る舞いだけを変える。

## 計画（公開 IF）

- `Network.run(*, pause_timeout=30.0, _manual=False)`、`mujou.run(network, *, pause_timeout=30.0, _manual=False)`。`capture_state_on_exception` を削除
- `Network.exception_state_dict()` を削除
- `Network.state_dict()`: 実行中に加えて正常に終了した後も呼べる。run の前は `RuntimeError`、例外で終了した後は `NetworkError`
- `Network.shutdown(timeout: float = math.inf)` / `Network.join(timeout: float = math.inf)`: 時間内に終わらなければ `TimeoutError`。負か NaN は `ValueError`
- `Network.subscribe(port)`: 呼んだ時点の最新値を最初に返す
- `TensorSpec.zeros(*, device="cpu")`、`TensorSpec.rand(seed=None, *, generator=None, device="cpu")`、`TensorSpec.empty(*, device="cpu")`

## 計画外の判断

- **終了後の木のタイマーの時刻は、各ノードの最後の更新の時刻とスロットの書き込みの時刻のうち最も遅いもの。** スケールは今のタイマーのもの。終了後もグローバルタイマーは進むので、`timer.state_dict()` をそのまま入れると、復元した系の最初の Δt に終了から保存までの時間が混ざる。コアは終了した時刻を記録していないので、観測できる最も遅い時刻で代える。`on_stop` が send した値や、終了後にホストが write した値の時刻も含まれる（スロットの時刻がタイマーより後にならないように）。テスト: 実行中の木と、終了直後に仮想時計を 100 秒進めてから取った木が一致する
- **終了後の取得はホストのスレッドで `_capture_state()` を呼ぶ。** `mujou._node.capture_state(node)` を足した（`thread_ident` と同じく `_node` の中から private な属性に触る）。CUDA のノードは `_capture_state` が自分のストリームを同期する。`_hooks.inside` が一時的に立つので、`state_dict()` の中から Network のメソッドを呼ぶと `RuntimeError`、Port の `recv` は所有ノードのスレッドでないので `RuntimeError`（テストあり）
- **実行中に `state_dict()` を呼び、待つ間に正常に終了した場合**、コアの `capture` は `RuntimeError`（"the runtime has finished"）を返す。そのときは `is_finished` を確かめて終了後の取得に切り替える（`NetworkError` はそのまま送出）
- **`shutdown(timeout)` が有限のとき、終了の要求は daemon のスレッドで出す。** コアの `Runtime.shutdown()` は終わるまで待つだけで、待たずに要求する API が無い（Rust は触らない）。呼んだスレッドは `Runtime.join(timeout)` で待つ。時間切れでも要求は残り、ネットワークは終了を続ける。`timeout=math.inf` のときは今までどおり `Runtime.shutdown()` を直接呼ぶ。要求用のスレッドの `NetworkError` は握りつぶし（呼んだ側の `join` が同じ例外を送出する）。後で Rust に「待たずに終了を要求する」API を足せば、このスレッドは要らない
- **`timeout` の検証は Python 側で行う**（`not timeout >= 0` で負と NaN を `ValueError`）。`shutdown` は検証してから要求するので、誤った `timeout` では何も起きない
- **`subscribe` の最初の値は、イテレータを最初に回したときではなく `subscribe` を呼んだ時点で読む。** ジェネレータの本体は遅延実行なので、`outbox.read()` を `subscribe` の中で呼んで渡す。最初の値を返す前にも `check()` する（例外で終了していれば `NetworkError`）。終了した後に呼ぶと、最後の値を 1 つ返して終わる
- **インタプリタ終了時の停止（`threading._register_atexit` の置き換え）:** 最初の `run` で daemon の監視スレッド（`mujou-exit`）を 1 本立て、`threading.main_thread().join()` で主スレッドの終わりを待ってから走っているネットワークを止める。`atexit.register(_shutdown_all)` も残し、non-daemon の join の後に、止まるまで待つ（監視スレッドと並行に `shutdown` を呼んでもコアの終了の要求は冪等）。監視スレッドの中の例外は握りつぶし、atexit の側がもう一度試して送出する
  - CPython 3.14 の `threading._shutdown` は `_threading_atexits` を呼ぶ → `_main_thread._os_thread_handle._set_done()` → `_thread_shutdown()`（non-daemon の join）の順。`main_thread().join()` は `_set_done` で戻る
  - 確かめたこと: 小さなスクリプト（daemon が `main_thread().join()` で起きて Event を立て、non-daemon がそれを待つ）が 3.14t の `-X gil=0` / `-X gil=1`、GIL ありの通常ビルド（Homebrew の 3.14.8）のすべてで終了した。`_watch_exit()` を無効にすると `TestInterpreterExit::test_ends_a_subscribing_thread` が 60 秒でタイムアウトすることも確かめた（テストが置き換えを検出できる）
- **Rust の `on_exception_capture` 引数は Python から渡さないだけにした。** `_core.pyi` の `Runtime.start(on_exception_capture=...)`・`Runtime.exception_state()` と `tests/mujou/test_core_runtime.py` の該当テストは残っている。Rust の変更が入った後で掃除する
- `Node` の docstring と `docs/nodes.md` から「`on_exception` は `state_dict()` が返す状態を変えない」という約束を外した。例外のときの取得のための約束だったため
- `TensorSpec.rand` の `seed` は `torch.Generator(device)` で作る。同じ seed でもデバイスの種類が違えば値が違う（docstring に書いた）。`generator` は `device` 上のものを渡す
- `device` は `zeros`・`empty`・`rand` のどれもキーワード専用にした。`DictSpec.zeros` には足していない（依頼の範囲外）
- テストで device の引数を確かめるのに meta デバイスを使った（GPU なしで `device` が効いていることを見られる）
- コミットは関心事ごとに分けた（spec、状態の取得と例外時の取得の削除、timeout、subscribe、pause_timeout の既定値、終了時の停止、docs、この記録）。各コミットで `just run` を通した

## 確かめたこと

- `just run`（format → cargo test + pytest 全件 + pyright strict）が各コミットで通過
- `tests/` 全件を free-threaded（3.14t）で 5 回、`PYTHON_GIL=1 python -X gil=1` で 5 回: すべて通過（304 passed, 13 skipped）。gil=1 で `sys._is_gil_enabled()` が True になることも確認
- CUDA のテスト（`TestCuda::test_state_dict_after_the_end_waits_for_the_work_of_on_stop`、`TestTensorSpec::test_rand_on_cuda_is_deterministic` を含む）は CUDA の無い環境なので skip（未確認）

## 残る質問

- 終了後の木のタイマーの時刻を「最後の更新・書き込みの最も遅い時刻」にしたが、終了した時刻（`shutdown` を要求した時刻など）のほうがよいか
- `shutdown(timeout)` の時間切れの後、要求用の daemon スレッドが残る。Rust に「待たずに終了を要求する」API を足してよいか（Rust の worktree の作業が終わった後）
- `DictSpec.zeros` にも `device` を足すか

## レビュー対応（2026-10-16、code-reviewer / algorithm-reviewer の request-changes、orchestrator の裁定）

- **終了後の `write` は `RuntimeError`**（algorithm must-fix 1）。終了後の時刻が木のタイマーに入るのを防ぐ。code should-fix 2（終了後の木でスロットとタイマーを別々に読む競合）もこれで消えるうえ、`_ended_tree` は `_tree` が読んだスロットの時刻からタイマーを決めるようにした（スロットは 1 回だけ読む）。`write` が終了の確認を通った直後に終了が完了した場合は、その値が終了後に書かれ得る（まれな競合）。そのときも max を取るので、スロットの時刻は木のタイマーを越えない
- **木のタイマーは `max(T_req, 各 last_update_ns, 木のスロットの時刻)`**（質問 1 への裁定）。T_req は `Session.shutdown` が最初に終了を要求する直前に `timer.state_dict()` で記録する（スケールも同時に記録。algorithm nit 2）。前の「最後の更新・書き込みの最大値 M」は、最後の更新から shutdown までの時間を捨てていた。正常な終了の経路は `Session.shutdown`（ホストの shutdown と終了時の `_shutdown_all`）だけなので、終了後の取得では必ず記録されている（`cast` で前提を置いた）
- **`state_dict()` は `is_finished` を読んでから `check()` する**（code must-fix）。`write` も同じ順序にした
- **有限 timeout の shutdown の要求スレッドは Session ごとに 1 本**で、終了済みなら起動しない（code should-fix 1）
- **`on_stop` は `state_dict()` が返す状態を変えない**、を `Node` の docstring と `docs/nodes.md` に書いた（algorithm should-fix 2）。on_stop が例外を投げた後は、今どおりログに出すだけで正常な終了として扱う（約束を破ったノードの責任）
- **`DictSpec` にも `zeros(*, device)` と `empty(*, device)` を足した**（質問 3 への裁定）。`MessageSpec` の抽象メソッドも `zeros(*, device)` と `empty(*, device)` にした
- テスト: 終了後の木から復元して続けた結果が止めずに続けた結果と一致する（`b_first` でパラメータ化、最後の入力を 0.3 s 見せてから shutdown）、木のタイマーが T_req とそのときのスケールになる、終了後の write の `RuntimeError`、負の timeout で何も要求しないこと（`join(timeout=0)` の `TimeoutError` と `_step`）、有限 timeout の shutdown が例外の終了で `NetworkError`、要求スレッドが 1 本、終了後の `state_dict()` の中から Network のメソッドを呼ぶと `RuntimeError`。自由実行のテストは実装の式の写しをやめ、「要求の前に読んだ時刻 ≤ 木のタイマー ≤ shutdown の後に読んだ時刻、木の中の時刻はすべてタイマー以下」にした
- nit: 終了時の停止の docs と docstring を「join の前に終了を始め、atexit で終わるのを待つ」に直した。`_stop_at_exit` に CPython の実装に依存し `TestInterpreterExit` で守る旨を書いた。終了後の取得でノードの `state_dict()` の例外がそのまま伝わることを `Network.state_dict` の docstring に書いた
- **記録だけ（対応しない）:**
  - code nit 5: fork した子では `_watching` が True のまま残り、子に監視スレッドが無い。子でネットワークを走らせると、non-daemon の join の後の atexit でしか止まらない。今は fork 後の子で走らせる用途が無い
  - algorithm should-fix 2 の後半: on_stop が例外を投げた後の `state_dict()` は、送出も警告もせずに木を返す
- 実行中の取得が終了と競合する分岐（`except RuntimeError`）と `state_dict` の check の競合は、決定的に起こせないのでテストを足していない
