# core-runtime WP-D 再々レビュー（2 回目の差し戻しへの修正）

対象: worktree `.claude/worktrees/run-network`、ブランチ `feature/2026-10-16/run-network`。修正コミットは a37a88d（コアの `_after_update` / `_capture_state`）、229d8d8（CUDA の順序付け・デバイスの正規化・lock・load の検査）、065459c（docs）、032751f（判断ログ）。差分は `git diff 032751f~4..032751f` で見た。前回の指摘は `memory/agents/{code-reviewer,algorithm-reviewer}/core-runtime-D-r2.md`。

## verdict: request-changes

前回の must-fix 3 件（添字の無い cuda、取得のコピーの後の同期、取得の前の順序付け）は直っている。should-fix と nit も、意図どおりに直っている。インスタンス属性で包む方式をやめ、コアの hook にした裁定も正しく実装されている。Rust と bindings の変更は正しい。例外のときの hook 名、bound method を引くタイミング、例外のときの取得が `_capture_state` を通ることを確かめた。

それでも request-changes にするのは、CUDA の順序付けに起動時の穴が 1 つ残っているからである（must-fix 1）。ホストのスレッドが `run()` の前に積んだ仕事（`__init__` での確保など）と、ノードのストリームとの間に順序が無い。取得と復元は直したが、もっともよく通る `run()` の経路が抜けている。また、`_run_lock` を起動の間ずっと持つ変更で、`write` が起動の間ブロックするようになった。`on_start` がホストの書き込みを待つ場合は、待ち合いになる（should-fix 1、再現済み）。

前回の指摘ごとの判定:

- 直った: code M1（`__init__` で添字を決める）、code M2 / algorithm must-fix の WAR（`_tree` の最後で制御スレッドの current stream を同期）、algorithm must-fix の RAW（`_capture_state` の最初の同期）、code S1（インスタンス属性が残らない。テストあり）、S2（docstring）、S3（スロットの組と時刻の型を先に検査し、確認してから書き換える。テストあり）、code nit 1〜6、algorithm nit 1〜3・5
- 直したことで新しい問題が出た: code nit 2 / algorithm nit 1（`_run_lock`）。should-fix 1 を参照
- 記録だけにした: algorithm nit 4（裁定どおり）

## must-fix

### M1. `run()` の起動の前に、ホストのスレッドの current stream を同期していない（起動時の RAW）

- 対象: `src/mujou/_session.py:192-227`（`Session.run`）、`docs/torch.md:71-81`
- 問題: ノードが `__init__`（ホストのスレッド）で確保・初期化した CUDA の Tensor（`torch.zeros(..., device=self.device)` など。`Node.device` の docstring もこの書き方を勧めている）は、ホストのスレッドの current stream（通常はレガシーの既定のストリーム）に非同期に積まれる。ノードのストリームはプールから取るので `cudaStreamNonBlocking` であり、既定のストリームを暗黙には待たない。`run()` はホストのストリームを同期せずにノードを起動する。そのため、ノードが `on_start` / `on_update` で自分のストリームから読むとき、初期化のカーネルが終わっている保証が無い。起動までにスレッドの作成などで時間がかかるので、ふつうは間に合う。それでも、大きな初期化や GPU が混んでいるときには、何の知らせも無く初期化前の値を読みうる
- 根拠: r2 の algorithm must-fix と同じ種類の問題（ホストや制御スレッドのストリームとノードのストリームの順序付け）である。`load_state_dict` の経路は、コピーの後に `_synchronize_current_streams()` で直した。`load_state_dict` を通らない、ふつうの `run()` の経路だけが抜けている。docs/torch.md は取得と復元の順序付けだけを書いていて、構築のときに確保した値には触れていない。CUDA の無い環境なので実機では再現していない（PyTorch のストリームの意味論から推論した）
- 直し方の方向（参考）: `run` の `runtime.start` の前（lock の中）で `_synchronize_current_streams()` を呼び、docs に「`run()` を呼んだホストのスレッドが積んだ仕事は、起動の前に終わる。ほかのホストのスレッドで確保したものは、そのスレッドで同期してから `run()` する」と書く
- 確信度: 中 / 深刻度: 中（起こる確率は低いが、起きると初期値が気づかれないまま壊れる）

## should-fix

### S1. `run()` が起動の間ずっと `_run_lock` を持つので、ほかのホストのスレッドの `write` が起動の終わりまでブロックする。`on_start` がホストの書き込みを待つと、待ち合いになる

- 対象: `src/mujou/_session.py:195-227`（`run` が `runtime.start` の間 lock を持つ）と `:281-290`（`write` は、どちらの分岐でも lock を取る）
- 問題: 起動の待ちには、全ノードの `on_start` / `load_state_dict` が含まれる。モデルの読み込みのように長い `on_start` の間、ほかのホストのスレッドの `write` はすべて止まる。`on_start` がホストからの最初の入力を待つ（`recv` を poll する、外部の Event を待つ）と、`write` は起動を、起動は `write` を待つ。r2 の時点では、`write` は `_runtime` を lock なしで見て `_group.write` に進んだので、ノードに届いていた
- 根拠（再現、scratchpad の `r3/probe_start_write.py`）: `on_start` で `x.recv_with_time()` を 3 秒まで poll し、別のホストのスレッドが 0.2 秒後に `net.write` する。free-threaded と `PYTHON_GIL=1` の両方で、`on_start gave up: the write never arrived`、`writer: write returned after 2.81s`（起動が終わるまで戻らない）になった。上限を付けなければ待ち合いになる。docs/network.md:154 は複数のホストのスレッドからの `write` を認めているが、起動の間にブロックすることは書いていない
- 直し方の方向（参考）: lock が要るのは「復元した、まだ走っていない」ときの分岐だけである。`_timer_state is None` のときは lock を取らずに `_group.write` へ進める。あるいは、起動中かどうかの状態を別に持ち、lock は分岐の判定の間だけ持つ。少なくとも docs に「`run()` の起動が終わるまで、ほかのスレッドの `write` は待つ。`on_start` でホストの書き込みを待たない」と書く
- 確信度: 高（事実は再現済み）/ 深刻度: 中〜低（ホストのスレッドが 2 本あり、`on_start` が書き込みを待つときだけ待ち合いになる）

### S2. `_synchronize_current_streams` が見えるすべての CUDA デバイスを同期するので、使っていない GPU にも CUDA のコンテキストを作る。docs の「使っているデバイスごとに」とも違う

- 対象: `src/mujou/_session.py:485-490`。今回から、取得のたびに `_tree`（`:455`）でも呼ばれる
- 問題: `torch.cuda.current_stream(index).synchronize()` は、`DeviceGuard` でそのデバイスに切り替えて `cudaStreamSynchronize` を呼ぶ。そのデバイスにまだコンテキストが無ければ、ここで primary context を作る。GPU が複数ある計算機で、ノードが GPU 0 だけを使う場合でも、最初の `state_dict()` / `load_state_dict()` で全 GPU にコンテキストを作る（1 つあたり数百 MB）。exclusive-process モードの GPU をほかのプロセスが使っていれば、`cudaErrorDevicesUnavailable` で `state_dict()` が失敗しうる。コンテキストの無いデバイスには、このスレッドが積んだ仕事も無いので、同期する意味が無い。docs/torch.md:79 は「制御スレッドの現在のストリームを（使っているデバイスごとに）同期する」と書いており、実装と合っていない（r2 の `_synchronize_cuda` も全デバイスを回していたが、そのときは load だけだった）
- 直し方の方向（参考）: ネットワークのノードのデバイス（`{n.device for n in self._nodes if n.device.type == "cuda"}`）か、コピーした Tensor のデバイスだけを同期する
- 確信度: 中（PyTorch の C++ 実装と CUDA の挙動から判断した。実機では確かめていない）/ 深刻度: 中（複数 GPU の共用機だけで起きる）

## nit

- N1 `crates/mujou-python/src/runtime.rs:254-257`: `_after_update` が投げると、`NetworkError` のメッセージと例外の木の `"hook"` に、private な名前 `_after_update` が出る。CUDA の非同期のエラー（`on_update` のカーネルの device-side assert など）は、ここの `synchronize` で表に出るので、利用者には `on_update` ではなく `_after_update` のエラーに見える。docs（nodes.md の「利用者は override しない」の段落か、execution-model.md の例外の節）に一言あるとよい。`state_dict` のようにエラーの名前を `on_update` に寄せる手もある（確信度: 中）
- N2 `src/mujou/_node.py:145-146` のコメント「a node paused between updates has no work left on its stream」: `on_pause` などほかの hook の仕事は残りうる。今は `_capture_state` の最初の同期で受けているので、正しさには影響しない。コメントが言いすぎている（確信度: 中）
- N3 CUDA のキャッシュアロケータとストリーム: `load_state_dict` の deepcopy（ホストのスレッドのストリーム）や `__init__` で確保した Tensor は、確保したストリームのプールに属する。ノードがそれを手放すと（`self.w = new` のように置き換えると）、ノードのストリームのカーネルがまだ読んでいる間でも、ブロックは既定のストリームのプールに戻る。ホストが並行して既定のストリームで確保すると、そのブロックを再利用しうる（`record_stream` が無い）。docs/torch.md の「状態」はその場での更新を勧めているので、起きにくい。今回の差分の外の設計の話で、記録だけでもよい（確信度: 低〜中）
- N4 `src/mujou/_node.py:81-91`: `device="cuda"` のノードは、構築したとき `torch.cuda.current_device()` で CUDA を初期化する。構築の後に fork する使い方（DataLoader の fork のワーカーなど）では、子プロセスで CUDA が使えなくなる。`cuda:N` の N が範囲外であることは、構築のときには検査せず、`_on_thread_start` の `set_device` で初めて分かる（確信度: 低）
- N5 テストの抜け: 例外のときの取得が `_capture_state` を通ることと、そのときの制御スレッドの印（`in_hook`）を、Python の側で確かめるテストが無い（Rust の経路は通常の取得と同じ関数なので、コードを読めば分かる）。`_shutdown_all` の `BaseException` の分岐にもテストが無い（確信度: 低）
- N6 `src/mujou/_session.py:202`: `run` は `_runtime` を lock の中で先に立てる。そのため、起動中に別のホストのスレッドが `state_dict` / `step` / `join` / `subscribe` を呼ぶと、`_started()` を通ってからコアで `RuntimeError: the runtime has not started` になる。メッセージが状況と合わない。r2 からある挙動である（確信度: 低）
- N7 `src/mujou/_session.py:512-524`: `_shutdown_all` が例外を送出し直すと、`threading._shutdown` は残りの atexit と non-daemon のスレッドの join を飛ばす（"Exception ignored on threading shutdown"）。Ctrl-C で待ちを中断したセッションのノードのスレッドは、インタプリタの終了処理の間も走りうる。r2 の nit 5 から一歩進んではいるが、送出し直すより、全部を止めてから例外をログに出すだけのほうが安全かもしれない。再現はしていない（確信度: 低）
- N8 `src/mujou/_core.pyi:191-197`: 「The hooks other than ... are looked up once」とあるが、`state_dict` は `_capture_state` の中で毎回 `self.state_dict()` として引かれる。そのため、起動の後にインスタンス属性で置き換えても効く。害は無く、docs の「インスタンス属性で置き換えても、同期は漏れない」とは合っている（確信度: 低）
- N9 CPU のノードでも、ステップごとに `_after_update`（Python の関数）を 1 回余計に呼ぶ。測ってはいないが、更新の計算に比べれば小さいはずである（確信度: 低）

## 検証結果

- just run: pass（pre-commit の全 hook が通過、cargo test が通過、pytest は 287 passed / 2 skipped（CUDA）、pyright は 0 errors、coverage は `_node.py` 82%・`_session.py` 94%（CUDA の経路と `_shutdown_all` の分岐）。実行の後の `git status` は clean）
- 繰り返し実行（`tests/mujou`、slow を含む、`--no-cov`）: free-threaded 3.14t で 20 回、`PYTHON_GIL=1`（`sys._is_gil_enabled()` が True）で 20 回を 2 プロセス並行で回し、すべて 266 passed / 2 skipped だった。flaky は見られない
- `</content>` などの混入: WP-D 全体の変更ファイルを grep して無し
- 追加の確認（scratchpad の `r3/`。worktree は変更していない）
  - 起動中の `write` がブロックし、`on_start` に届かない（S1。free-threaded と GIL ありの両方）
  - Rust の `Methods::of` / `attribute`: `state_dict` は `_capture_state` を引き、エラーは `state_dict` の名前になる。引く前（`start` の途中で失敗したノード）も `call_method1(attribute(hook))` で `_capture_state` を通る。例外のときの取得（`finish` の `s.invoke(i, "state_dict", ...)`）も同じ `PyNode::state_dict` を通る。`_after_update` は `on_update` が成功したときだけ、同じスレッドで呼ばれ、失敗すると `_after_update` の名前で全体を終了する
  - CUDA の順序（取得）: ノードのストリームの下で、同期 → `state_dict` → 同期をしてから、制御スレッドの既定のストリームで deepcopy し、`build` から戻る前に同期する。RAW と WAR はともに満たす。例外のときの取得は、全ノードが `on_exception` を終えて止まった後に行い、`on_stop` の前に同期が終わる。ノードの中ではデバイス全体の同期を使っていない（`torch.cuda.synchronize` は消えた）
  - `_run_lock` の待ち合い: ノードの hook からホストのメソッドを呼ぶと、lock の前に `_session()` の `in_hook()` で `RuntimeError` になる。例外のときの取得のコールバック（`_capture_on_exception`）は lock を取らないので、起動中の例外でも待ち合わない。待ち合いになるのは S1 の形だけである
