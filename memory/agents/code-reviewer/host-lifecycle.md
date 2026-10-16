# host-lifecycle（ホスト側の振る舞いをユーザーの決定に合わせる）レビュー

対象: `git diff feature/2026-10-16/cuda-tests..HEAD`（8 commits）。要件: decisions-2026-10-16 の「状態」「終了」「subscribe」「TensorSpec」。`mujou._core` への移動と run がハンドルを返す形への作り直しが後で入る前提で、それで消える点は指摘していない。

## verdict: request-changes

must-fix は 1 件（終了直前の例外と state_dict の競合）。修正は小さい。ほかの観点（`main_thread().join()` + atexit による終了時の停止、subscribe の最初の値、TensorSpec、docs と実装の一致）は要件どおりで、free-threaded と `-X gil=1` で安定していた。

## must-fix

1. **`src/mujou/_session.py:390-392, 407-408` / 例外で終了した直後の `state_dict()` が木を返すことがある / 確信度: 高（起こる確率は低い）**
   - 問題: `self.check()` で例外が無いことを確かめた後に `runtime.is_finished` を読む。この 2 つの間にノードが例外を投げて終了まで進むと、`is_finished` が True になり、`_ended_tree` が例外で終了したネットワークの木を返す。`except RuntimeError` の側は `is_finished` を見た後に `self.check()` を呼んでいるが、最初の分岐には同じ確認が無い
   - 根拠: 要件「例外で終了した後は `NetworkError`。例外時の状態は保存しない」（decisions「状態」）。Rust の `fail()` は failure を記録してから `mark_finished` に進むので、`is_finished` を見た後に `check()` すれば取りこぼさない
   - 再現: `Session.check` を「元の check を呼ぶ → ノードに例外を投げさせる → 0.3 秒待つ」に差し替えると（ホストのスレッドが check の直後に止められた状況を作る）、`net.state_dict()` が `{'node': {'ok': 1}}` を返した（scratchpad の `hlprobe/race.py`）

## should-fix

1. **`src/mujou/_session.py:264-269` / 有限の `timeout` で `shutdown` を呼ぶたびに daemon スレッドが 1 本増える / 確信度: 高**
   - 問題: `shutdown(timeout)` は呼ぶたびに `mujou-shutdown` スレッドを起動する。どのスレッドもネットワークが終わるまで `runtime.shutdown()` で待ち続ける。`while True: try: net.shutdown(timeout=0.1) except TimeoutError: ...` のように時間切れを繰り返す書き方は自然だが、その数だけスレッドがたまる。終了した後に呼んでも 1 本起動する（すぐ終わる）
   - 根拠: 実測で、`shutdown(timeout=0.005)` を 40 回時間切れにすると `mujou-shutdown` が 40 本あった（`hlprobe/poll.py`）。Session ごとに要求用のスレッドを 1 本にする（起動済みなら join だけ）か、`is_finished` なら起動しないようにすれば足りる。Rust に「待たずに要求する」API を足せば消える問題だが、それまでの間も防げる
2. **`src/mujou/_session.py:418-426` と `517-520` / 終了後の木で、タイマーの時刻とスロットの値を別々の時点に読んでいる / 確信度: 中**
   - 問題: `_ended_tree` がスロットを読んで `latest` を決めた後に、`_tree` がスロットをもう一度読む。終了した後でもホストの `write` は通るので、その間に別のスレッドが書くと、木の中でスロットの時刻がタイマーの時刻より後になる。実装者ノートが守るとした不変条件（スロットの時刻はタイマーより後にならない）が崩れる
   - 根拠: `write` は `check()` だけで、終了したかは見ない（`_session.py:310-331`）。スロットを 1 回だけ読み、その値から時刻と木の両方を作れば防げる
   - 関連（docs）: `docs/execution-model.md` の「終了後の時間は系のダイナミクスの外」「復元した系の最初の Δt に終了後の時間が混ざらない」は、終了した後にホストが書いたときには成り立たない（書いた時刻がタイマーの時刻になる）。実装者ノートはこれを承知で入れているので、docs に例外として 1 文足すか、終了後の `write` をどう扱うかを残る質問に足す
3. **`tests/mujou/test_network.py:967-972`（`TestShutdown::test_rejects_a_negative_timeout`）/ 「何も要求していない」を確かめられていない / 確信度: 高**
   - 問題: 最後の `net.read(net.next)  # still running` は、ネットワークが正常に終了した後でも成功する（`read` は終了後も読める）。誤った `timeout` で終了を要求していても、このテストは通る
   - 根拠: `Session.read` は `check()` と `outbox.read()` だけ。`net.join(timeout=0)` が `TimeoutError` になることや、`_step` が通ることで確かめれば、docstring の "nothing was asked" をテストで確かめられる（`TestJoin` の同名テストの後半はそうなっている）
4. **`tests/mujou/test_network.py`（`TestShutdown`）/ 有限の `timeout` で、例外で終了したネットワークを `shutdown` するテストが無い / 確信度: 中**
   - 問題: `_request_stop` は `NetworkError` を握りつぶし、呼んだ側の `runtime.join` が送出する設計である。この経路（`shutdown(timeout=1.0)` が `NetworkError` を送出し、`__cause__` がノードの例外になること）を確かめるテストが無い。`TestNodeException` の一覧は既定の `math.inf` の経路しか通らない
   - 根拠: docstring の `Raises: NetworkError` は両方の経路に掛かる。カバレッジでも `_session.py:399-406`（実行中の取得が終了と競合した分岐）が未実行（こちらは決定的に作りにくいので nit 相当）

## nit

1. **`docs/network.md:282` と `src/mujou/_network.py:146-149`（`Network` の docstring）/ 終了時の停止の記述が実装より強い / 確信度: 中** — 「non-daemon のスレッドを join するより前に終了させる」と書いているが、実装では監視スレッドが停止を始めるのが join の前で、停止は join と並行に進む（終わるのを待つのは atexit 側）。「join の前に停止を始め、atexit で終わるのを待つ」と書けば正確になる
2. **`tests/mujou/test_network.py:712` / `frozen_at == max(times)` は実装の式をそのまま書き直している / 確信度: 中** — 隣の `test_after_a_normal_end_is_the_state_at_the_end`（実行中の木と、終了から 100 秒後の木が一致する）がオラクルとして働くので、こちらは自由実行でも壊れないことの確認と割り切るか、削る
3. **`src/mujou/_session.py:418` / 終了後の取得でノードの `state_dict()` が投げた例外は、そのままホストへ伝わる / 確信度: 中** — 実行中なら `NetworkError` になりネットワークが終わるが、終了後は生の例外が出る（ログにも出ない）。振る舞いは妥当なので、`Network.state_dict` の docstring に 1 行あればよい
4. **`src/mujou/_session.py:591-615` / CPython の実装の詳細に頼っている / 確信度: 低** — `threading._shutdown` が `_set_done()` を non-daemon の join より先に呼ぶ順序は CPython の文書に無い。`TestInterpreterExit::test_ends_a_subscribing_thread` が退行を検出できる（実装者が確認済み）ので、コメントに「CPython の実装に依存し、上のテストで守る」と 1 行あると、後で読む人に親切
5. **`src/mujou/_session.py:604-615` / `_watching` は fork した子で True のまま残り、子には監視スレッドが無い / 確信度: 低** — 子でネットワークを走らせて終わると、non-daemon の join の後の atexit でしか止まらない。fork 後の子でネットワークを走らせる用途は今は無いので、記録だけ
6. **`tests/mujou/test_network.py`（`TestStateDict`）/ 終了後の `state_dict()` の中から Network のメソッドを呼ぶと `RuntimeError`、のテストが無い / 確信度: 低** — 実装者ノートは「テストあり」としているが、見つかったのは `recv` の側（`test_after_a_normal_end_cannot_receive_in_the_node_state_dict`）だけだった

## 確認して問題が無かった点（判断の材料として）

- 終了後の取得をホストのスレッドで行う安全性: Rust の `finished` はノードのスレッドを join した後（`on_stop` が終わった後）に立つので、ノードと並行に状態に触れることは無い。`_capture_state` が `_hooks.inside` を立てるので、ホストのメソッドは `RuntimeError` になる。`thread_ident` は `_settle` で外れているので `recv` も `RuntimeError` になる。CUDA の同期は制御スレッドの経路と同じ（`_capture_state` が自分のストリームを同期し、`_tree` がホストの現在のストリームを同期する）。CUDA のテストは GPU が無いため未実行
- `capture` が "the runtime has finished" の `RuntimeError` を返すのは、Rust の `done`（`mark_finished` の後）か制御スレッドが抜けた後だけなので、`is_finished` は必ず True になっている。偽の `RuntimeError` は出ない
- `timeout` に 1e300 や `sys.float_info.max` を渡しても `join` と `shutdown` は正しく待つ（Rust の `checked_add` が None になり無期限になる）
- 終了時の停止を、free-threaded と `-X gil=1` で、次の場合に確かめた: `SystemExit(3)`（終了コードは 3 のまま）、主スレッドの未処理例外、ほかのスレッドで run して subscribe、`on_stop` の例外、non-daemon のスレッドの `join()`、`shutdown(timeout)` の時間切れの直後の終了、ネットワーク 2 つ、atexit の中での最初の run。すべて 1 秒前後で終わり、`on_stop` が走った。うち 3 つのスクリプトを各モード 12 回ずつ（計 72 回）回し、ハングは 0 回

## 検証結果

- just run: pass（format → cargo test + pytest 304 passed / 13 skipped → pyright 0 errors）。作業ツリーに差分は出なかった
- 繰り返し: `tests/` を free-threaded（3.14t）で 5 回、`PYTHON_GIL=1 python -X gil=1` で 5 回（`sys._is_gil_enabled()` が True であることを確認）。10 回とも 304 passed / 13 skipped
- 成果物汚染: 変更したファイルに `</content>` などは無い
