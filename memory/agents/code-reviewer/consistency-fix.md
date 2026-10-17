# 整合性の修正（動作に関わる差分）レビュー

対象: `git diff HEAD -- crates src tests`（branch chore/2026-10-17/consistency、未コミット）と specs/execution-model.md の「保持」の段落（82 行・235 行）

## verdict: approve

must-fix なし。例外時の取得経路の除去で、例外で終了したときの流れ（`fail` → Stop → 全ノード `on_exception` → `on_stop` → `mark_finished` の `release_all` → `NetworkError`）は変わっていない。保持は Startup と Capture だけになり、それぞれ解く経路は次のとおりで、漏れは見つからなかった。

- Startup: 起動が成功すれば `startup()` で解き、起動中の例外なら `end()` → `finish()` → `mark_finished` の `release_all` で解く
- Capture: 成功・PauseTimeout・build の panic では `resume_after_pause` で解き、取得中の例外なら `end()` → `release_all` で解く

`pause_timeout` の検証は `start()` の中で build より前にあるので、構築の印は付かない。ノード間の通信経路は変わっておらず、局所性・非同期性への影響もない。

## must-fix

なし

## should-fix

1. `tests/mujou/_core/test_native_runtime.py:138-145`（`assert_ended_by_exception`）
   - 問題: `on_exception` と `on_stop` の間に `state_dict` が入るのを今も許している（docstring も「the state capture on the exception」のまま）。除去した機能のための許容で、今は何もこの許容に当たらない。このままだと、`on_exception` の後に `state_dict` を呼ぶ退行が入っても通ってしまう
   - 根拠: 許容を外した厳密版を pytest plugin で差し替えて `test_native_runtime.py` を走らせると、49 passed だった（差し替えが効いていることは 21 件で確認した）
   - 確信度: 高 / 深刻度: 低

2. `tests/mujou/_core/test_native_runtime.py:821-837`（`test_an_error_around_a_capture_stops_every_node`）
   - 問題: 取得中の例外の後でタイマーの保持が解けていることを確かめていない（`assert not virtual_timer.is_paused()` が無い）。specs/execution-model.md:235 は「状態の取得中に例外が起きたときは、その保持を終了処理の最後に解く」と書いている。例外で終了した後に `not is_paused()` を確かめていたのは、削除した Python と Rust のテストだけだった（保持の種類は Failure）。起動中の例外は `:205` で確かめている。Rust のコアには、例外で終了した後の保持の解除を確かめるテストが無い
   - 根拠: 実装の動作自体は正しい。scratchpad のスクリプトで、`on_pause` / `state_dict` / `on_start` で例外を出したとき、free-threaded と GIL ありの両方で `is_paused() == False` になり、同じタイマーで動く別の実行もその後 `advance` / `step` できた
   - 確信度: 中 / 深刻度: 低（検証の穴で、バグではない）

3. `crates/mujou-core/src/time.rs:252-253`（`Timer::hold` の doc comment）
   - 問題: 「(a runtime pausing to take the state, or stopping on an exception)」が残っている。例外で保持を取る経路は今回取り除いた
   - 根拠: `FreezeReason::Failure` が消えた。保持は Startup と Capture だけになった
   - 確信度: 高 / 深刻度: 低

4. `tests/mujou/_core/test_run.py:316-325`（`test_rejects_a_pause_timeout_that_is_not_finite_and_positive`）
   - 問題: コメントは「Nothing was built」と言っているが、この確認は単独の Network（`Chain`）を単独で走らせ直すだけである。単独の `build` は冪等で、断られても取り消されないので、検証が build の後（例えば `Session.start`）にあってもこのテストは通る。つまり「構築の印が付かない」ことは固定できていない。specs/network.md:255 によれば、印が残った単独の Network はほかの部品とまとめて走らせられない。そこで、拒んだ後に `mujou.testing.run(net, other)` のようにまとめて走らせれば区別できる
   - 根拠: 実装は正しい。scratchpad のスクリプトで、Network 単独・Node 単独・両方をまとめた 3 通りについて、0 / -1 / nan / inf のどれでも `_tree_of(...)` が None、`net._graph` が None のままだった。その後、まとめて走らせることもできた
   - 確信度: 中 / 深刻度: 低

## nit

- `crates/mujou-core/src/runtime.rs:563-564`: 文を削ったあとの折り返しが「... did not see. The」で切れている。確信度 高
- `pause_timeout` を検証しているのは Python の `start()` だけで、bindings の `duration()` とコアは 0 を受け付ける（private なので実害はない）。また 1e-12 のような正の極小値は `start()` を通るが、0 ns に丸まり、`capture` がすぐに `PauseTimeoutError` になる。仕様の「有限の正」は満たしている。確信度 低
- `tests/mujou/test_time.py:38-51`（`TestPublicNames`）: `vars(mujou.time)` の公開名を完全一致で比べているので、`mujou.time` に `_` の付かない import を 1 つ足しただけで落ちる。意図した固定ならそれでよい。確信度 低

## 検証結果

- just run: pass（依頼元の報告。511 passed）
- cargo test --workspace: pass（28 + 26 + 37）
- pytest `test_native_runtime.py` / `test_run.py` / `test_time.py`: free-threaded と `-X gil=1` で各 3 回、すべて 148 passed, 7 skipped
- pytest 全体（slow を除く）を `-X gil=1` で 1 回: 498 passed, 8 skipped（`sys._is_gil_enabled()` が True であることを確認した）
- 残った参照の grep（`ExceptionCapture` / `on_exception_capture` / `exception_state` / `capture_on_failure` / `FreezeReason::Failure`）: src・crates・tests・examples・docs・specs に 0 件
- 成果物の汚染（`</content>` など）: 0 件
- specs/execution-model.md の「保持」の段落: 82 行（保持は起動と状態の取得の間だけ）、235 行（例外は時間を止めない。起動中や取得中の例外では終了処理の最後に解く）とも実装に合う
