# host-lifecycle の監査（algorithm-reviewer）

対象: `git diff feature/2026-10-16/cuda-tests..HEAD`（ブランチ `feature/2026-10-16/host-lifecycle`）。基準は skill `locality-and-async` と `docs/execution-model.md`。

## verdict

**request-changes**（確信度: 中〜高）

既定の流れ（最後の更新・書き込みと同じ時刻に shutdown → state_dict → 復元 → 続行）は、止めずに続けた結果とビット単位で一致した。ただし、公開 API で許されている「終了後の write」をすると、終了後の時間が木のタイマーに入る。docstring と docs が約束している「終了後の時間は混ざらない」がこの操作で破れる。

## must-fix

1. **終了後にホストが write すると、終了後の時間が木のタイマーに入る**（確信度: 挙動は高、重大度は中〜高）
   - `Session.write` は正常終了の後も拒まない（`check()` は例外による終了しか見ない）。書いた値には書いた時点のグローバル時刻が付き、`_ended_tree` はそれを含めて最大値を取る
   - 再現: 手動モードで 6 ステップ → shutdown → 仮想時計を 100 秒進める → 同じ値を write → state_dict。木の `timer.time_ns` は 6 ms ではなく 100.006 s になった。復元後の最初の Δt は全ノードで約 100 s になり、止めずに続けた結果と最大 0.95 ずれた（`Leaky` は `min(dt/tau, 1)` で打ち切るのでこの程度で済む。打ち切らない陽的 Euler の更新則なら発散する）
   - Network の docstring（「the time that passed since」を除く）、`docs/execution-model.md`（「復元した系の最初の Δt に終了後の時間が混ざらない」）、テストのコメント（「the time after the end is not part of the run」）と食い違う。判断ログには「終了後にホストが write した値の時刻も含まれる」と書いてあるので意図した挙動だが、約束と両立しない
   - 修正案: 正常終了後の `write` を `RuntimeError` にするのが最も単純。ノードはもう読まないので、終了後の write に意味は無い。ストリームを流すホストのスレッドが shutdown と競合して書いてしまう、というよくある状況も、例外で気づける。除外する案（最大値の計算から外す）は、スロットの時刻が木のタイマーより後になる（未来の時刻のメッセージを復元する）ので避けたい

## should-fix

1. **木のタイマーを「最後の更新・書き込みの最も遅い時刻 M」にすると、終了の時刻ではなくなる**（確信度: 中。判断ログの残る質問 1 への回答）
   - 停止（状態の取得）では、要求した時刻 T_p でタイマーを凍結する。最後の更新から T_p までの空き時間は木に入り、復元後の最初の Δt に含まれる。正常終了の木は空き時間 [M, T_req] を捨てるので、「shutdown の直前に state_dict を取って復元」した場合と結果が変わる
   - 再現: 最後の write の後、仮想時計を 0.3 s 進めてから shutdown した。実行中に取った木のタイマーは 0.306 s、終了後の木のタイマーは 0.006 s で、復元して続けると止めずに続けた結果から最大 0.29 ずれた
   - M は on_stop が send するかどうかでも変わる。更新が 1 度も無いノードを仮想時刻 0.5 s で shutdown したとき、on_stop が send すれば M = 0.5、send しなければ M = 0 になる。ホストが決めた終了の時刻ではなく、ノードの実装の副作用で決まっている
   - `docs/execution-model.md` の「終了した時点の時刻を入れれば」は不正確。M は終了の時刻ではなく、最後に観測できた出来事の時刻である
   - 自由実行での差は、最も速いノードの更新間隔以下なので小さい。手動モード、遅いノード、ホストが入力を見せる時間（`mujou.time.sleep(0.1)` → shutdown）を意味として持つ使い方では差が出る
   - 修正案（Rust に触れずにできる）: `Session.shutdown` で最初に終了を要求する直前に `_core.timer.time_ns()` を記録して T_req とし、木のタイマーを `max(T_req, 各 last_update_ns, 終了までに書かれたスロットの時刻)` にする。正常終了の経路は Session.shutdown だけなので（join は他スレッドの shutdown を待つだけ、終了時は `_shutdown_all` → `shutdown`）、必ず記録できる。こうすると「shutdown は T_req での停止と同じ」になり、停止の不変条件（docs の「停止要求のフラグを立ててすぐにグローバルタイマーを凍結」）と同じ見方で説明できる。on_stop の send は T_req より後になり得るが、max で取ればスロットの時刻が木のタイマーを越えない
   - どちらにするかは意味の選択なので、ユーザーに確認してほしい。M のままにするなら、docs の「終了した時点の時刻」を「最後の更新・書き込みの時刻（その後の空き時間は捨てる）」と書き直す

2. **終了後の木に on_stop の変更が入ることの約束が無い。例外を投げた on_stop の後でも、正常な終了として木を返す**（確信度: 中〜高）
   - 実行中の木は on_stop の前、終了後の木は on_stop の後の状態である。on_stop で資源を解放する（docs/nodes.md の用途どおり）ノードが、テンソルを手放したり CPU に移したり 0 にしたりすると、終了後の木は学習の状態ではなくなる
   - on_stop の例外はログに出るだけ（コアの `invoke`、`call` ではない）で、終了は正常扱いになる。再現: on_stop が `w.mul_(0)` の後に raise すると、終了後の `state_dict()` は何も送出せず、`w = 0` の木を返した
   - on_exception の約束を外したこと自体は妥当である（下の「確かめたこと」）。ただ、同じ種類の約束が、今度は on_stop について必要になった。次のどちらかを `Node` の docstring と `docs/nodes.md` に書く
     - (a) 「on_stop は `state_dict()` が返す状態を変えない」と約束する（外した約束を on_stop に移す）
     - (b) 「終了後の木は on_stop の後の状態」と明記する
   - on_stop が例外を投げた後の `state_dict()` を、送出にするか警告にするか、黙って返すかも決めて書く

3. **「終了後の木から復元して続けた結果 = 止めずに続けた結果」を直接確かめるテストが無い**（確信度: 高）
   - `test_after_a_normal_end_is_the_state_at_the_end` は、最後の write と shutdown の間に仮想時計を進めていないので、実行中の木と一致する。そのため、空き時間をどう扱うかを固定していない
   - 推移律（終了後の木 = 実行中の木、実行中の木から復元すると一致）で間接的には言えるし、手元の probe でも一致した。それでも理論のオラクルとして直接持つべきである
   - 足すテスト:
     - (i) `TestEndToEnd::test_a_restored_network_continues_the_same` の終了後版（`b_first` でパラメータ化）
     - (ii) 最後の更新から shutdown までに時間を進めたときの木のタイマー（should-fix 1 で決めた意味を固定する）
     - (iii) 終了後の write の扱い（must-fix 1）

## nit

1. `test_after_a_normal_end_of_a_free_running_network` の `frozen_at == max(times)` は、実装の式をそのまま書き直しただけのオラクルである。should-fix 1 で意味が決まったら、意味から導いた期待値（例: T_req 以上、かつ全スロットの時刻以上）にする（確信度: 中）
2. 終了後の木の `scale` は、`state_dict()` を呼んだ時点の `time_scale()` である。終了後にホストがスケールを変えると、それが木に入る。実行中の取得は停止した時点の値なので、終了の時点の値（T_req と一緒に記録する）にそろえるとよい（確信度: 低）

## 確かめたこと（問題なし）

- **subscribe の最初の値はノードを待たせない。** `subscribe()` の中で、ホストのスレッドが `outbox.read()` を呼ぶ。これはスロットの最新値の `Arc<Message>` を lock で読むだけで、書き手を止めない。メッセージは不変で、バッファは drop したときにプールへ戻るので、最初の `next()` まで保持しても書き手と競合しない。以降は `wait_newer(last)` で待つので、同じ値を 2 度返さない。最初の yield の前にも `check()` するので、例外による終了は最初の値より先に伝わる
- **on_exception の約束を外したことは妥当。** その約束を使っていたのは例外時の取得だけで、Python から `on_exception_capture` を渡さなくなった。例外で終了した後の `state_dict()` は `check()` で `NetworkError` になる。取得の途中で例外が起きても `NetworkError` になる。on_exception の後に状態を読む経路は残っていない。コアの `on_exception_capture` と `exception_state()` は残っているが、使われていない（後で掃除する予定であることが docs に書いてある）
- **終了後の取得は状態の所有を壊さない。** コアの `is_finished` は、全ノードのスレッドを join した後（`mark_finished`）に立つ。そのため、ホストのスレッドが `_capture_state()` を呼ぶ時点では、on_stop を含めてノードの書き手はもういない。Port の `recv` / `send` は所有ノードのスレッドでないので `RuntimeError` になる（テストあり）
- **学習則の焼き付けは無い。** 変更はホスト側のライフサイクル、タイマーの時刻、TensorSpec の `device` だけで、PC-ALM などの量は入っていない。ノードをまたぐ autograd も無い
- **新しいグローバルバリアは無い。** 有限の `timeout` で shutdown するときの daemon スレッドと、終了時の `mujou-exit` スレッドはホスト側の仕組みで、ノードの更新には関わらない
- **待っている間に終了したときの切り替え。** 実行中の `state_dict()` を待つ間に正常に終了すると、`capture` は `RuntimeError` を返す。そのとき `is_finished` を確かめて、終了後の取得に切り替える。例外による終了なら `check()` が `NetworkError` を送出する。どちらも正しい
- `env -u VIRTUAL_ENV just run`: format は差分なし。304 passed / 13 skipped。pyright は 0 errors

## 照合メモ

- コアの `node_loop`: `dt = (now_ns - last_ns)/1e9`。`last_update_ns[i]` は更新を始めた時刻（Δt を測った時刻）
- 復元: `freeze_for_startup` がグローバルタイマーを `timer_state` に戻し、`last_update_ns` を各ノードに入れる。復元後の最初の Δt_i は `tree.timer + (再開後の経過) - last_i` である
  - 停止の木（T_p で凍結）: Δt_i = (T_p - last_i) + 経過。止めずに続けた場合と同じ
  - 終了後の木（M）: Δt_i = (M - last_i) + 経過。止めずに続けた場合より (T_req - M) だけ短い
  - 終了後の write があると M > T_end になり、Δt_i が終了後の時間だけ長くなる
- probe（`scratchpad/algrev/test_ended_restore.py`、Chain、手動モード、k=5、m=7）の結果

  | 条件 | 実行中の木のタイマー | 終了後の木のタイマー | 一致 | 最大のずれ |
  | --- | --- | --- | --- | --- |
  | 空き時間 0、終了後の write なし | 6 ms | 6 ms | する | 0 |
  | 空き時間 0、終了後の write あり | 6 ms | 100.006 s | しない | 0.95 |
  | 空き時間 0.3 s、終了後の write なし | 306 ms | 6 ms | しない | 0.29 |

- probe（`probe_onstop.py`）: on_stop が `w.mul_(0)` → send → raise の順に動くと、終了後の木は `w = 0`、`timer = 0.5`（on_stop の send の時刻）で、例外は送出されなかった
