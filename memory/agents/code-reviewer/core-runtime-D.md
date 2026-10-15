# core-runtime WP-D（Port の結び付け・mujou.Node・ホスト入出力・mujou.run・docs）レビュー

対象: worktree `.claude/worktrees/run-network`、`git diff 81d29b3..HEAD`（745a40f まで）。WP-B のレビュー修正（スレッド状態の保持・停止の保持数）は未 merge の前提で見た。

## verdict: request-changes

## must-fix

### M1. `mujou.Node` を継承して `__init__` を書くと、pyright strict で必ず `reportInconsistentConstructor` になる

- 対象: `src/mujou/_core.pyi:196`（`def __new__(cls, *args: object, **kwargs: object) -> Self`）、`src/mujou/_node.py:69`（自分だけ ignore している）
- 問題: ignore は `mujou.Node.__init__` にしか効かない。利用者のサブクラスが `__init__` を定義するたびに、strict ではエラーになる。これから `src/mujou/pc/`（strict の対象）に置く PC-ALM のノードもすべて該当する
- 根拠: scratchpad で `class N(Node): def __init__(self) -> None: super().__init__(seed=0)` を `# pyright: strict` で検査すると `クラス "N" の__new__と__init__のシグネチャの不一致 (reportInconsistentConstructor)` が出た。`*args: object, **kwargs: object` を `*args: Any, **kwargs: Any` にした基底では、同じ形のサブクラスでもエラーが出ないことを確かめた。stub をそう直せば `_node.py:69` の ignore も要らなくなる
- 確信度: 高

### M2. 走っているネットワークへの参照を手放すと、非同期に止まり、終了時の atexit が待たない（`on_stop` が終わらないままプロセスが終わる）

- 対象: `src/mujou/_session.py:404`（`_RUNNING = weakref.WeakSet[Session]()`）、`src/mujou/_session.py:407-411`、`docs/network.md` の「インタプリタの終了時に、実行中のネットワークを atexit で終了させる」
- 問題: `_RUNNING` は弱参照なので、ノードからネットワークへの循環参照が無ければ（普通はこの形）、Network → Session → `_core.Runtime` が参照カウントで回収される。Rust の `Drop` は終了の job を送るだけで待たないので、atexit の `_shutdown_all` からはそのネットワークがもう見えない。関数の中で `net = mujou.run(...)` として、`shutdown()` を呼ばずに戻ってすぐスクリプトを終える、というよくある形で `on_stop` が完了しない。docs が約束する「終了時に止める」は、循環参照があるときにしか成り立たない。ロボットのアクチュエータを止める `on_stop` が走らないこともありうる
- 根拠（再現）: `on_stop` で 0.3 秒眠ってからファイルを書くノードを用意した。`main()` の中で `mujou.run` して参照を捨てると、ファイルは書かれない（rc=0）。参照を返して保持した場合は書かれる。別の確認では、`del net` の後 0.3 秒で `on_stop` が呼ばれていた（参照を捨てると止まる）。`tests/mujou/conftest.py:39` の docstring「A running network is not collected by the garbage collector」とも食い違う
- 推奨: 実行中の Session を強参照の集合で持ち、正常終了・`shutdown` の完了で外す。そうすれば、走っているネットワークは暗黙に回収されず、atexit が必ず止めて `on_stop` を待つ。循環参照の有無で振る舞いが変わらなくなり、conftest の docstring と docs の記述も正しくなる。計画書（WP-D 5）は `weakref.WeakSet` を指定しているので、orchestrator の裁定が要る
- 確信度: 高

### M3. CUDA のノードの状態を、別のスレッドの別ストリームから同期なしで読み書きしている（取得と復元）

- 対象: `src/mujou/_session.py:342-358`（`_tree` の `copy.deepcopy(state)` は制御スレッドで動く）、`src/mujou/_session.py:338`（`load_state_dict` の `copy.deepcopy` はホストのスレッドで動き、結果をノードのスレッドが使う）、`src/mujou/_node.py:104`
- 問題: ノードの演算は `torch.cuda.Stream(...)`（PyTorch のストリームプールから取るので `cudaStreamNonBlocking`）に積まれる。制御スレッドで CUDA の Tensor を deepcopy すると、clone はそのスレッドの既定ストリームに積まれる。非ブロッキングのストリームとは暗黙の同期が無いので、ノードが最後の `send`（`.cpu()` で自分のストリームを同期する）の後にその場で更新したパラメータは、書き込みが終わる前に読まれうる。復元の向きも同じで、ホストの既定ストリームで作った複製を、ノードのストリームが同期なしで読む。計画書の「想定リスク」は「CUDA のノードは各ステップの終わりに自分のストリームを同期している」を前提にしているが、その同期は実装されていない
- 根拠: 計画書の「想定リスク・トレードオフ」の最後から 2 番目の項目。`_node.py` にも `_session.py` にもステップごとの同期が無い。CUDA の無い環境なので実機では再現していない
- 推奨（例）: `_tree` の deepcopy の前と `load_state_dict` の deepcopy の後に、使っている CUDA デバイスで `torch.cuda.synchronize(device)` を呼ぶ（全ノードが止まっている間なので、遅くなるのは許容できる）。または停止の区切りで、ノードのスレッドが自分のストリームを同期する
- 確信度: 中

## should-fix

### S1. non-daemon のスレッドで `subscribe` を回したまま `shutdown` せずに main が終わると、プロセスが終了しない

- 対象: `src/mujou/_session.py:407`（`atexit` での停止）
- 問題: CPython は終了時に、non-daemon スレッドの join（`threading._shutdown`）を atexit より先に行う。購読のスレッドはネットワークの終了を待ち、ネットワークは atexit でしか終わらないので、互いに待ってプロセスが止まる
- 根拠（再現）: `threading.Thread(target=lambda: [_ for _ in net.subscribe(...)]).start()` の後に main を抜けると、5 秒で kill するまで終わらなかった
- 推奨: `threading._register_atexit`（スレッドの join より前に呼ばれる。private な API）で止めるか、`subscribe` の docstring と docs に「終了前に `shutdown()` を呼ぶ。購読するスレッドは daemon にする」と書く
- 確信度: 高

### S2. `Network.write` の docstring は「run の前に書いた値は load_state_dict より優先」と読めるが、実際は `load_state_dict` の前に書いた値は上書きされる

- 対象: `src/mujou/_network.py:297-298`（"Written before ``run``, they are the initial values; they take precedence over load_state_dict."）
- 問題: `InputSlot.restore` は保留中の書き込みを捨てる。優先されるのは `load_state_dict` の後に書いた値だけである（`docs/network.md` は正しく「`load_state_dict()` の後に書いた値が優先」と書いている）
- 根拠（再現）: `write(7.0)` → `load_state_dict(5.0 の木)` → run → step で、ノードが読んだ値は 5.0 だった
- 確信度: 高

### S3. 実時間の E2E テストが OS のスケジューリングに依存する（flaky の要因）

- 対象: `tests/mujou/test_network.py` の `TestEndToEnd.test_approaches_the_equilibrium_in_real_time`
- 問題: 実時計で 0.3 秒眠った後、b の出力が `atol=1e-6` で u に一致することを求めている。b が収束後に送っているかは、2 本のノードのスレッドがその間に十分スケジュールされたかで決まる。CI や GIL ありのビルドで coverage 付き、といった負荷の下では落ちうる。数値の正しさは手動モードのテスト（Gauss–Seidel / Jacobi のビット一致）が決定的に押さえているので、このテストが加える情報は「自由実行でも動く」だけである
- 根拠: skill `testing-strategy` の「たまに落ちるテストを作らない」
- 推奨: `@pytest.mark.slow` にして判定を緩める（ノードが更新したことと、誤差が初期誤差より小さいことだけを見る）か、削る
- 確信度: 中

### S4. 実装者の質問 3（recv が呼び手のスレッドを照合していない）について: 照合を足すべき

- 対象: `src/mujou/_session.py:84-97`（`Binding.recv`）
- 問題と意見: 足す方がよい。理由は次の 3 つ
  1. 局所性（CLAUDE.md の不変の原則、計画書の「ノードが読めるのは自分の Port に結び付いたスロットだけ」）を、send と同じ仕組みで実行時に守れる。今は、ほかのノードのスレッドから他人の Port を `recv` すると、隣でないノードの入力を読めてしまう（ホストのスレッドから `net.n.inp.recv()` が値を返すことを確かめた）
  2. `recv` は所有ノードの `device` へコピーするので、ほかのノード（別のデバイスや別のストリーム）が呼ぶと、デバイスもストリームも食い違った Tensor を受け取る
  3. 費用は `threading.get_ident()` と比較が 1 回ずつで、Tensor のコピーに比べて無視できる
- 照合の注意点: `state_dict` は制御スレッドで呼ばれるので、ノードの `state_dict()` の中で `recv` するとエラーになる。スロットは木に別に入るので、それで困ることは無い。docstring にそう書けば足りる。ホストが入力側を観測したい用途は `read`（送る側の外向きスロット）で足りる
- 確信度: 中

### S5. 復元しても `mujou.Node` の乱数生成器の状態は戻らない。docs にもその注意が無い

- 対象: `src/mujou/_node.py:105-110`、`docs/torch.md` の乱数生成器の段落
- 問題: `_on_thread_start` は、復元のときも `seed` から生成器を作り直す。乱数を使うノードは、状態の木に `generator.get_state()` を自分で入れないと、復元後の乱数列が中断せずに回した場合と変わり、「時刻 T から続ける」が乱数については成り立たない。決定的なテストの前提にも響く
- 推奨: フレームワークが生成器の状態を木に含めないなら、`Node.generator` の docstring と `docs/torch.md` に「再開後も同じ乱数列にしたければ、`state_dict` に `self.generator.get_state()` を入れ、`load_state_dict` で `set_state` する」と書く
- 確信度: 中

### S6. `Network.write` の引数の型では、変数に組んだ dict を渡すと pyright strict がエラーにする

- 対象: `src/mujou/_network.py:288-291`（`values: Mapping[Port[Any, Any, Any], ...]`）
- 問題: `Mapping` のキーは不変なので、`values = {net.prev: x}; net.write(values)` は `dict[PrevPort[Never, Tensor, Single], Tensor]` が割り当てられないというエラーになる（dict リテラルを直接渡すときだけ、双方向の推論で通る）。種類の違う Port を混ぜた dict も同じである
- 根拠（再現）: strict で `reportArgumentType` になる。`def write[P: Port[Any, Any, Any]](self, values: Mapping[P, torch.Tensor | Mapping[str, torch.Tensor]])` にすると、変数・混在・リテラルのどれも 0 errors になることを確かめた
- 確信度: 高

### S7. free-threaded のビルドで、例外のときの取得のコールバックが `Runtime.last_update_ns()` で失敗しうる（起動中の例外に限る）

- 対象: `src/mujou/_session.py:379`、`crates/mujou-python/src/runtime.rs` の `start`（`core::Runtime::start` の後に `self.inner.set(runtime)`）
- 問題: ホストのスレッドが `inner.set` する前に、ノードのスレッドが `on_start` で投げ、制御スレッドが `finish` からコールバックを呼ぶと、`last_update_ns()` は「the runtime has not started」を投げる。コールバックの失敗はログに出るだけで、`exception_state_dict()` は `None` になる。GIL ありのビルドでは、`inner.set` が終わるまでノードが attach できないので起きない。free-threaded では窓は非常に狭いが、ある。また、起動中の例外では `last_update_ns` がすべて 0 のまま木に入るので、その木から復元すると、最初の Δt が T になる
- 推奨: B 側でコールバックに `last_update_ns` を渡すか、セッションが `last_update_ns` を読めなかったときの扱いを決める
- 確信度: 低（コードを読んで判断した。再現はしていない）

### S8. `tests/mujou/conftest.py` の `run_network` の docstring が実際と違う

- 対象: `tests/mujou/conftest.py:39`（"A running network is not collected by the garbage collector"）
- 問題: M2 のとおり、循環参照が無ければ回収されて止まる。M2 を強参照で直せば、この docstring は正しくなる
- 確信度: 高

## nit

- N1 `src/mujou/_network.py:381`: `shutdown` の docstring は「Does nothing before ``run``」だが、build の前は `RuntimeError` になる。Raises にも `RuntimeError` が無い（確信度: 高）
- N2 `src/mujou/_node.py:112`: 実行の終了後も `_thread_ident` が残る。OS のスレッド ID が再利用されると、後から作ったスレッドが送信の照合を通りうる（終了後なので害は小さい）（確信度: 低）
- N3 `src/mujou/_node.py:103-104`: CUDA のとき、`torch.cuda.set_device` を呼んでいない。`device="cuda:1"` のノードでも、hook の中のスレッドの現在のデバイスは 0 のままなので、`device="cuda"` と書いた演算は cuda:0 の既定ストリームに積まれる。プールのストリームは 1 デバイスにつき 32 本で使い回されるので、ノードが多いと複数のノードが 1 本のストリームを共有する（正しさには影響しないが、docs の「自分のストリーム」と違う）（確信度: 中）
- N4 `src/mujou/_session.py:262-282`: `run` の前に `subscribe` したイテレータは、`run` されるまで（されなければ永遠に）待つ。docstring に書かれていない。ノードの hook の中から `subscribe` / `read` を呼ぶことも防いでいない。`subscribe` を回すと、ノードが他のノードを待つことになる（確信度: 中）
- N5 `src/mujou/_session.py:189-195`: 2 つのホストのスレッドが同時に `run()` を呼ぶと、free-threaded では両方が `_runtime is None` を通り、同じノードで 2 つの Runtime が動きうる（確信度: 低）
- N6 `src/mujou/_session.py:340`: `load_state_dict` は `timer` のキーと `last_update_ns` の型を確かめない。誤りは `run` のときに `KeyError` / `TypeError` として出て、docstring の「ValueError: The tree does not match」と違う（確信度: 中）
- N7 `tests/mujou/test_network.py` の `test_a_restored_network_continues_the_same`: 手動モードでは全ノードの最後の更新が凍結時刻と等しいので、`last_update_ns` を戻さなくても通る。既定の順 `[a, b]` では `a.next/ch0` を戻さなくても通る。`load_state_dict` の deepcopy（同じ木から 2 つ復元しても共有しないこと）もテストが無い（確信度: 中）
- N8 `tests/mujou/test_network.py` の `TestLoadStateDict`（`test_a_refused_run_can_be_retried` など）: 実時計のまま木のタイマー状態を読み込むので、プロセスのグローバル時刻を巻き戻したまま後のテストへ渡す。`virtual_timer` を使う方が独立する（確信度: 中）
- N9 `tests/mujou/test_network.py` の `TestNodeException.test_ends_a_subscription`: b が 100 回の更新で投げるのが `subscribe()` の呼び出しより早いと、例外はイテレータではなく `subscribe()` 自身から出る。どちらでも通るので、このテストではイテレータの経路が保証されない（`test_stops_the_network_and_reaches_the_host` が決定的に押さえているので実害は無い）（確信度: 中）
- N10 `src/mujou/_session.py:364-387`: `_exception_capture` はモジュール関数として Session の private 属性に 3 か所 ignore を付けている。`weakref.WeakMethod` で Session のメソッドにすれば ignore が要らない（確信度: 高、好みの範囲）
- N11 `docs/execution-model.md` の「ほかのネットワークが走っている間は、状態の木から復元して `run()` できない（エラーになる）」: WP-B のレビュー修正（未 merge）に依存している。merge 後に実装と一致するか確かめる（確信度: 高）
- N12 Port を持たないノードは `build` の凍結の検査にかからないので、同じインスタンスを 2 つのネットワークに入れて両方を `run` できる。その場合、2 つのランタイムが同じノードの hook を 2 本のスレッドから呼ぶ（C の範囲だが、D の `run` で実際に起きるようになった）（確信度: 中）
- N13 `src/mujou/_session.py:352`: `torch.enable_grad()` の下で作った非葉の Tensor をノードの状態が持つと、`copy.deepcopy` が失敗し、`state_dict()` がその RuntimeError を送出する（ネットワークは動き続ける）。docs の「pickle できれば中身を問わない」より条件が狭い（確信度: 中）

## 検証結果

- just run: pass（pre-commit の全 hook が通過、cargo test がすべて通過、pytest は 259 passed / 1 skipped（CUDA）、pyright は 0 errors。実行後の `git status` は clean）
- `</content>` などの混入: diff の全ファイルを grep して無し
- 追加の再現（scratchpad。worktree は変更していない）
  - M1: 利用者のサブクラスで `reportInconsistentConstructor` が出る。`Any` の stub なら出ない
  - M2: 参照を手放したネットワークは、終了時に `on_stop` が完了しない
  - S1: subscribe するスレッドが残ると、プロセスが終わらない
  - S2: `load_state_dict` の前の write は上書きされる
  - S4: ホストのスレッドから `Port.recv()` が値を返す
  - S6: dict を変数に組むとエラーになり、ジェネリックなキーにすると通る
