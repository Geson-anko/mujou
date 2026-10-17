# core-runtime WP-D 再監査（差し戻しへの修正: 84a9225・021100e・054d485）

対象: worktree `.claude/worktrees/run-network`、ブランチ `feature/2026-10-16/run-network`、`git diff feature/2026-10-16/network-structure..HEAD`。前回の監査は `memory/agents/algorithm-reviewer/core-runtime-D.md`、対応記録は worktree の `memory/agents/plan-implementer/core-runtime-D.md`。

## verdict

**request-changes**（確信度: 中）。前回の must-fix（復元待ちの write の時刻）は直っている。再現スクリプトで確かめ、復元後も「全スロットの時刻 ≤ T」が保たれた。recv の照合、ホストのメソッドの hook 内での拒否、復元 E2E のオラクルも十分である。オラクルはミューテーションで Channel スロットと `last_update_ns` の復元漏れを検出できた。

残る must-fix は CUDA の経路の 1 件だけである。状態の取得で、制御スレッドが既定のストリームで行うコピーと、ノードのストリームとの順序付けが前後とも欠けている。CPU の経路には影響しない。CUDA の無い環境なので実測はしておらず、PyTorch のストリームの意味論から推論した。CUDA を「未実測」のまま先送りすると判断する場合でも、最低限 docs/torch.md の「そこで読む値は書き終わっている」の記述を直す必要がある。

観点ごとの判定:

- **must-fix（復元待ちの write の時刻）**: 直った。`Session.write` は `_runtime is None and _timer_state is not None` のとき、全件を検査してから各スロットを `restore(tensors, T)` で書く。`InputSlot.restore` は group の `write_lock` を取り、保留中の次の値を捨てる。`run` 前は読み手がいないので、一斉切り替えは要らない。前回の再現（T ≈ 1 ms で木を取り、100 s 進めてから load → write → run）では、スロットの時刻が 1000417 ns（= T）、受けた時点のノードの時刻が 2000417 ns、取り直した木の全スロットの時刻が T 以下になった。テスト `test_after_load_state_dict_takes_precedence` は、プロセスの時刻を 100 s 離したうえで「受けた時刻 == 木の T」を確かめており、旧実装なら落ちる
- **recv の照合（局所性）**: 実行時に守られる。`Binding._check_owner` は send と同じく OS のスレッド ID で照合する。前回の再現（B が A の `prev` をクロージャで `recv`）は `RuntimeError` になった。ホストのメソッドは `_session()` の `in_hook()` で拒否する。この判定は、ノードのスレッドでは `_on_thread_start` が、制御スレッドでは `state_dict` を包んだ wrapper が立てる印で行う。B の `on_thread`（スレッド状態の保持）に依存するが、`TestHostMethodsFromANode` が `on_update` から 6 つのメソッドで確かめている
- **CUDA の同期**: ステップごとの同期は docs と一致する（`on_update` を包み、自分のストリームだけを `synchronize` する。ランタイムは hook を `_on_thread_start`・`on_start`・`load_state_dict` の後に引く。`crates/mujou-python/src/runtime.rs` の `start` で確認）。ノードの中ではデバイス全体の同期（`torch.cuda.synchronize`）を使っていない。それを使うのはホストの `load_state_dict` だけである。ただし取得の前後の同期に穴がある（must-fix 1）
- **復元 E2E のオラクル**: 検出できる。(a, b) と (b, a) の両方で回し、復元の直前に a だけを進めて `last_update_ns` を食い違わせている。そのうえで復元直後の `state_dict()` を木と突き合わせる。ミューテーションの結果（照合メモ 2）、Channel スロットをゼロにすると両方のパラメータで落ち、`last_update_ns` を T にそろえても、復元を落としても落ちた。(b, a) の順では、b が最初のステップで復元された `a.next/ch0` を読むので、ダイナミクスの側でも検出できる
- **docs と実装の一致**: おおむね一致する。torch.md と execution-model.md の取得の同期の記述は、実装と同じく穴を含む（must-fix 1）。ほかは nit
- **学習則の焼き付け・グローバルなバリア・モード**: 新たに持ち込んだものは無い。`_RUNNING` と atexit の停止は終了処理だけで、ダイナミクスには入らない

## must-fix

1. **CUDA の状態の取得で、制御スレッドのコピーとノードのストリームの順序付けが前後とも欠けている**（確信度: 中。CUDA の無い環境なので未実測。推論の根拠は照合メモ 3）
    - **後（WAR）**: `_tree` は既定のストリーム（制御スレッドの current）で `copy.deepcopy`（= `clone` のカーネルを積むだけで、完了を待たない）を行い、そのまま戻る。コアは `build` の直後に `resume_after_pause` するので（`crates/mujou-core/src/runtime.rs` の `capture`）、再開したノードは自分のストリームで重みをその場で更新する。PyTorch のプールのストリームは `cudaStreamNonBlocking` で作られ、レガシーの既定のストリームとは暗黙に同期しない。そのため、コピーのカーネルが終わる前に更新が走り、木に更新途中の重みが入り得る。`state_dict` が参照を返す（PyTorch の慣習どおりの）ノードで常に起き得る。例外のときの取得でも、その後にノードのスレッドで走る `on_exception` / `on_stop` との間に同じ穴がある
    - **前（RAW）**: コアは、全ノードの `state_dict` hook を呼んでから `build`（`_tree`）を呼ぶ。`_tree` の先頭の `synchronize(node)` は、ノードの `state_dict` が制御スレッドで行うコピー（`.clone()` や、pickle のための `.cpu()`）より後に来る。ステップの終わりの同期は `on_update` だけを包むので、`on_start`・ノードの `load_state_dict`・`on_pause`・`on_resume` が積んだ仕事は、`state_dict` の時点で終わっている保証が無い。たとえば手動モードや、`run()` の直後の `state_dict()` では、`on_start` での GPU 上の初期化が未完了のまま読まれ得る
    - 該当する docs: torch.md「ノードのストリームはステップの終わりに同期済みなので、そこで読む値は書き終わっている」は、`on_update` 以外の hook では成り立たない。execution-model.md「制御スレッドでコピーする前に各ノードのストリームを同期する」は、ノードの `state_dict` 自身のコピーより後に同期していることを書いていない
    - 案（自分のストリームだけを同期する、という docs/torch.md の規約のまま直せる）: `_mark_state_dict` の wrapper で、`state_dict()` を `with torch.cuda.stream(self._stream):` の中で呼ぶ。`_tree` では、ノードごとの `deepcopy` を同じくそのノードのストリームの下で行い、`build` から戻る前に各ノードのストリームを `synchronize` する。こうすれば、制御スレッドの仕事がノードの過去の仕事の後ろに並び（RAW）、再開前に完了する（WAR）。もっと小さく直すなら、`_tree` の最後で各 CUDA デバイスの `torch.cuda.current_stream(d).synchronize()` を呼び（WAR）、wrapper の先頭で `synchronize(self)` を呼ぶ（RAW）。どちらでも、docs の 2 か所を実装に合わせて直す
    - 先送りするなら、docs/torch.md の「状態」の注記に、この穴（取得の前後の順序付け）を明記する

## should-fix

なし。

## nit

1. **`run` と `write` を別のホストのスレッドから同時に呼ぶと、窓が残る**（確信度: 中。実害は小さい）。`write` は `self._runtime is None` を lock なしで読む。`run` が `_runtime` を立ててから、`runtime.start` の `freeze_for_startup` がタイマーを T に戻すまでの間に、`write` が group の経路に入ることがある。するとプロセスの時刻（T より後のこともある）が付き、「全スロットの時刻 ≤ T」が破れる。逆に、`restore` の経路に入った直後にノードが走り出すと、`InputSlot.restore` の前提（「ノードが走っていない間だけ」）を外れる。docs は複数のホストのスレッドからの `write` を認めている。直すなら、`write` の分岐の判定と `restore` を `_run_lock` の下で行い、`run` は `runtime.start` が戻るまで `_run_lock` を持つ。または docs に「`run` と `write` を同時に呼ばない」と書く
2. `load_state_dict` の `_synchronize_cuda` はデバイス全体を待つ（`torch.cuda.synchronize(index)`）。ホストの操作なのでノードは止めないが、`load_state_dict` は別のネットワークが走っている間も呼べるので、そのネットワークの GPU の仕事まで待つ。コピーはホストのスレッドの current stream に積まれるので、`torch.cuda.current_stream(index).synchronize()` で足りる。docs/torch.md の「同期は自分のストリームに対してだけ行う」とも揃う
3. docs/network.md の「終了したネットワークのノードはスレッドを忘れる」は、実装ではホストが終了を見たとき（`check`・`shutdown`・`join`、次の `run`）に限られる。例外で終わり、ホストが触っていないネットワークでは ID が残る（対応記録の「残る懸念」には書かれている）。docs も「終了を見たとき」と書くとよい
4. `exception_state_dict` は `_session()` を通らないので、hook の中から呼べる。読み取りだけで待たないので害は無いが、docs/network.md の「これらはホストのためのメソッドで」の列挙と揃っていない（列挙に含まれていないので矛盾ではない）。記録だけ
5. 制御スレッドの印（`state_dict` の中からホストのメソッドを呼ぶと `RuntimeError`）を確かめるテストが無い。`test_cannot_receive_in_the_node_state_dict` が見ているのは Port の所有者の照合だけである。`TestHostMethodsFromANode` と同じ形で、`state_dict` から `net.write` などを呼ぶ例を 1 つ足すとよい

## 前回の指摘の扱い

| 前回 | 状態 |
| --- | --- |
| must-fix 1（復元待ちの write の時刻） | 解消（再現・テストで確認） |
| should-fix 1（recv の照合・hook からのホストのメソッド） | 解消 |
| should-fix 2（`state_dict` が制御スレッド・CUDA の同期） | 一部解消。ステップごとの同期・device の設定・復元後の同期・docs は入った。取得の前後の順序付けが残る（今回の must-fix 1） |
| should-fix 3（Channel スロットの復元のオラクル） | 解消（ミューテーションで確認） |
| nit 1〜4 | 解消 |
| nit 5（起動時のタイマー凍結） | 記録のまま（ユーザー確認待ち） |

## 検証結果

- `just run`（worktree）: 終了コード 0。cargo test ok、pytest 279 passed / 1 skipped（CUDA）、coverage 97%、pyright 0 errors。実行の前後で `git status` は空のまま
- 前回の再現スクリプトを修正後に再実行（scratchpad の `probe_d2.py`）: 他人の Port の `recv` は `RuntimeError`。復元後のスロットの時刻は T、全スロットの時刻 ≤ 取り直した木の T
- 復元 E2E のミューテーション（scratchpad の `mut/mut_plugin.py`。`Session.load_state_dict` を包んで木を書き換える）: `channel`（Channel スロットをゼロ・時刻 0 に）、`last`（`last_update_ns` を全部 T に）、`drop_last`（`last_update_ns` を渡さない）のどれでも、`test_a_restored_network_continues_the_same[False/True]` が 2 件とも落ちた

## 照合メモ

1. **時刻の不変条件**。復元した系は時刻 T から始まる。スロットの時刻は、木の値（取得時に T 以下）、load 後の write（ちょうど T）、run 後の write（タイマー ≥ T）のどれかになる。そのため全スロットの時刻 ≤ 現在時刻が保たれる。load の前の write は load の `restore` で木の値に置き換わる。スロットごとの時刻の単調性も、木の値 ≤ T ≤ run 後の時刻の順で保たれる。残る穴は、別スレッドとの競合の窓だけである（nit 1）
2. **ミューテーションの効き方**。木では、a の最後の更新が T、b の最後の更新が T − H。`last` / `drop_last` では、復元直後の `again["last_update_ns"]` が `{'a': T, 'b': T}` になり、直接比較で落ちる（ダイナミクスでも、b の最初の Δt が 2H になる）。`channel` では、直接比較 `torch.equal(again["slots"][name][0], value)` で落ちる。(b, a) の順では、b が最初のステップで `a.next/ch0` を読むので、直接比較が無くても終状態が変わる
3. **CUDA のストリームの意味論**（推論の根拠）。`torch.cuda.Stream()` はデバイスごとのプールから取り、プールのストリームは `cudaStreamNonBlocking` で作られる。そのため、レガシーの既定のストリームとは互いに待たない。`clone` / `deepcopy` / `.cpu()` は呼んだスレッドの current stream に積まれる。`.cpu()` は、そのストリームだけを同期してから戻る。制御スレッドの current stream は既定のストリームである（ノードの `_on_thread_start` の `set_stream` はスレッドローカル）。コアの `capture` の順序は、全ノードの区切りでの停止 → 各ノードの `state_dict`（制御スレッド）→ `build`（制御スレッド）→ `resume_after_pause`。したがって、ノードのストリームに積まれた仕事と制御スレッドのコピーの間に順序を与えるのは、`_tree` の先頭の `synchronize(node)` だけで、後ろ側には何も無い
