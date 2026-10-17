# core-runtime WP-D 監査（Port の結び付け・mujou.Node・ホスト入出力・mujou.run・docs）

対象: worktree `.claude/worktrees/run-network`、ブランチ `feature/2026-10-16/run-network`、`git diff 81d29b3..HEAD`（`76c58f2`・`35bf08c`・`fad3871`・`745a40f`）。計画書 `memory/agents/implementation-planner/core-runtime.md` の WP-D、判断ログ `memory/agents/plan-implementer/core-runtime-D.md`。

## verdict

**request-changes**（確信度: 高）。must-fix は 1 件だけで、小さく直せる。`load_state_dict` の後、`run` の前に `write` すると、そのスロットの時刻が復元後の時間軸に乗らない。その結果、状態の木の不変条件「全スロットの時刻 ≤ T」が破れる（再現済み）。ほかに局所性・非同期性・継続学習を壊す変更は無い。学習則も焼き付けていない。

観点ごとの判定:

- **Port とスロットの結び付け**: 正しい。各 Port は、自分の Outlet（外向きと Channel ごとのスロット）と、相手側 Outlet の「相手の Port での Channel の番号 k」のスロットだけに結び付く。ホストが書けるのは、未接続で相手が 1 つ、かつ受ける spec を持つ Port の入力スロットだけ。ノードが所有するスロットにホストが書き手として加わる経路は無い（例外は `run` 前の `load_state_dict` による復元だけで、これは許された例外）
- **send の照合**: OS のスレッド ID で照合しており、書き手 1 人の不変条件を守る。`_on_thread_start` は `on_start` より先に呼ばれる（`PyNode::start`）ので、`on_start` から送れる
- **recv の照合**: していない。別のノードのスレッドから、他人の Port を `recv` できることを確かめた（should-fix 1）
- **勾配がノードをまたがないか**: またがない。送る側は `detach().cpu()` の後、コアがプールのバッファへコピーする（`write_values` → `copy_into`）。受ける側は `from_dlpack(...).to(device, copy=True)` で毎回新しい Tensor を作る。`requires_grad=True` の葉を送るテストと、受けた値をその場で書き換えるテストがある。ノードのスレッドの grad 無効は `on_update` でも効いている（実測: `torch.is_grad_enabled()` が False）
- **ホストがノードを待たせないか**: 待たせない。`write`（`InputGroup.write`）、`read`、`subscribe`（`wait_newer`）はどれも GIL を手放して待ち、ノードの側には待ちを作らない。待つ側はホストだけである
- **状態の木の時刻のそろい**: `capture` の `build` は、全ノードが区切りで止まり、タイマーが凍結している間に制御スレッドで呼ばれる。`_tree` はその中でスロットとタイマーを読むので、自由実行ではそろう。ただし復元と `write` を組み合わせると崩れる（must-fix 1）
- **スレッドごとの torch 設定**: `set_grad_enabled`、`cuda.set_stream`、専用の `Generator` はどれもスレッドローカルか、ノードが自分で持つもので、プロセス全体の設定は変えない（ホストの grad mode が True のままであることはテスト済み）。ただし `state_dict()` の hook はノードのスレッドではなく制御スレッドで呼ばれるため、これらの設定が効かない。CUDA ではストリームの同期が抜ける（should-fix 2）
- **E2E のオラクル**: Gauss–Seidel と Jacobi の参照は理論的に正しい（照合メモ 2）。復元の一致は決定性のオラクルとして正しいが、Channel スロットの復元は検証できていない（should-fix 3）
- **学習則の焼き付け・モード・エポック**: 無い。`run` は 1 回だけだが、続きは新しいインスタンスに `load_state_dict` すればよく、エポックを前提にしていない。`_step` はテスト専用の private API で、自由実行と同じループを区切りで止めるだけなので、「同期は非同期の特殊なスケジュール」に従う
- **docs と実装の一致**: おおむね一致する。torch.md の「各ステップの終わりに自分のストリームを同期する」だけは実装に無い（should-fix 2）

## must-fix

1. **`load_state_dict` の後、`run` の前に `write` した値の時刻が、復元後の時間軸に乗らない**（確信度: 高。再現済み、照合メモ 1）
    - `InputGroup.write` は書いた時点のプロセスのタイマーで時刻を付ける（`crates/mujou-core/src/slot.rs` の `let time_ns = self.timer.time_ns();`）。一方、タイマーを木の時刻 T に戻すのは `run` のとき。そのため docs が勧める「`load_state_dict` → `write` → `run`」の順にすると、入力スロットの時刻は復元前のプロセス時刻になる
    - 再現: 時刻 ≈ 1 ms で木を取り、仮想時計を 100 s 進めてから、新しいネットワークで `load_state_dict` → `write` → `run` → `_step` した。ノードの `recv_with_time` は 100.001 s を返し、ノードの現在時刻 0.002 s より未来の時刻になった。続けて `state_dict()` を取ると、`slots["a.prev/in"]` の時刻 100001000375 ns が `timer.time_ns` 2000375 ns を超え、docs/execution-model.md の「全スロットの値の時刻も T 以下」が破れた
    - 別プロセスで復元する場合（プロセスの時刻 ≈ 0 < T）も、書いた値の時刻は復元した過去より前になる。不変条件は破れないが、スロットごとの時刻の単調性が崩れる。メッセージの古さ（`now − t`）を使うノードは、どちらの場合も誤った値を見る
    - 直し方の案: `Session` が復元待ちの `_timer_state` を持っている間、`write` はスロットを `restore(tensors, timer_state["time_ns"])` で書く（`run` 前は読み手がいないので、一斉切り替えは要らない。ただし全件を検査してから書く）。または `run` の直前に、復元後に書いた入力の時刻を T に付け直す。テストは `test_after_load_state_dict_takes_precedence` に「時刻 == 木の T」の確認を足せばよい

## should-fix

1. **`recv`（と `Network.read` / `write` / `subscribe`）をノードのスレッドから呼べてしまい、局所性の穴になる**（確信度: 中。再現済み）
    - `Binding.recv` はスレッドを照合しない。ノード B の `on_update` から、B と接続されていないノード A の `prev` を `recv` すると値が返った（照合メモ 1）。A の Port の参照さえ持っていれば（クロージャなど。`build` は属性で他のノードを持つことしか検出しない）、B は隣でないノードへの入力を読める
    - 同じく、ノードの hook から `Network.read` を呼べば任意のノードの出力を読め、`write` を呼べばホストの入力スロットに書ける。待つメソッド（`state_dict` / `shutdown` / `join`）だけは hook から呼ぶと `RuntimeError` になる
    - Python の属性アクセスによる違反まではフレームワークで防げない。それでも send を照合している以上、同じ費用（`get_ident` の比較 1 回）で recv も照合するのが筋である。照合すれば、CUDA の H2D コピーが所有ノードのストリームに載ることも保証できる（他のスレッドで呼ぶと、そのスレッドの現在のストリームに載る）
    - 案: `Binding.recv` に `_sending` と同じ照合を入れる（`run` 前の呼び出しを許すかどうかは決める）。`write` / `read` / `subscribe` は、呼んだスレッドがこのネットワークのノードのスレッドなら `RuntimeError` にする（コアが待つメソッドで使っている hook 中の判定を流用できるはず）。照合しないと決めるなら、その理由を docs/network.md の「Port の読み書き」に書く
2. **`state_dict()` の hook が制御スレッドで呼ばれるため、mujou.Node のスレッド設定が効かず、CUDA ではストリームの同期が抜ける**（確信度: 中。CUDA の無い環境なので未実測）
    - コアの `capture` は `s.call(i, "state_dict", ...)` を制御スレッドで呼び、`_tree` も同じスレッドで `copy.deepcopy` する。このスレッドは grad が有効で、現在のストリームは既定のストリームである
    - torch.md は「隣へ送る値は重みを更新する前に計算する」と勧める。この順だと、`send` の `.cpu()` が同期した後に重みの in-place 更新がノードのストリームに積まれ、区切りで止まった時点でもそのカーネルは終わっていないことがある。PyTorch のプールのストリームは non-blocking なので、制御スレッドの既定のストリームで行う clone はそれを待たない。その結果、木に更新途中の重みが入り得る。`load_state_dict` も同じで、ホストのスレッドで deepcopy した CUDA の Tensor を、ノードのストリームが同期なしに使う
    - torch.md の「各ステップの終わりに自分のストリームを同期するので……」は、フレームワークがそうするように読めるが、実装に同期は無い。docs/nodes.md の `state_dict()` の行も、どのスレッドで呼ばれるかを書いていない
    - 案: mujou.Node が `on_update` の後（少なくとも停止の区切りに入る前）に自分のストリームを同期する、フレームワーク内部の hook を持つ（`_on_thread_start` と同じ形）。復元では `load_state_dict` の前に同期する。最低限、docs に「`state_dict` は制御スレッドで呼ばれる。CUDA のノードは区切りまでに自分のストリームを同期しておく」と書く
3. **復元のテストが Channel スロットの復元を検証していない**（確信度: 高）
    - `test_a_restored_network_continues_the_same` は既定の順（a → b）で進める。b は毎ステップ、a がその直前に書いた値を読むので、`a.next/ch0` が復元されなくても（ゼロでも）結果は同じになる。`b.next/out` の比較も、復元後に b が書き直した値を見ている。このテストで効いているのは、ノードの状態・`a.prev/in`・タイマーの復元だけである
    - 案: 同じテストを順序 (b, a) でも回す（Jacobi では b が最初のステップで復元された `a.next/ch0` を読む）。または復元直後に `read` と各スロットの値・時刻を木と比べる

## nit

1. `test_is_taken_at_one_time` は時刻の上限だけを見ている。Leaky は更新の最後に送るので、区切りで止まっていれば `tree["slots"]["b.next/out"][0] == tree["nodes"]["b"]["x"]` が成り立つ。これを足すと、更新途中の状態（書きかけの値）を読んだ場合にも気づける
2. `Network.read` の docstring「before the first one it is zeros」は、`load_state_dict` の後では復元した値になる。docs/network.md の表も同じ
3. `thread_ident` による照合は、終了したネットワークのスレッド ID が別のスレッドに再利用されると通ってしまう。書き込み先は止まったネットワークのスロットなので実害は無い。気になるなら、終了時に `_thread_ident` を消す
4. `_on_thread_start` は `torch.cuda.set_device` を呼ばない。`device="cuda:1"` のノードが `torch.zeros(..., device="cuda")` と書くと、cuda:0 に確保される。docs に「確保には `self.device` を使う」と一言あるとよい
5. 1 つのネットワークの起動（新規の起動でも）がプロセスのタイマーを凍結し、同時に走る別のネットワークを `on_start` の間ずっと止める。docs の「複数のネットワーク」に書かれており、ユーザー確認待ちに挙がっているので、ここでは記録だけにする。ロボティクスの制御ループを動かしたまま別のネットワークを起動する使い方では、実時間の応答が止まる。`timer_state` の無い起動では、凍結の代わりに各ノードの起点を `on_start` の後の時刻に置くこともできる（コアの WP-B の範囲）

## 検証結果

- `just run`（worktree）: 終了コード 0。pytest 259 passed / 1 skipped（CUDA）、coverage 99%、pyright 0 errors。実行の前後で `git status` は空のまま（format による書き換えは無かった）
- 再現スクリプト: scratchpad の `probe_d.py`（照合メモ 1）

## 照合メモ

1. **再現**（free-threaded 3.14t、仮想時計）。A → B の 2 ノードで、B はクロージャで A の `prev` を持ち、`on_update` で `recv` する
    - `grad in on_update: False`。grad の無効は C++ のスレッドローカルなので、hook ごとに Python のスレッド状態が作り直されても残る
    - `other-port recv from B thread: tensor([1., 1.])`。B のスレッドから A の入力が読めた
    - 木を T ≈ 1.000 ms で取り、`advance(100.0)` の後に新しいネットワークで `load_state_dict` → `write` → `run` → `advance(0.001)` → `_step` を行った。`recv_with_time` は 100001000375 ns、ノードの現在時刻は 2000375 ns。続けて `state_dict()` を取ると、`slots["a.prev/in"]` の時刻 100001000375 > `timer.time_ns` 2000375 になった
2. **E2E のオラクル**（`test_manual_steps_match_the_discrete_algorithm`）。Leaky の更新は $x \leftarrow x + \min(\Delta t/\tau, 1)(u - x)$。ステップごとに `write(u(k))` → `advance(H)` → `_step(order)` を行う
    - (a, b) の順: a は今のステップの $u_k$ を読んで $x_a^{k}$ を送る。b はそれを読む。つまり $x_b^{k} = E(x_b^{k-1}, x_a^{k})$ で、Gauss–Seidel。参照の `xa = sent_a = euler(xa, u(k)); xb = euler(xb, sent_a)` と一致する
    - (b, a) の順: b はスロットに残っている $x_a^{k-1}$（初回はゼロ）を読む。つまり $x_b^{k} = E(x_b^{k-1}, x_a^{k-1})$ で、一方向の鎖ではこれが Jacobi と同じになる。参照は `sent_a` を更新する前に `xb` を計算しており、一致する
    - Δt: 起動時の `last_update_ns` は起動時の時刻で、各ステップで `advance(H_NS)` するので、`(now_ns - last_ns)/1e9 = 1e6/1e9` は参照の `H = H_NS/1e9` とビット単位で同じ。演算の順（`x + c*(u - x)`）も同じなので、`torch.equal` での比較は妥当
    - 実時間のテスト（τ = 0.01、0.3 s = 30τ）: 2 段の鎖の誤差の包絡は $(1 + t/\tau)e^{-t/\tau} \approx 3 \times 10^{-12}$ で、atol 1e-6 は十分に余裕がある。Δt > τ のときは率を 1 で打ち切るので、x = u に一致し、発散しない
3. **送受信のコピー**: `PyOutlet.send_with` → `write_values` は DLPack で借りた Tensor を GIL を手放してプールのバッファへ `copy_from` する。`.cpu()` が CPU の Tensor に対して同じストレージを返しても、コアには参照が残らない（`test_copies_the_values` で確認済み）
