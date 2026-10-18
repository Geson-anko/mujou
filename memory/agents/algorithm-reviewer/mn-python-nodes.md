# algorithm-reviewer: M:N PR 2（Python のノードをワーカーに載せる）

対象: `git diff d1ca90f..HEAD`（feature/2026-10-18/mn-python-nodes、82183ce まで）。観点: 実行モデルとノード内 autograd の不変条件。更新式そのものは diff に無い。

## verdict

**approve（should-fix 付き）**。確信度: 高。

ノードをまたいで計算グラフがつながる経路、ノードが互いを待つ構造、Δt の歪み、ほかのノードのストリームやデバイス全体への待ちは見つからなかった。ただし「勾配モードは hook の前に毎回戻る」という文書の約束と、実装（スレッドが切り替わったときだけ戻す）が食い違っている。また、ワーカーを共有したことで、フレームワークが戻さないスレッドごとの torch の設定（autocast など）がほかのノードに持ち越されるようになった。どちらも実験で確かめた。

## must-fix

なし。

## should-fix

### S1. 勾配モードが戻るのは「切り替えのとき」だけなのに、文書は「毎回」と約束している（確信度: 高）

- 実装: コアは `LAST_NODE != this` のときだけ `enter` を呼ぶ（`crates/mujou-core/src/runtime/mod.rs:280-283`）。`_on_enter`（`src/mujou/_core/node.py:391`）の `torch.set_grad_enabled(False)` も、切り替えのときだけ走る。計画書（L111, L162）と `specs/torch.md`「切り替えごとの設定」も、切り替えのときだけと書いている
- 文書は「毎回」「各 hook の前」と書いている:
  - `docs/en/user-guide/torch.md:49`（"before the hooks of every node"）、`:93`（"back to the node's before each of its hooks"）
  - `docs/en/user-guide/writing-a-node.md:230`（"set back by Mujou before each hook"）
  - `docs/ja/user-guide/torch.md:49`、`:93`（「hook の前に毎回ノードの設定に戻す」）、`docs/ja/user-guide/writing-a-node.md:230`
  - `examples/torch_node.py:3`（"turns gradient recording off before every hook"）
  - `.claude/skills/locality-and-async/SKILL.md:47`、`memory/project_mn_scheduler.md:14`（「フレームワークが hook の前に戻す勾配モード」）。`specs/torch.md:83` と `specs/native-core.md:68` の「hook の間で保たれる／保つ」も同じ誤読を招く
- 実測（scratchpad の再現スクリプト、workers=1、ノード 1 つ）: hook の最後で `torch.set_grad_enabled(True)` のまま返すと、同じノードの次の `on_update` は `is_grad_enabled() == True` になった（3 回の更新で `[False, True, True]`）
- 影響: ノードをまたぐ漏れではない。ほかのノードに切り替われば戻るし、`recv` はコピー、`send` は detach + コピーなので、グラフはつながらない。ただし、戻し忘れたノードの次の hook で勾配の記録が有効かどうかが、スケジュール（同じワーカーに続けて載ったかどうか）で決まる。計算結果は普通変わらないが、メモリと速度がスケジュール依存になり、「勾配を切っている」前提で書いたノード（`requires_grad` のパラメータへの in-place 更新など）がスケジュール次第で `RuntimeError` になりうる。決定性の原則に反する
- 直し方（どちらか）:
  - (a) 勾配モードだけは hook の前に毎回切る。実測では `torch.set_grad_enabled(False)` が 0.13 µs、`Context.run` が 0.02 µs で、hook 1 回の費用に比べて無視できる。たとえば bindings の `call_as` で、`_on_enter` とは別の安い内部 hook を毎回呼ぶ。あるいは `_after_update` と同じく hook の後に戻す
  - (b) 実装はそのままにして、上の文書をすべて「スレッドがノードに切り替わるたびに戻す。hook の中で変えた勾配モードは hook を抜ける前に戻す（`with` に閉じる）。戻さないと、同じスレッドで続けて走る自分の次の hook に残る」に直す
  - 決定性を重く見るなら (a) を勧める

### S2. フレームワークが戻さないスレッドごとの torch の設定が、同じワーカーのほかのノードに漏れる（確信度: 事実は高、重さは中）

- 実測（workers=1 でノード 2 つ。片方が `torch.autocast("cpu").__enter__()` を閉じずに返す）: もう片方のノードの `on_update` が、毎回 `torch.is_autocast_enabled("cpu") == True` で始まった。専用スレッドだけだった PR 1 までは、こうした漏れは本人にしか届かなかった
- これは不変条件 1（局所性: 更新に使うのは自分の状態と、メッセージで受け取った値だけ）を実質的に破る。ほかのノードのやり残しが、メッセージを経ずにノードの数値計算（dtype）を変えるからである。原因は利用者の誤用（`with` を使わない）だが、新しい既定（ワーカー）がその誤用の被害をほかのノードにまで広げる
- 文書は「`with` に閉じよ」と書いているが、その理由を「次の hook に持ち越されない（ので頼るな）」としか書いておらず、ほかのノードに漏れることを書いていない（`src/mujou/_core/node.py:83-87`、`docs/*/user-guide/writing-a-node.md:227-231`、`docs/*/user-guide/torch.md:93`）。`node.py:86` の "does not carry over to the next hook" は事実と逆で、正しくは「持ち越されるかどうかが決まっていない」
- 直し方:
  - 最低限: skill `locality-and-async` の不変条件（1 か 4）と、ユーザー向け文書に「hook の中で変えたスレッドごとの設定は、hook を抜ける前に必ず戻す。戻さないと、同じワーカーで次に走るほかのノードに漏れる。フレームワークが戻すのは勾配モードと CUDA の device・stream だけ」と書く
  - 任意: `_on_enter` で autocast も切る（`torch.set_autocast_enabled(...)`）。あるいは debug 時だけ hook の後に `is_grad_enabled()` と `is_autocast_enabled()` を確かめて警告する。どちらも要求外の機能なので、入れるかどうかはユーザーの判断

### S3. skill `locality-and-async` の修正の正確さ（確信度: 高）

骨子（ワーカーが既定、ノードはワーカーの間を移る、hook は重ならず happens-before がある、ブロッキングするノードは専用スレッドへ、Port の所有は「今走らせているノード」で判定、`on_thread_start` は専用スレッドだけ）は、実装・specs と合っていて不変条件として正しい。直すべき点:

- `SKILL.md:47`「フレームワークが hook の前に戻す勾配モード…」は S1 のとおり不正確。「スレッドがノードに切り替わるたびに戻す」とする。あわせて S2 の「hook を抜ける前に戻す」を不変条件として足す
- `SKILL.md:47` の「内部の hook `_on_enter`（切り替えのたびに呼ぶ）に書く」は、利用者が `_on_enter` を override してよいと読める。ユーザー向け文書と docstring は override を禁じている。「（フレームワーク側の設定は）`_on_enter` に書く」と主語を限る
- `SKILL.md:43` の「既定は論理 CPU 数」は不正確。既定は `min(ワーカーに載るノード数, 論理 CPU 数)`（`specs/native-core.md`、`run.py` の docstring）

## nit

- N1（確信度: 中）`_on_enter` は `enter_node(self)` を最後に呼ぶ（`src/mujou/_core/node.py:391-400`）。途中（`torch.cuda.set_device` など）で失敗すると、コアは `exception` / `stop` を BestEffort で呼ぶ（`mod.rs:280-292`）ので、それらの hook は前のノードの `current_node()` と CUDA の stream のまま走る。Port の所有の判定は前のノードを所有者とみなす（`binding.py:105`）。自分の Port は拒まれるだけで、ほかのノードの Port への参照を持っていなければ実害はない。`enter_node(self)` を先頭に移すほうが安全
- N2（確信度: 中）CUDA のノードは `_after_update` で自分のストリームを同期する（`node.py:403-406`）。ワーカーの上ではその間ワーカーを占有するので、実質的にブロッキングする hook になり、ほかのノードの Δt を伸ばす。不変条件は破らない（待つのは自分のストリームだけ）が、`docs/*/user-guide/torch.md` の「ブロッキングする hook」の注意に、CUDA のノードの同期もそうだと一言あるとよい
- N3（確信度: 中）CUDA のノード A（cuda:0）の後に別デバイスのノード B（cuda:1）が同じワーカーで走ると、cuda:0 の current stream は A のストリームのまま残る。B が自分のデバイス以外を触らなければ無害（`_on_enter` は自分のデバイスの stream しか設定しない）。仕様上触らないので、記録だけ
- N4（確信度: 高）ユーザー向けの `contextvars` の説明（`docs/en/user-guide/writing-a-node.md:231` ほか）には、`state_dict` が文脈の外（制御スレッド・ホスト）で走ることが書かれていない。specs（`native-core.md:67`）と bindings の docstring には書いてある

## 確かめて問題がなかった点

- **ノードをまたぐ計算グラフ**: `recv` は `torch.from_dlpack(b).to(device, copy=True)`（`binding.py:33`）、`send` は `t.detach().to("cpu", copy=copy)` のうえでコアがコピーする（`spec.py:371`）。どちらも新しいテンソルなので、ワーカーを共有してもグラフはつながらない。`examples/torch_node.py:53-55` と `examples/online_learning.py:86-88` の `# mujou: local-autograd` は `with torch.enable_grad():` の行に付き、`autograd.grad` の対象は自ノードのパラメータだけ。ブロックを抜けると勾配モードは戻る
- **別のノードへの勾配モードの持ち越し**: 同じワーカーで別のノードに切り替わると、必ず `_on_enter` が走って勾配の記録が切れる（実測と `test_gradients_stay_off_in_every_hook_of_a_node_sharing_a_worker`）
- **Port の所有**: スレッド ID ではなく threading.local の `current_node()` で判定する。同じワーカーの上でも、ほかのノードの Port は拒まれる（`TestOwnerOnAWorker`）。制御スレッドとホストは `enter_node` を呼ばないので `None` のまま。lock は増えていない
- **非同期性・GIL**: ワーカーは hook を呼ぶ前に制御の lock を外す（`pool.rs` の `run_work` の `drop(c)`）。待つ間は detach している（`runtime.rs:318`）。lock を握ったまま Python を呼ぶ経路は見当たらない。GIL ありのビルドで GIL を取り合うスレッドは M 本から N 本に減る。GIL を待つのはノード同士の待ちではない。`contextvars.Context` はノードごとに空で作る。ノードの hook は重ならないので、`Context.run` が二重に入ることもない
- **Δt**: 時刻は制御の lock の下で 1 回読み（`pool.rs:145-152`）、その時刻を `dt` と次の起点（`record` の `last_ns = now_ns`）の両方に使う。`_on_enter` と GIL 待ちの時間はその時刻の後に入るので、次の Δt に実際の経過時間として含まれ、Δt の総和は失われない。起動中の `_on_enter` は凍結したタイマーの下で走るので、最初の Δt に入らない
- **CUDA**: ストリームは最初の `_on_enter` で 1 回だけ作り、切り替えのたびに同じものを `set_stream` する（`node.py:392-396`）。同期するのは自分のストリームだけ（`_synchronize`、`_capture_state`）。src に `torch.cuda.synchronize()` は無い。ホストの `_synchronize_current_streams` は今回の変更の前からあり、制御スレッドとホストの current stream を同期するだけ
- **更新手法の焼き付け**: diff の src・crates に PC-ALM 固有の量（λ、Wᵀε）は無い。ノード・スケジューラは更新手法を知らない

## 検証結果

- `just run`: 成功（exit 0）。pytest 586 passed / 20 skipped（CUDA のテストはこのマシンに CUDA が無いので skip。`TestCuda.test_nodes_sharing_a_worker_keep_their_own_device_and_stream` などは未検証）。pyright 0 errors、docs-build 成功。作業ツリーに変更は出なかった
- 再現スクリプト（`uv run python`、VirtualTimer、workers=1）の結果
  - 勾配モード、ノード 1 つ: 3 回の更新で `[False, True, True]` → S1
  - 勾配モード、ほかのノード: 3 回とも `False` → 漏れない
  - autocast、ほかのノード: 3 回とも `True` → S2
- 費用: `torch.set_grad_enabled(False)` は 0.13 µs、`Context.run(noop)` は 0.02 µs

## 照合メモ

- 計画書（L111, L162）の約束は「切り替えのときだけ `_on_enter`」。spec-test-author のテスト `test_a_worker_enters_a_node_only_when_it_switches_to_it` もそれを固定している。食い違っているのは利用者向けの文書・example・skill・memory の「毎回」という表現のほう
- 局所性の 3 条件（受け取った値は detach 済み、勾配の対象は自分の状態だけ、マーカーとコメント）は、ワーカーを共有しても保たれる。グラフがつながる経路はメッセージだけで、メッセージは常にコピーだから。スレッドごとの状態はグラフではなく「計算の設定」の経路で、S1・S2 はそこで起きている
