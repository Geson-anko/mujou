# node-state（ノードの状態の自動収集と on_thread_start の公開）の監査

対象: worktree `.claude/worktrees/node-state`、ブランチ `feature/2026-10-16/node-state`（`c4cb9ad`）、`942f5ca..HEAD` の差分。基準は skill `locality-and-async`、`specs/nodes.md`・`specs/torch.md`・`specs/execution-model.md`、判断ログ `memory/agents/plan-implementer/node-state.md`。

## verdict

**approve**（確信度: 高。CUDA の経路だけは手元に GPU が無く、コードの読みでの判断なので中）。must-fix は無い。should-fix が 3 件ある。1 件は公開した `on_thread_start` を経由して局所性のガードが黙って外れる経路で、もう 1 件は仕様の「どの hook で代入してもよい」が復元では成り立たないこと。

- 時刻 T の一貫した状態: 成り立つ。既定の `state_dict` は記憶域を共有する `detach()` と `nn.Module.state_dict()`（これも keep_vars=False の detach）を返す。`_capture_state` は自分のストリームを同期してから呼び、その後もう一度同期する。続いて `Session._tree` が、ノードが止まっている間に（`runtime.capture` のコールバックの中で）ノードごとに `copy.deepcopy` し、制御スレッドの現在のストリームを同期してから再開させる。detach もモジュールの state_dict もストリームに仕事を積まないので、既存の同期の順序のままで足りる。1 回の deepcopy の memo は storage を共有するので、同じノードの中で別名の付いたテンソル（ビュー、モジュールのパラメータを属性にも置いたもの）の別名関係は木の中でも保たれる
- 「その場で更新する」規約: 整合する。`torch.no_grad()` の下の `copy_` と strict な `nn.Module.load_state_dict`（これも中で `param.copy_`）なので、オブジェクトは差し替わらない。`requires_grad` の leaf は leaf のまま。shape をコピーの前に確かめるので、`copy_` のブロードキャストで黙って通ることも無い
- 復元して続けた結果と止めずに続けた結果の一致（`TestRestore`）: オラクルは理論的に正しい。停止中はタイマーが凍結し、復元はタイマーと `last_update_ns` を T_k に戻すので、どちらの経路でも k+1 歩目以降の Δt は 0.001 で同じになる。CPU の演算は決定的なので、ビット一致を期待してよい。更新は Δt に比例し（`100*dt*g`、`dt*(1-x)`）、計数の `steps` が k+m になることも確かめている。そのため、Δt がずれても、木が生きたテンソルと記憶域を共有していても（k+2m になる）、復元が効かなくても（fill=5 の値が残る）落ちる
- ノード内 BP の勾配: ノードをまたがない。木は detach 済みで、復元は no_grad の `copy_`。受け取る値は `recv` のコピーで、送る値は `send` が detach する。今回の差分にノードをまたいでテンソルを参照させる経路は無い。`autograd.grad` の行には `# mujou: local-autograd` が付いていて、ノードの中で閉じている
- 学習則の焼き付け: 無い。集めるのは型（Tensor / nn.Module）だけで決まる。λ のような PC-ALM 固有の量の名前も扱いも、基盤に入っていない
- 連続時間: ノードの状態に時刻を入れず、時刻はフレームワークが `last_update_ns` と `timer` で持つという分担が守られている

## must-fix

なし。

## should-fix

1. **`on_thread_start` の override が `super()` を呼ばないと、ネットワークのメソッドを禁じるガード（`_hooks.inside`）と所有スレッドの記録が黙って外れる**（確信度: 高。実際に確かめた）
    - `_hooks.inside = True` と `_thread_ident` は、torch の設定と同じく、override できる `on_thread_start` の中で立てている。`super()` を呼ばない override を書いて `on_start` で調べると、`in_hook()` は False、`torch.is_grad_enabled()` は True で、`holder["net"].read(self.out)` も通った（照合メモ 2）。ノードの hook から `Network.read` / `write` が使えると、ノードが Port を通らずに他のノードの出力を読める。これは局所性の抜け道になる。同じ状態で `Network.state_dict` を呼ぶと、自分自身の停止を待ってしまい、停止のタイムアウトまで固まる
    - 気づけるのは、Port の `recv` / `send` を最初に呼んだときだけである（`_check_owner` の RuntimeError）。そのメッセージは「only the node that owns the port ... from its own hooks」で、原因が `super()` の呼び忘れだとは分からない。docstring（`on_thread_start` と Node のクラスの docstring）も、呼び忘れの帰結として「勾配の記録・generator・Port」しか挙げておらず、ガードが外れること、CUDA のデバイスとストリームが設定されず `_after_update` / `_capture_state` の同期も素通りになることを書いていない
    - 案: ユーザーの決定（`on_thread_start` を公開する）は保ったまま、フレームワークが外せない分（少なくとも `_thread_ident` と `_hooks.inside`。できればデバイス・ストリーム・勾配・generator も）を Rust から呼ぶ内部の `_on_thread_start` に戻し、その後で公開の `on_thread_start`（既定は何もしない）を呼ぶ。こうすれば `super()` の規約自体が要らなくなる。設計の変更になるので、メインからユーザーに確認する。最小の対応は docstring と仕様に帰結を全部書き、`_check_owner` のメッセージで `super().on_thread_start()` の呼び忘れにも触れること

2. **`on_start` の後に初めて代入した Tensor は、取得はできても復元できない。仕様の「どの hook で代入してもよい」と食い違う**（確信度: 高。実際に確かめた）
    - 既定の `load_state_dict` は `on_start` の直後に、そのとき登録されている名前の集合が木のキーと等しいことを要求する。`on_update` の中で初めて `self._cache = ...` を代入するノードを 1 歩動かして取得すると、木のキーは `['_cache']` になる。新しいインスタンスに復元すると `ValueError: ... unexpected ['_cache']` で run が止まる（照合メモ 1）
    - 受け取った値を `self._x = self.prev.recv()` として持つ書き方は、PC 系のノードでは自然に出てくる。この書き方をすると、保存はできても読み戻せないチェックポイントになる。逆向きも同じで、`on_start` で登録した名前を `on_update` や `on_stop` で `None` にすると「missing」になる。時間とともに shape が変わるテンソル（伸びるバッファ、最初の `recv` で batch の大きさを決める遅延確保）は shape の検査で落ちる
    - `specs/nodes.md` の「`__init__` でも `on_start` でも、どの hook で代入してもよい」は、取得については正しいが、復元については誤りである。少なくとも「復元できるのは、`on_start` を終えた時点の登録（名前と shape）が取得したときと一致する状態だけ。状態にする Tensor は `__init__` か `on_start` で確保し、その後はその場で更新する」と書く。torch.md の「その場で更新」の推奨とも揃う
    - 実装で救う案もある（木にあって登録に無い Tensor は、木のコピーをそのまま属性に置く）。ただし、その場で更新する規約の例外が増え、デバイスの扱いも要るので、文書化だけで足りるかをメインが判断する

3. **復元した木の GPU のコピーが、run の間ずっと `Session._states` に残る。既定の CUDA の復元の安全性は、この保持に暗黙に頼っている**（確信度: 中。CUDA は読みでの判断）
    - 以前の利用者の `load_state_dict` は、木のテンソルを属性として引き取ることが多く、コピーは 1 つで済んでいた。既定がその場の `copy_` になったので、`Session.load_state_dict` の deepcopy（CUDA のノードなら GPU 上）と、ノード自身のテンソルの 2 つが run の間ずっと並ぶ。状態の分の GPU メモリが 2 倍になる
    - 一方で、ノードの `copy_` はノードのストリームに積まれるだけで、`start` の後に同期は無い。いまは `_states` が参照を持ち続けているので、コピー元の block がキャッシュアロケータに返らず、読み終わる前に再利用されることは無い。メモリを返そうとして `_states` を start の後に捨てると、ノードのストリームの `copy_` がまだ読んでいる block を、ホストのストリームで再利用する競合が生まれる
    - 案: `_states` を start の後に手放す。そのときは、ノードのスレッドで `load_state_dict` の後に自分のストリームを同期する（または `record_stream`）。手放さないなら、この依存をコメントで残す。どちらにしても CUDA のテストで確かめる

## nit

1. `.grad` は状態に入らない（`nn.Module.state_dict` と同じ）。`.backward()` で数歩分の勾配を溜めてから更新するノードは、復元すると止めずに続けた結果と一致しない。torch.md の「状態」の節で、optimizer の状態と並べて書くとよい（確信度: 高）
2. `on_stop` で GPU メモリを返すつもりで `self._w = None` とすると、終了後の `state_dict` からそのキーが消える。「`on_stop` は `state_dict` が返すものを変えない」の規約は既にあるが、自動収集でこれを破りやすくなった。具体例として書いておくとよい（確信度: 高）
3. `TestRestore` は取得の直後に `pickle.loads(pickle.dumps(...))` するので、この 1 件だけではフレームワークの deepcopy を確かめていない。手動モードでは間にノードが動かないため、pickle がコピーの役を果たしてしまう。既存の `test_network.py` の `test_state_dict_is_whole_while_a_node_updates_in_place` などが deepcopy を見ているので、穴にはなっていない（確信度: 高）
4. 既定の `state_dict` / `load_state_dict` を CUDA のノードで通すテストが無い（既存の `TestCuda` は自前の state_dict を使う）。仕組みは同じだが、should-fix 3 の保持への依存を固定する意味で、1 件あるとよい（確信度: 中）
5. 2 つのノードが同じ Tensor / nn.Module を属性に持つと、状態の所有（書き手は 1 人）に反する。しかし自動収集は、それを両方のノードの状態として黙って集める。値は停止中に同じなので、取得・復元の結果は壊れない。今回の範囲ではないが、build で storage の重複を検出する余地がある（確信度: 中）

## 救済措置の下地（`state_items`）の設計メモについて

妥当である。生のオブジェクトを返し、`state_dict` の override に左右されない、という切り分けは救済の用途に合っている。メモに足すとよい点:

- CUDA の sticky なエラー（device-side assert など）で止まった場合は、コンテキストが使えなくなっている。ストリームの同期も、CUDA のテンソルを読むことも例外になる。「同期してから読む」だけでは足りず、CUDA のエラーで止まったときは CUDA のテンソルを救済できないことがある、と書く
- 例外で止まったとき、ほかのノードはループの区切りで止まっていて、その後に `on_exception` と `on_stop` も走る。救済した木はノードごとに時刻が揃っておらず、`on_stop` 後の値でもある。通常の `load_state_dict` に渡せる木と型で区別するか、少なくとも木に印を付ける
- 例外を投げたノードのスレッドは終わっているが、救済をどのスレッドで読むか（ホストのスレッドか）、読む前に全ノードのスレッドの終了を待つかを決めておく

## 検証結果

- `just run`: format（変更なし）、cargo test / pytest 382 passed・14 skipped（CUDA は環境に無く skip）、pyright 0 errors、mkdocs build --strict が通った。作業後の `git status` は clean
- 実験（scratchpad の `ns/exp.py`・`ns/exp2.py`、worktree の外）:
    - `on_update` で初めて Tensor を代入するノードを、取得してから復元すると、`load_state_dict` の `ValueError: ... unexpected ['_cache']` で止まる
    - `super()` を呼ばない `on_thread_start` では、`on_start` の中で `in_hook()` が False、勾配の記録が True、`Network.read` が通る。`on_update` の `send` は `_check_owner` の RuntimeError で止まる

## 照合メモ

1. 取得の流れ: `Session.state_dict` → `runtime.capture(cb)`（停止中に cb）→ 各ノードの `_capture_state`（CUDA: 自分のストリームで sync → `state_dict()` → sync）→ `_tree` がノードごとに `copy.deepcopy` → 制御スレッドの現在のストリームを sync → 再開。既定の `state_dict` は `detach()` と `module.state_dict()` だけで、カーネルを積まない。`Tensor.__deepcopy__` は leaf であることを要求する。detach の結果は leaf で、`.grad` も持たない。storage の deepcopy は `memo[storage._cdata]` で共有を保つ
2. 復元の流れ: `Session.load_state_dict` が deepcopy する → ホストの現在のストリームを sync する → `run()` でもう一度 sync する → ノードのスレッドで `on_thread_start` → `on_start` → `load_state_dict`（`copy_` はノードのストリーム）→ 最初の `on_update`（同じストリームなので順序は保たれる）。コピー元は `Session._states` が持ち続ける（should-fix 3）
3. `TestRestore` の Δt: 元の経路は T_k で停止（凍結）→ 再開 → advance(0.001) で Δt = 0.001。復元の経路は timer と last_update_ns を T_k に戻す → advance(0.001) で Δt = 0.001。m 歩とも同じ Δt と同じ演算なので、ビット一致する
