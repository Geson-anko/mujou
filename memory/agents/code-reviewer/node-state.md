# node-state（状態の自動収集と on_thread_start の公開）レビュー

対象: `git diff 942f5ca..HEAD`（ea9684b..c4cb9ad）、ブランチ `feature/2026-10-16/node-state`。
再現に使ったスクリプトは scratchpad に置いた（コミットしない）。

## verdict: request-changes

must-fix は 1 件。仕様書の記述が実装と食い違う。ほかは動作が正しい範囲の改善と、run-handle との merge の注意。

## must-fix

### M1. 「どの hook で代入してもよい」が、既定の load_state_dict の復元と矛盾する

- 対象: `specs/nodes.md`「状態の自動収集」の「`__init__` でも `on_start` でも、どの hook で代入してもよい」、`src/mujou/_core/node.py:229-235`
- 問題: 初めて代入するのが `on_update` の Tensor 属性（直前の入力のキャッシュ `self.last = x` など、よくある書き方）は、実行中の `state_dict` には入る。しかし復元では `load_state_dict` が `on_start` の後、最初の `on_update` の前に呼ばれる。その時点ではまだ名前が登録されていないので、`ValueError: ... missing [], unexpected ['last']` になる。保存はできるのに、その木から復元した実行は起動できない
- 根拠: 次で再現した（確信度 高）

  ```python
  class L(Node):
      def on_update(self, dt): self.last = torch.ones(1)
  l = L(); l.on_update(0.1); tree = copy.deepcopy(l.state_dict())
  L().load_state_dict(tree)  # ValueError: ... unexpected ['last']
  ```

- 直し方の候補（裁定は orchestrator）: (a) 仕様と docstring を「既定の復元を使うなら、状態は `__init__` か `on_start` で代入する」に直し、エラーメッセージにもその手がかりを出す。(b) 登録されていないキーの Tensor は、`copy_` の代わりに代入して登録する（ただし「その場で copy_」というユーザーの決定から外れる）
- 深刻度: 中（仕様違反で、利用者が普通に踏む）

## should-fix

### S1. setter 付きの property に Tensor を代入すると、property の名前まで登録される

- 対象: `src/mujou/_core/node.py:170-176`
- 問題: 登録するかどうかを、インスタンスの記憶域に入った値ではなく、代入した値で決めている。setter が `self._w = v` と書く property `w` に `n.w = tensor` を代入すると、`_w` と `w` の両方が登録される。結果、同じ値が状態に 2 回入り、復元で 2 回コピーされる。setter が値を変換する場合（`self._wl = v.tolist()`）は、`getattr(node, "w")` が list を返すので、`state_dict()` が `AttributeError: 'list' object has no attribute 'state_dict'` を投げる
- 根拠: 再現した（`property: ['_w', 'w']`、`property-convert raises: AttributeError`）。`Node` 自身も `device` / `generator` を property で持ち、利用者のノードでも property は普通に書く。`type(self)` 上のデータ記述子（property、`__slots__` の member は除く）の名前を登録しないようにすれば防げる
- 確信度: 高 / 深刻度: 中

### S2. `__setattr__` を override すると、pyright がどの Node のサブクラスでも未宣言の属性への代入を通すようになる

- 対象: `src/mujou/_core/node.py:169-181`
- 問題: pyright は `__setattr__` を持つクラスへの任意の属性の代入を許す。そのため、クラスの外からの綴り間違い（`node.lerning_rate = 0.1`）が strict でも検出されなくなる。宣言済みの属性の型の検査（`a.n = "x"` や `a.w = None`）は残る
- 根拠: 確かめた。`__setattr__` を持たないクラスへの `Plain().undeclared = 1` は `reportAttributeAccessIssue` になり、持つクラスへの `Hooked().undeclared = 1` と、`Node` のサブクラスへの `a.undeclared = torch.ones(1)` はエラーにならない。`if not TYPE_CHECKING:` の下で定義すれば、型検査からは見えなくなる。あわせて、`tests/mujou/_core/test_node.py` の `setattr(node, "y", y)  # noqa: B010  # not declared by Stateful` の理由のコメントは、今は成り立たない（直接代入しても pyright は通す）
- 確信度: 高 / 深刻度: 低〜中

### S3. `copy.copy(node)` をすると、登録表が元のノードと共有される

- 対象: `src/mujou/_core/node.py:310-318`（登録表を `vars(node)["_state_names"]` に置く設計）
- 問題: 浅いコピーは `__dict__` を浅く写すので、`_state_names` の dict が同じオブジェクトになる。コピーに Tensor を代入すると元のノードにも名前が載り、元の `state_dict()` が `AttributeError: 'D' object has no attribute 'z'` を投げる
- 根拠: 再現した（`copy shares registry: ['x', 'z'] True`）。`deepcopy` は問題ない。ノードを浅くコピーする場面は少ないので深刻度は低い。`__copy__` を定義するか、docstring に注意を書くかで足りる
- 確信度: 高 / 深刻度: 低

### S4. 「何も書き換えない」のテストが、検査とコピーを交互に行う素朴な実装を見分けられない

- 対象: `tests/mujou/_core/test_node.py` の `test_rejects_a_tree_that_does_not_match_and_changes_nothing`
- 問題: shape の不一致を入れるのが、登録順で最初の `w` である。項目ごとに「検査してコピー」する実装でも、最初の項目で失敗するので何も書き換わらず、このテストは通ってしまう。登録順で後ろの Tensor（`p` など）の shape を崩し、前の `w` が書き換わっていないことを確かめれば、「まとめて検査してからコピーする」という契約を検証できる。また、Tensor の位置に Tensor でない値がある場合の `TypeError` の経路はテストされていない
- 根拠: skill `testing-strategy`（契約を見分けるテスト）。docstring の「The keys and the shapes of the tensors are checked before anything is copied」
- 確信度: 高 / 深刻度: 低

### S5. `state_items` は、いまの要求にない救済のための下地として公開名で足され、専用のテストまである

- 対象: `src/mujou/_core/node.py:321-325`、`tests/mujou/_core/test_node.py` の `test_lists_the_registered_state`、`specs/nodes.md`「mujou は、override の有無にかかわらず…」
- 問題: 決定（`memory/project_runtime_api_revision.md`）は「救済は自動の状態収集が入ってから別に設計する」である。node.py の中でしか使わない関数を `_` なしの名前にし、テストと仕様の記述まで足すのは、要求から辿れない先取りに当たる。`_state_items` にして専用のテストを外す（振る舞いは `state_dict` を通して確かめられている）か、orchestrator が先取りとして承認するかを決める
- 根拠: CLAUDE.md 開発原則 2・3、skill `refactor-conventions`（テストのための公開は理由にならない）
- 確信度: 中 / 深刻度: 低

## nit

- N1. `node.py:231-235`: エラーメッセージが空のリストも出す（`missing [], unexpected ['last']`）。空でない方だけを出し、M1 の (a) を採るなら「`__init__` か `on_start` で代入する」という手がかりも添える（確信度 高）
- N2. `node.py:249-254`: 「まとめて検査してから書く」のは Tensor の分だけである。登録順で Tensor の後ろにある nn.Module の中が合わないと、前の Tensor はもう書き換わっている（再現: `module mismatch partial: x = tensor([1.])`）。docstring に書いてあり、復元に失敗すれば実行は止まるので影響は小さい。`module.state_dict()` のキーと shape を先に照合すれば、全体を書き換えない形にできる（確信度 高、深刻度 低）
- N3. 同じ Tensor を 2 つの名前で持つと（`self.wref = self.lin.weight`）、状態に 2 回入る。害はないが、仕様に一言あるとよい（確信度 高）
- N4. `copy_` は dtype を黙って変換する。float の木を int64 の属性に読み込むと値が切り捨てられる（`[0.7, 1.5, -2.2] → [0, 1, -2]`）。docstring には書いてあり、`nn.Module.load_state_dict` と同じ振る舞いなので nit とする（確信度 高）
- N5. `expand` などで要素が記憶域を共有している Tensor の属性は、事前の検査を通った後、`copy_` で `RuntimeError` になる。きわめてまれ（確信度 高）
- N6. 登録表の属性名 `_state_names` は、利用者の属性名とぶつかりうる。ぶつかると登録表が壊れる（確信度 中）

## 確認して問題がなかった点（観点の網羅の記録として 1 行ずつ）

- Port、数値、list、Generator、Stream、Timer は登録されない。`nn.Parameter` と、`nn.Module` の中の Parameter や buffer（BatchNorm の `running_mean` など）は正しく扱われる
- `super().__init__()` を呼ばない場合、`__init__` より前の代入（`Node.__new__` の直後）、`__slots__` の member（登録され、復元もできる）、setter のない property への代入の失敗（登録は変わらない）、存在しない属性の `del`（登録は変わらない）
- `state_dict` の `detach` と記憶域の共有は、`Session._tree` が `copy.deepcopy` するので、停止中も正常に終了した後も木は live の値から切り離される。復元側も `_load_state_dict` が deepcopy してからノードに渡す
- `on_thread_start` の改名: Rust の `Node` の既定メソッド、`start` の `call0`、コメント、`_native.pyi` の stub、specs が揃っている。`src` / `crates` / `tests` / `specs` / `docs` に旧名は残っていない（古い `memory/agents/*` のログだけ）
- テストの期待値（`TestOnThreadStart` の乱数列、`TestRestore` の k+m ステップ、`Stateful` の fill の値）は正しい

## run-handle（`.claude/worktrees/run-handle` の未コミットの変更）との merge で衝突しうる点

- `src/mujou/_core/node.py`
    - docstring の hook の一覧（`load_state_dict` と `state_dict` の項目、`Network.state_dict` → `Run.state_dict`）と、その直後の「If any hook raises」の段落。両方が同じ箇所を書き換えているので、テキストが衝突する
    - run-handle は `timer` property を `generator` の直後、つまり `_on_thread_start` の直前に足した。こちらも同じ位置に `__setattr__` などを入れ、`_on_thread_start` を `on_thread_start` に改名したので衝突する。merge するときは、run-handle 側の docstring にある `_on_thread_start`（「framework does this in hooks of its own, ``_on_thread_start``」の段落）も新しい名前に直す
    - 意味の上では問題ない。`set_timer` の `node._timer = timer` は `__setattr__` を通るが、Timer は Tensor ではないので登録は変わらない。`timer` は setter のない property なので S1 にも当たらない
- `specs/nodes.md`: 内部の hook の段落（旧 70 行目付近）を、こちらは書き換え、run-handle はその直後に timer の段落を足しているので、衝突しやすい
- `specs/execution-model.md` 216 行目: 両方が同じ行を書き換えている（こちらは `on_thread_start`、run-handle は `mujou.run`）ので、確実に衝突する。両方の変更を取り込む
- `specs/torch.md`: 書き換えた行が離れているので、おそらく自動で merge できる
- `tests/mujou/_core/test_node.py`: 衝突が最も大きい。run-handle は import（`GlobalTimer` と `RunNetwork` を外し、`StartManual` / `VirtualTimer` を入れる）と、`run_network(..., _manual=True)` の呼び出しをすべて書き換え、`Network.load_state_dict` と `Network._step` も無くしている。こちらで新しく足した `TestOnThreadStart` と `TestRestore`（`run_network`、`virtual_timer: GlobalTimer`、`net._step()`、`restored.build(); restored.load_state_dict(tree)`、`net.state_dict()`、`net.shutdown()`）は、merge の後で `start_manual(Solo(node))`、`start_manual(restored, state=tree)`、`ManualRun.step`、`run.state_dict()` / `run.shutdown()` に書き直す必要がある。書き直さないと、テキストの衝突を解いた後も動かない
- `crates/mujou-python/src/runtime.rs`、`_native.pyi`、`test_native_runtime.py`: run-handle は触っていないので衝突しない

## 検証結果

- just run: pass（format、cargo test、pytest 382 passed / 14 skipped、pyright 0 errors、mkdocs build --strict）
- free-threaded（GIL 無効を確認）: `test_node.py` / `test_native_runtime.py` / `test_network.py` を 5 回実行し、いずれも 151 passed / 11 skipped。全体は 382 passed
- `python -X gil=1`（GIL 有効を確認）: 同じ 3 ファイルを 5 回実行し、いずれも 151 passed / 11 skipped。全体も 382 passed
- 成果物の汚染（`</content>`）: なし
