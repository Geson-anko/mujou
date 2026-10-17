# docs/en/concepts/ 理解度テスト

## 対象
- ファイル: `docs/en/concepts/` の `index.md`・`nodes.md`・`ports.md`・`networks.md`・`time.md`・`state.md`（6 ページ、計 397 行）
- 読者と用途: 日本語版と同じ。mujou を初めて使う利用者（PyTorch は使えるが mujou の設計は知らない）が、Concepts の章だけを読んでノード・Port と Channel・ネットワーク・実行ごとの時間・状態の考え方を説明でき、従来の学習との違いが分かり、User Guide に進める
- 内容: `docs/ja/concepts/` の同名ページ（R2 と汎化確認後の改稿を経た 397 行）の翻訳。節・表・リスト・mermaid・数式・コードの構造をそろえた
- 根拠の確認: `src/mujou/_core/{node,port,network,run,timer}.py`・`src/mujou/{time,testing}.py` の docstring、`specs/nodes.md`
- 表記: 散文は "Mujou"、コードは `mujou`（`docs/en/` の既存ページに合わせた）。User Guide へのリンク文はそのページの見出し

## 日本語版との食い違い（日本語版は直さず報告）
- `docs/ja/concepts/time.md` は大きすぎる `dt` への対処として「上限で打ち切るか、刻みを分割する」と書く。`specs/nodes.md`（period の節）と `Node` の docstring は「打ち切ると経過時間が失われる。ノードの内側で刻みを分割する」としている。英語版は後者に合わせた（`specs/execution-model.md` と `specs/pc-alm-continuous-time.md` には前者の記述が残る）。R1 の時点で、日本語の担当が `docs/ja/concepts/time.md` と specs の 2 か所を直した

## 評価用問題
（日本語版ノート `concepts-ja.md` の Q1〜Q13 を英訳したもの。要点は同じ）

### Q1 In PyTorch you switch between training and inference with `model.train()` / `model.eval()`. Does Mujou have a way to switch a network between a training mode and an inference mode? Also explain why Mujou is designed that way.
- 要点: モードの切り替えは無い（ノードは走っている間、更新を続ける）
- 要点: 理由: ストリームから学び続ける機械が対象で、学習と推論を分けると推論している間は学べないから

### Q2 There are two nodes, A and B. To let gradients flow from B's loss to A's weights, you want to pass A's output to B and call `loss.backward()` inside B. Can you do this? Explain why. If you want to train A and B together with BP, what should you do?
- 要点: できない（ノードをまたぐ backward は禁止）
- 要点: 送る値は送るときに計算グラフから切り離されるので、勾配は Port を通らない
- 要点: 1 つのノードのクラスにまとめる（ノードの内側に閉じた BP は使える）

### Q3 Node B calls `self.prev.recv()` in its `on_update`. The previous node A has not written anything since B last read. What does B get? What if A has never written at all?
- 要点: 待たずに、前回と同じ（最新の）値を受け取る
- 要点: 一度も書いていなければ初期値（ゼロ）

### Q4 One node updates less often than the others because of a heavy computation. How should the size of its updates be handled compared with nodes that update more often? Answer using the argument of `on_update`.
- 要点: 更新量を `dt`（前回の更新からの経過時間、秒）に比例させる（例: `x = x + dt * f(x)`）
- 要点: 頻度が低いと `dt` が大きくなるだけで、同じ時間の上で同じダイナミクスを近似できる（反復回数あたりで決めない）

### Q5 In host code you want to show one sample to the network for 0.1 seconds, then write the next sample. The network runs with `mujou.run(net)`, without passing `timer`. What do you use to wait? Also explain why the standard `time.sleep(0.1)` is wrong here.
- 要点: `mujou.time.sleep(0.1)`（グローバルタイマーで待つ）
- 要点: `time.sleep` は実時間で待つので、時間の速さの変更や停止と合わない

### Q6 Inside a node's `on_update`, you want to call `mujou.time.set_time_scale(0.5)` to slow down that node's time. Can you?
- 要点: できない。hook の中で呼ぶと `RuntimeError`
- 要点: 理由として「時間を動かすのはホストだけ」か「ノードが速さを変えると同じ実行のほかのノードの `dt` も変わる」のどちらかを挙げる

### Q7 You called `run.state_dict()` on a running network to take its state. Is the time the network was paused for this included in each node's `dt` after it resumes? Explain why.
- 要点: 含まれない
- 要点: 止めている間は実行のタイマーも止まるから（ダイナミクスから見ると停止は無かったのと同じ）

### Q8 A node of the network raised an exception in `on_update`, and the run ended. Then the host tried to save the state with `run.state_dict()`. What happens? When should the state be taken if you want to save it?
- 要点: 取得できず `NetworkError` になる
- 要点: 実行中か、正常に終了した後に取る

### Q9 In a node's `on_start`, you assigned `self.w = torch.randn(256, 256)`, `self.lr = 0.01` and `self._h = torch.zeros(32, 256)`. Without overriding `state_dict`, which of them are in the node's state by default?
- 要点: `self.w` と `self._h`（Tensor は名前の `_` に関係なく入る）
- 要点: `self.lr`（数値）は入らない

### Q10 To free GPU memory, a node sets `self.w = None` in its `on_stop`. What goes wrong when you take `run.state_dict()` after a normal end?
- 要点: `w` が状態から消える（`None` を代入すると状態から外れる）
- 要点: `on_stop` は `state_dict` が返す状態を変えてはならない

### Q11 The root attribute `body` of the network `Classifier` is a `Stack`, and the attribute `l2` of `Stack` is a node. What is this node's path? And what is its path when you run `mujou.run(Compose(cls=Classifier()))`?
- 要点: `"body.l2"`
- 要点: `"cls.body.l2"`

### Q12 Inside `class Pair(mujou.Network)`, you hold nodes `a` and `b` as attributes. Does wrapping them in a network make `a` and `b` wait for each other and update synchronously? If you want them synchronized, what do you do?
- 要点: 同期しない。ネットワークは構築時だけの構造で、中のノードはそれぞれ独立に互いを待たずに動く
- 要点: 同期させたいなら、それらの計算を 1 つのノードのクラスにまとめる

### Q13 Node `head` has a `NextPort` and is connected to two peers with `head.connect(x)` and `head.connect(y)`. What is the form of the value `head` gets from `self.next.recv()`? If you want the sum of the two peers' values, who adds them?
- 要点: Channel をキーにした dict（接続した順）
- 要点: 足すのはノード自身（mujou は束ね方を決めない）

## 保留問題
### H1 The host tries to write a value with `run.write` to a port that is already connected to a neighboring node. Can it? Explain why.
- 要点: 書けない（ホストが書けるのはつながっていない Port だけ）
- 要点: 各値の置き場所の書き手を 1 つに保つため

### H2 Runs X and Y both run on the global timer. When you call `run.state_dict()` on X, what happens to Y's time? What do you do if you do not want Y to stop?
- 要点: Y の時間も止まる（停止は同じタイマーを使うすべての実行に及ぶ）
- 要点: どちらか片方に別の `mujou.time.Timer()` を `timer=` で渡して走らせる

### H3 When you restore a saved state tree with `mujou.run(net, state=tree)`, which is called first, `on_start` or `load_state_dict`? And what happens if you pass the tree to a network in which a node's attribute was renamed after the tree was taken?
- 要点: `on_start` が先、その後最初の `on_update` の前に `load_state_dict`
- 要点: パスが変わるので `mujou.run` が `ValueError` を送出し、何も起動しない

## ラウンド記録
### R0（397 行）
- 初稿（日本語版 397 行の翻訳。行数も同じ）。`uv run pre-commit run --files` で mdformat が表を整形、`uv run mkdocs build --strict` は通過。生徒の解答はまだ無い

### R1（397 → 397 行。長くなった文はあるが行数は同じ）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q13 | 正解 | - | - |

読みにくかった箇所への対応:
- ports.md の初期値: 「受け取り側の spec の shape と dtype のゼロ（`DictSpec` ならキーごとのゼロのテンソル）」と書き、括弧書きを 2 文に分けた（相手が複数のつながっていない Port は Channel が無いので空の dict）。`Port.recv` と `TensorSpec.zeros` の docstring で確認
- ports.md の「(only one slot between one-way ports)」: 両方が片方向の Port なら値は一方向にしか流れず、置き場所は 1 つ、と言い換えた
- nodes.md の `on_thread_start`: 「このノードのスレッドだけに効く設定」とし、[PyTorch in Nodes](../user-guide/torch.md) へのリンクを足した
- time.md の `dt` の打ち切り: 理由（実際に経った時間の一部が落ち、連続時間のダイナミクスから遅れる）を足し、「打ち切らない」と命令形にした
- networks.md の `Compose`: 「1 つだけ渡したとき」の文が `mujou.run(net)` と `Compose(enc=a)` のどちらを指すか曖昧だった（曖昧）。`mujou.run(net)` は名前を足さず、`Compose` は 1 つでも名前を足す（例: `Compose(enc=a)` で `"enc.l1"`）と書いた。`src/mujou/_core/run.py` の `start` で確認（ネットワークが 1 つなら `Compose` で包まない）。Q11 の答えそのもの（`cls.body.l2`）は書いていない
- 採点の訂正: 最初は Q4 を誤読（不正解）とした。teacher に届いた解答は要約で、「`dt` に比例」の部分が落ちていた。orchestrator から全文（「a slow node receives a larger `dt`, and `x += dt * f(x)` takes a proportionally larger step」）を受け、要点 1 を満たすので正解に直した。R1 は 13/13。改稿はそのまま残す
- 日本語版にも同じ読みにくさがある（初期値の shape、`Compose` 1 つの場合）。報告に回す

### 汎化確認（保留問題 H1〜H3、397 行）
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1 | 正解 | - | - |
| H2 | 正解（確信: 高／中） | - | - |
| H3 | 正解（確信: 高／低） | - | - |

- H3 の問題文「a node's attribute was renamed」は「ノードを持つ属性」と「ノードの中の属性」の 2 通りに読める（生徒は両方を検討した）。再利用するときは「the attribute that holds a node in the network was renamed」と書く
- ノードの中のテンソル属性を名前替えした場合は、既定の `load_state_dict` が `ValueError`（登録した属性が木に無い）を送出する。User Guide の範囲なので Concepts には書かない

### 汎化確認の後の改稿（397 → 399 行）。ループは終了
生徒の読みにくかった箇所のうち、妥当なものを最小限で直した。
- ports.md: 片方向の Port 同士の括弧書きを独立した文に分けた。「unconnected port with one peer」の自己矛盾を「takes one peer but is not connected」に直した。`head` を定義してから `ch = head.connect(x)` の例を出した
- networks.md: `Graph` の段落を「何か・1 回だけ走る」と「何を公開するか」の 2 段落に分けた。パスの表の見出しを「Network passed to `mujou.run`」にし、各行を「`Compose(a, b)`, with `l1` in `b`」の形にした。名前替えでパスが変わる属性を「ノードかネットワークを持つ属性」と限定した
- state.md: 同じタイマーで走る別の実行のノードについて、「`dt = 0` で待つ」という表現をやめた。`Node` の docstring（period が無いノードは、時間が少しでも進むまで次の `on_update` を呼ばれない）に合わせ、「タイマーが再開するまで新しい `on_update` を始めず、停止は `dt` に入らない」と直した。`_` の文は主語を入れた
- 見送り: ノードの中の属性を名前替えしたときの扱い（User Guide の範囲）
