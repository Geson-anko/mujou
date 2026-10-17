# src/mujou/_network.py と src/mujou/_node.py の docstring 理解度テスト

## 対象

- ファイル: `src/mujou/_network.py`、`src/mujou/_node.py`（docstring とコメントだけ。英語）
- 読者と用途: mujou の利用者。docstring だけを読んで、Node を継承して hook を書き、Network に組み、`mujou.run` で実行し、write / read / subscribe / state_dict / load_state_dict / exception_state_dict / shutdown / join を正しい順序で使う
- 根拠: `docs/network.md`、`docs/execution-model.md`、`docs/nodes.md`、`docs/torch.md`、`_session.py`、`crates/mujou-core/src/runtime.rs`（復元時に別ネットワークが走っていると `ValueError`、`freeze_for_startup`）

## 評価用問題

### Q1 Node のサブクラスで重みの Tensor を `self.generator` を使って乱数で初期化したい。初期化をどの hook（またはメソッド）に書くか。`__init__` に書くと何が起きるか

- 要点:
  - `on_start` に書く（状態の確保と初期化は `on_start`）
  - `__init__` で `self.generator` を使うと `RuntimeError`（生成器はノードのスレッドで作るので、`on_start` 以降でしか使えない）

### Q2 あるノードの `on_update` の中から、別のネットワーク `other_net.read(port)` を呼んでその出力を使いたい。してよいか

- 要点:
  - してはいけない。hook の中から Network のメソッドを呼ぶと `RuntimeError`（どの Network でも）
  - ノードは自分の Port だけでやり取りする

### Q3 ノードの `state_dict()` の中で `self.prev.recv()` を呼び、最新の入力も状態に入れたい。できるか。理由も

- 要点:
  - できない
  - `state_dict` はノードのスレッドではなく制御スレッドで、ノードが止まっている間に呼ばれるので `recv` / `send` を使えない

### Q4 保存した状態の木 `state` からネットワークを再開したい。ホストが呼ぶメソッドを順に書け。また、各ノードの `load_state_dict` は `on_start` と最初の `on_update` に対していつ呼ばれるか

- 要点:
  - 同じ構造のネットワークを作る → `build()` → `load_state_dict(state)` → `run()`（`load_state_dict` は `run` の前でなければならない）
  - ノードの `load_state_dict` は `on_start` の後、最初の `on_update` の前

### Q5 ホストが入力 `net.prev` と教師 `net.teacher` を新しいサンプルに切り替える。`write` を 2 回に分けて呼ぶのと、1 回の `write` に両方を入れるのとどちらにするか。理由も

- 要点:
  - 1 回の `write` に両方を入れる
  - 1 回の `write` の値は一斉に切り替わり同じ時刻が付く（どの瞬間にもノードは全部新しいか全部古いかを見る）

### Q6 実行中にあるノードの `on_update` が例外を投げた。ホストはループで `net.read(...)` を呼んでいる。ホストでは何が起き、元の例外はどこから取れるか。例外の時点の状態の木を得るには、事前に何をしておき、何を呼ぶか

- 要点:
  - ネットワーク全体が止まり、次の `read` が `NetworkError` を送出する
  - 元の例外は `NetworkError` の `__cause__`
  - `run(capture_state_on_exception=True)`（または `mujou.run(..., capture_state_on_exception=True)`）で実行しておき、`exception_state_dict()` を呼ぶ（これは `NetworkError` を送出しない）

### Q7 ルートのネットワーク `net` が属性 `body` にサブネットワークを持ち、`body` が属性 `l1` にノードを持つ。`net.state_dict()` の木で、このノードの状態はどのキーの下にあるか。また、`body.l1.prev` を `net.prev` として公開していて、ホストがそこへ `write` した値のスロットのキーは何か

- 要点:
  - `tree["nodes"]["body.l1"]`
  - `tree["slots"]["body.l1.prev/in"]`（公開名の `prev/in` ではない）

### Q8 ホストのスレッドでネットワークを止めたいとき、`shutdown()` と `join()` のどちらを呼ぶか。もう一方を呼ぶとどうなるか

- 要点:
  - `shutdown()`。全ノードを止め、すべての `on_stop` が返ってから戻る
  - `join()` は止めるよう求めないので、ほかの誰かが `shutdown` するか例外で止まるまで戻らない

### Q9 ノードの中でノイズを生成し、テストで決定的にしたい。さらに状態の木から復元した後も同じ乱数列を続けたい。どう書くか

- 要点:
  - `super().__init__(seed=...)` で seed を渡し、`self.generator` を使う（`torch.manual_seed` や既定の生成器は使わない）
  - 生成器は起動のたびに seed から作り直されるので、`state_dict` に `self.generator.get_state()` を入れ、`load_state_dict` で `set_state` する

### Q10 ノードの内側で autograd を使った BP（`loss.backward()`）を `on_update` に書いた。そのままで勾配は計算されるか。どう書けばよいか

- 要点:
  - されない。ノードのスレッドでは勾配の記録が切られている
  - その部分を `with torch.enable_grad():` で囲み、`# mujou: local-autograd` を付ける

### Q11 `super().__init__(device="cuda:0")` としたノードで、`recv()` が返す Tensor はどのデバイスにあるか。ホストの `net.read(...)` が返す Tensor はどうか

- 要点:
  - `recv` はノードのデバイス（`cuda:0`）
  - `read` は CPU 上の新しいコピー

### Q12 `net.state_dict()` が `PauseTimeoutError` を送出した。ネットワークは動いているか。何が原因で、何を変えれば待ち時間を延ばせるか

- 要点:
  - ネットワークは動き続けている
  - あるノードが `pause_timeout`（実時間の秒）以内に止まらなかった（今の `on_update` が終わらなかった）
  - `run(pause_timeout=...)`（`mujou.run` も同じ引数）で延ばす

### Q13 同じプロセスで別のネットワークが走っている間に、`load_state_dict` で復元したネットワークを `run()` した。何が起きるか。どうすれば実行できるか

- 要点:
  - `ValueError`。何も起動しない
  - もう一方のネットワークを止めてから、同じネットワークで `run()` をもう一度呼べる

## 保留問題

### H1 ノードがロボットのアクチュエータを動かしている。ノードの例外で終わったときだけ安全な姿勢に戻し、通常の `shutdown` では戻さない。どの hook に書くか。その hook で守ることは

- 要点:
  - `on_exception()`（例外のとき全ノードに 1 回、`on_stop` の前。`on_stop` は通常の終了でも呼ばれるので使わない）
  - `state_dict` が返す内容を変えない

### H2 最初の出力を見逃さないように、`run()` の前に `net.subscribe(net.head.next)` を呼びたい。できるか。`run` の前に出力を見る方法はあるか

- 要点:
  - できない。`subscribe` は `run` の前だと `RuntimeError`
  - `read` は `build` の後なら `run` の前でも呼べる（最初の `send` まではゼロ）

### H3 ノードの `state_dict` が、`on_update` でその場更新している `self._w` をそのまま `{"w": self._w}` として返している。clone する必要はあるか

- 要点:
  - 不要。フレームワークがノードの再開前にコピーするので、返った木は後の更新で変わらない
  - 返す値は pickle できること

## ラウンド記録

### R0（執筆。_network.py 496 → 532 行、_node.py 193 → 215 行）

- Node: hook の一覧（呼ばれる時機・スレッド・制約）を追加。hook の例外で全体が止まること、hook から Network のメソッドを呼べないこと、ノード内 BP の書き方、`_on_thread_start` 等を override しないことを明記。`generator` に `torch.manual_seed` を使わない旨
- Network: ホストの呼び出し順を番号付きリストに。例外の伝わり方（`run` 中なら `run` が、それ以後は各メソッドが送出、`exception_state_dict` は送出しない）
- `run`: 復元時に別ネットワーク稼働中なら `ValueError` を追加（実装 `freeze_for_startup` で確認）
- `state_dict`: 木の形を具体例で、スロットのキーの付け方。`load_state_dict`: ノードへの渡し方と時機。`shutdown` / `join`: 戻るタイミングと違い。`exception_state_dict`: `NetworkError` を送出しない
- 行数の増加は初回の執筆（欠落の補完）による

### R1（_network.py 532 → 533 行、_node.py 215 → 217 行）

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1〜Q13 | 正解 | - | - |

生徒が挙げた「読みにくかった箇所」のうち、妥当なものを直した。

- `join`: 停止させる主体がいないと戻らないことを明記した（曖昧）
- ノード内 BP: 「that part」を「forward の計算と `backward` の両方」に言い換えた（曖昧）
- `state_dict` の中の `recv` / `send`: `RuntimeError` になると明記した（欠落）
- `on_start`: 復元するときも呼ばれると明記した（曖昧）
- 見送り: generator のコード例（行が増えるうえ、手順は文で足りる）、`/in` の位置（スロットのキーを説明する 1 文の中にあり、探す範囲が狭い）

### 汎化確認（保留問題、750 行のまま）

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| H1 | 正解 | - | - |
| H2 | 正解 | - | - |
| H3 | 正解 | - | - |

保留問題に基づく改稿は禁止なので、文書は変えていない。生徒の指摘のうち、次回の候補として次を残す。

- `on_exception` は例外を投げていないノードでも呼ばれることを明記する
- `state_dict` のコピーが deepcopy であることを書く
- `run` が「戻る時点」に最初の `on_update` を含まないことを明記する
- 手順 4 で、`subscribe` が `run` の後だけ使えると明記する
