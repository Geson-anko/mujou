# user-guide 4 ページ（英語）理解度テスト

## 対象
- ファイル: `docs/en/user-guide/` の `state.md`・`time.md`・`building-a-network.md`・`compose.md`（日本語版 `docs/ja/user-guide/` の同名ページも同じ構成で直す）
- 生徒が読むファイル: ページ 1 枚だけ（ページごとに別の生徒）。`--8<--` で取り込むサンプルは渡さない
- 読者と用途: ノードを書ける利用者。ネットワークを組み、複数のネットワークをまとめて走らせ、時間を扱い、状態を保存・復元するコードを、これらのページを見ながら間違えずに書く

## 事実確認（R0）
`src/mujou` の docstring と照合した（`run.py`・`node.py`・`port.py`・`network.py`・`timer.py`・`time.py`・`session.py`・`_native.pyi` の `Timestamp`）。4 ページに docstring と食い違う記述は見つからなかった。

## 評価用問題

### S1（state）You restore a run with `mujou.run(net2, state=tree, inputs={net2.prev: x_new})`. The tree also holds a value for that input port. Which value does the node first receive on `prev`?
- 要点: `x_new`。`inputs` の値が木の入力の値より優先される

### S2（state）To "clean up", a node calls `self.w.zero_()` in `on_stop`. After the `with` block ends normally, the host calls `run.state_dict()` and saves it. What is saved for `w`, and what should the node do instead?
- 要点: ゼロにした `w` が木に入る（正常終了後の state は `on_stop` の後のノードの状態を写す）
- 要点: `on_stop` で状態を変えない

### S3（state）A node keeps `self.history = [t1, t2, ...]` (a list of tensors) and an optimizer whose moments matter. You save and restore the run. Are they restored? If not, what do you do?
- 要点: 復元されない（自動で集まるのは属性に直接入れた Tensor と nn.Module だけ。list と optimizer の状態は集まらない）
- 要点: `state_dict` と `load_state_dict` の両方を override し、`super()` を呼んで自動の部分を残す

### S4（state）A tree was saved with `torch.save` on a machine where the node computes on CUDA. You want to restore it on a machine without a GPU, with the node on the CPU. How do you read the file, and do you have to move the tensors to the CPU yourself before passing them to `state=`?
- 要点: `torch.load(..., map_location="cpu")` で読む
- 要点: 自分で移す必要はない。そのまま `state=` に渡せる（既定の `load_state_dict` がノードの device にコピーする）

### T1（time）A run was started with its own `timer = mujou.time.Timer()`. The host wants to present an input for 0.5 s of that run's time, then compare the time returned by `run.read` with "now". Which call waits, and which call gives "now"?
- 要点: `timer.sleep(0.5)`（`mujou.time.sleep` はグローバルタイマー、標準の `time.sleep` は実時間）
- 要点: `timer.now()` と比べる。`mujou.time.now()` / `time()` とは比べない

### T2（time）A node should update every 10 ms. May it call `mujou.time.sleep(0.01)` at the end of `on_update`? If not, what should it do?
- 要点: いけない。hook の中で呼ぶと `RuntimeError`
- 要点: `super().__init__(period=0.01)` で周期を渡す

### T3（time）The global time scale is 5. The host presents an input with Python's standard `time.sleep(1.0)`. About how much global time passes? What should it call instead?
- 要点: 約 5 秒（標準の `time.sleep` は実時間で待ち、スケールと一時停止を無視する）
- 要点: `mujou.time.sleep(1.0)`（その実行のタイマーの `sleep`）を使う

### T4（time）The host wants to freeze the global time for a moment by calling `mujou.time.set_time_scale(0)`. What happens?
- 要点: `ValueError`。スケールは 0 より大きい有限の数でなければならない

### T5（time）A node wants to record the time it was created: `self.t0 = self.timer.now()` in `__init__`. What happens, and where can it read the time?
- 要点: `RuntimeError`（`__init__` では読めない）
- 要点: 実行がノードを開始してから読める（`on_start` や `on_update` で）

### B1（network）In a network's `__init__`, you write `self.layers = [Hidden(4, 4), Hidden(4, 4)]` and connect `self.layers[0].connect(self.layers[1])`. Are these nodes part of the network? How should you write it?
- 要点: 入らない（ネットワークは直接の属性だけをたどり、list や dict の中のノードは入らない）
- 要点: 1 つずつネットワークの属性に代入する（`self.l1 = ...`、`self.l2 = ...`）

### B2（network）A node `split` sends to its two branches only with `send_to`. The host calls `run.read(net.split.next)`. What does it get? How can the host see what the branches received?
- 要点: そのポートの初期値（`initial`、なければゼロ）。`send_to` は `Run.read` の値を更新しない
- 要点: 送られた値そのものは読めない。分岐先のノードが受け取った値から計算して `send` した結果を読む

### B3（network）A node merges two peers with `PrevPort(recv=..., multi=True)`. What does `self.prev.recv()` return? What is in it for a peer that has not sent anything yet? Who adds the values together?
- 要点: `Channel` をキーとする dict（接続の順）
- 要点: 送っていない相手の値は、相手のポートの初期値（`initial` がなければゼロ）
- 要点: 受け取るノードが自分で合わせる。Mujou は合わせない

### B4（network）You call `net.build()` to check the structure of `net`. Afterwards you want to put `net` inside a bigger network and connect another node to it. Can you?
- 要点: できない。`build()` は凍結するので、その後の `connect` は `RuntimeError`、より大きなネットワークの一部にもなれない
- 要点: `build()` は走らせるネットワークにだけ呼ぶ

### B5（network）A network holds a node `head` with a `Port` named `teacher`, and does not expose it. Can the host write to it with `run.write`? If yes, how does it name the port?
- 要点: 書ける（未接続のポートなら公開していなくても書ける）
- 要点: 子の属性をたどって `run.write({net.head.teacher: y})` のように指定する。公開は短い名前を与えるだけ

### C1（compose）Network `a` holds a node at attribute `l1`. What is that node's path (its key in the state tree) when you run `mujou.run(a, b)`, and when you run `mujou.run(Compose(enc=a, dec=b))`?
- 要点: `"0.l1"` と `"enc.l1"`

### C2（compose）You saved a tree from `mujou.run(enc, dec)`. Later you restore with `mujou.run(Compose(enc=enc2, dec=dec2), state=tree)` using new instances. Does it work? Why?
- 要点: 動かない（`ValueError`）。パスが `"0..."`/`"1..."` から `"enc..."`/`"dec..."` に変わり、木のキーと合わない
- 要点: 保存したときと同じ合わせ方（同じ順、同じキーワード）で合成する

### C3（compose）You want network `a` to feed network `b` in one run. Is `mujou.run(Compose(a=a, b=b))` enough?
- 要点: 足りない。`Compose` は何も接続しない
- 要点: 合成する前に `a.connect(b)` で接続する

### C4（compose）Two independent networks must run at the same time. Taking the state of one must not pause the other, and an exception in one must not stop the other. How do you run them?
- 要点: 合成しない。別々の実行にする（`mujou.run` を 2 回）
- 要点: それぞれに自分の `mujou.time.Timer()` を渡す。`timer` を渡さない実行は 2 つともグローバルタイマーを共有し、一方の状態を取るともう一方の時間も止まる

## 保留問題

### H1（state）Run A is running on the default global timer. While it runs, you call `mujou.run(netB, state=tree)` to restore another network. What happens, and what are your two options?
- 要点: `ValueError` になり、何も起動しない
- 要点: 先に A を終わらせる、または B に自分の `mujou.time.Timer()` を渡す

### H2（network）A branching node takes its destinations as `self.plus = self.next.channels[0]` and `self.minus = self.next.channels[1]`. A colleague later swaps the order of the two `connect` calls in the network. What happens, and how should the destinations be passed instead?
- 要点: エラーなしに送り先が入れ替わる（`channels` は接続の順）
- 要点: ネットワークで `connect` が返す `Channel` を送る側のノードの属性に代入する

## ラウンド記録
生徒はページごとに別（ページ 1 枚だけを渡す）。正解したページは次のラウンドで出し直さない。行数は 4 ページとも en/ja で変わらない（state 86・time 81・building 114・compose 61）。

### R1（全 18 問）16/18
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| S1〜S4 | 正解 | - | - |
| T1, T2, T4 | 正解 | - | - |
| T3 | 正解（確信 中） | 欠落 | スケールの意味（実時間 1 秒にタイマーが何秒進むか）を書き、標準 `time.sleep` の説明を「スケールを無視し、一時停止の間も数え続ける」に直した |
| T5 | 不正解（読める hook を判断できない） | 曖昧 | 「once the run has started the node」→ どの hook でも読める（`on_thread_start` と `on_start` を含む）。docstring どおり |
| B1〜B4 | 正解 | - | - |
| B5 | 正解（確信 中） | 曖昧 | 「The host writes to unconnected ports. It can also write …」を 1 つの規則（受け取る・未接続のポートなら公開の有無を問わず書ける）にまとめた |
| C1, C3, C4 | 正解 | - | 読みにくさの指摘を受け、末尾のコード片を 2 つの実行に別々の Timer を渡す形にした |
| C2 | 不正解（「判断できない」。不一致の結果が書いていない） | 欠落 | 合わないと `ValueError` になると書いた（`mujou.run` の docstring） |
| - | - | 生徒の指摘 | Nesting のコメント「(in fact l2.next)」を言い換えた |

### R2（time・building・compose の 14 問）12/14
| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| T1〜T5 | 正解 | - | - |
| B1 | 不正解（「判断できない」と前置き） | 欠落 | 「ノードは 1 つずつ属性に代入する」を足した |
| B2〜B5 | 正解 | - | - |
| C2 | 不正解（「動く」と答えた） | 埋没・曖昧 | 「位置とキーワードでパスが変わる」を足し、復元の段落を表の直後に移した。「同じキーワード」を「キーワードは保存時に使ったところでだけ使う」に |
| C1, C3, C4 | 正解 | - | - |

### R3（building・compose の 9 問）9/9

### 汎化確認: H1 正解・H2 正解（2/2）
- H2 の生徒の指摘（`connect` の戻り値をどう代入するかがページ本文から見えない）を受け、取り込みのサンプルの 1 行（`self.split.plus = self.split.connect(self.double)`）を本文に引いた

### 最終
- 評価用 18 問すべて、最後に出したラウンドで正解。保留 2/2
- docstring とページの食い違いは見つからなかった
