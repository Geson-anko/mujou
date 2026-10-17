# docs/en/user-guide（index・writing-a-node・building-a-network・running・compose）理解度テスト

## 対象

- ファイル: `docs/en/user-guide/index.md`、`writing-a-node.md`、`building-a-network.md`、`running.md`、`compose.md`
- 取り込むサンプル（生徒も読む）: `examples/leaky_chain/leaky.py`、`examples/quickstart.py`、`examples/branching.py`、`examples/subscribe.py`、`examples/compose.py`
- 読者と用途: Concepts の章を読んだ英語の利用者が、自分のノードのクラスを書き、ネットワークに組み、`mujou.run` で走らせてホストから読み書きし、`Compose` で束ねる
- 日本語版 `docs/ja/user-guide/` の同名 5 ページ（確定版）の英訳。節・表・コード片・`--8<--` の行範囲を日本語版とそろえる。問題は `guide-a-ja.md` の評価用 15 問・保留 3 問を英訳したもの

## 前提にした API（api-polish、別ブランチ）

- `Run.write` が書いた時刻（`Timestamp`）を返す
- `mujou.run(node)` がノード 1 つを受け付ける。コミット 899b202 で確認: 単独のノードは `Compose(node)` と同じく、パス `"0"`（Port は `"0.prev"`）になる。日本語版はこのパスを書いていないので、英語版も書いていない（報告に載せた）

## 確かめたこと

- `--8<--` の行範囲（quickstart 43:51・54:62、branching 19:75・77:80、subscribe 37:47、compose 38:59）が日本語版の参照先と同じ内容を指す
- 英語の見出し `## Branching and merging` のアンカーは `#branching-and-merging`（built site で確認）。英語版ではページ内アンカーへのリンクを使った
- 用語は `docs/en/concepts/` と `docs/en/user-guide/{state,time,torch,testing,errors}.md` に合わせた: host, run, the run's timer, global timer, port, channel（コードでは `Channel`）, time scale, pause, Mujou（散文）

## 評価用問題

### Q1 You want to write a node that reads a sensor only once every 20 ms. Where and how do you set the period? May you set a default with a class attribute `period = 0.02`?

- 要点: call `super().__init__(period=0.02)` in `__init__`
- 要点: not as a class attribute; the start of the run raises `TypeError`

### Q2 In the `on_update(dt)` of a node with `period=0.01`, may you multiply by the constant 0.01 instead of `dt`? Why?

- 要点: no. `dt` is the time that actually passed and is not exactly `period`
- 要点: after a delay it is longer than `period`, and the next one is shorter. Even with a period, update in proportion to `dt`

### Q3 A node with `period=0.01` was 35 ms behind its target at one update. Is it updated several times in a row to catch up? How is the next target decided?

- 要点: no, not several times in a row
- 要点: since it was `period` or more behind, the next target is counted again from the time that update actually happened (+ `period`)

### Q4 When the machine is busy and `dt` gets large, the update may diverge. May you cut `dt` at the top of `on_update` with `dt = min(dt, 0.01)`? What should you do instead?

- 要点: no. The cut-off time is lost and the result drifts from the continuous-time solution
- 要点: split the step inside one `on_update` (divide `dt` into n steps and update n times)

### Q5 A node creates its state `w` in `on_start` and writes `self.w = self.w + dt * g` in `on_update`. How should it be fixed? Why?

- 要点: update in place (`self.w += dt * g` or `copy_`)
- 要点: assigning a new tensor raises no error, but references from elsewhere (an optimizer, another attribute) keep pointing at the old tensor and later updates do not reach them

### Q6 You want the initialization of a node to be reproducible. May you call `torch.manual_seed(0)` in the node's `on_start`? What should you do instead?

- 要点: no. It changes the random number generator that all nodes share
- 要点: pass `super().__init__(seed=0)` and draw random numbers from `self.generator`

### Q7 In a branch, one `NextPort` sends different values to two peers with `send_to`. What is the safe way to get the `Channel` of each destination? What is the problem with using the order of `self.next.channels`?

- 要点: give the `Channel` that `connect` returns in the network to an attribute of the sending node
- 要点: `channels` is in the order of connection, so changing the order of the `connect` calls silently swaps the destinations

### Q8 You run the `Diamond` of `branching.py`, and the host reads `run.read(net.split.next)`. What does it return? Where do you read to see the values sent to each branch?

- 要点: the initial value (zeros). `send_to` does not update the value `Run.read` returns
- 要点: the values themselves cannot be read; read the port of a branch node that sends with `send` (e.g. `net.double.next`, `net.square.next`)

### Q9 What does `recv()` of a `PrevPort(multi=True)` return? What is the value for a channel whose peer has not sent yet?

- 要点: a dict keyed by `Channel`, in the order of connection
- 要点: zeros

### Q10 In the `Classifier` example of Building a network, what is the path of the node `l2`? If you later rename the attribute `body` to `trunk`, can you restore a state saved before the rename?

- 要点: `"body.l2"`
- 要点: no, because the path changes

### Q11 You want to show the input `x0` from the very first `on_update`. How do you pass it? How does that differ from calling `run.write({net.prev: x0})` right after `mujou.run(net)`?

- 要点: pass it as `mujou.run(net, inputs={net.prev: x0})`
- 要点: when `mujou.run` returns the nodes are already running, so the updates before the `write` read zeros (the initial value)

### Q12 You want to show inputs `x` one after another and record the output for each `x`. In what order do you `write`, wait and `read`? Why must you not `read` right after `write`?

- 要点: `write` → wait for the showing time with `mujou.time.sleep` → `read` (take the output at the end of showing as the response)
- 要点: the output catches up after a delay, so reading right after returns the output for the previous input

### Q13 You want to switch an input and its teacher signal at the same time. May you split it into two calls, `run.write({net.prev: x})` and `run.write({net.teacher: y})`? Why?

- 要点: put them in one `write` (`run.write({net.prev: x, net.teacher: y})`)
- 要点: values written in one `write` switch at the same time. Split, a node may read the new value of only one of them

### Q14 In a run on the global timer, may you use the standard `time.sleep(0.1)` to show an input for 0.1 seconds? What do you use? Why?

- 要点: no. Use `mujou.time.sleep(0.1)`
- 要点: the standard `time.sleep` ignores changes of the time scale and the pauses for taking the state

### Q15 A network A that controls a robot and another network B run in the same process. When you take the state of B regularly, you do not want A to stop. How do you run A and B? (The two are not connected by ports.)

- 要点: do not compose them (no `Compose`, no `mujou.run(a, b)`); use separate runs
- 要点: give each a separate `mujou.time.Timer()` with `timer=`. Without it, both share the global timer and taking B's state stops A's time too

## 保留問題

### H1 You iterate over `run.subscribe(net.next)` with `for` to handle the output. How does the iteration end? If your handling is slow, do you receive every value that was sent?

- 要点: it ends when the run ends normally; call `run.shutdown()` from another thread to stop it
- 要点: no. Values in between are skipped; you get the latest value at that moment

### H2 You ran the network `a` alone with `mujou.run(a)` and it ended. What happens if you pass the same instance `a` to `mujou.run(a, b)`? How can you run them together?

- 要点: `RuntimeError` (a network runs once; it was built alone)
- 要点: create new instances, nodes included, and pass them together from the start

### H3 `run.shutdown(timeout=1.0)` raised `TimeoutError`. What state is the run in? What do you call to wait for the end?

- 要点: the request to stop is not withdrawn; the ending goes on in the background
- 要点: call `run.join()` to wait for it

## ラウンド記録

### R0（執筆、5 ファイル計 497 行。日本語版も 497 行）

英訳と出題。`uv run pre-commit run --files ...`（mdformat が表を整形）と `uv run mkdocs build --strict` が通る。採点はまだ。

### R1（497 → 497 行）

正解 15/15。正解の問題について、生徒が「読みにくかった箇所」に挙げた点のうち 2 点を直した。

| 問 | 判定 | 原因 | 直したこと |
| -- | ---- | ---- | ---------- |
| Q1–Q15 | 正解 | - | - |
| （Q3） | - | 曖昧 | 数え直しの条件が 2 通りの言い方で書かれていた。「period 以上遅れたら」を主にし、「次の目標を過ぎている」は括弧の言い換えにした |
| （Q5） | - | 曖昧 | 何が壊れるのか（ほかの参照）を主語にした。復元が on_start のテンソルにその場で書き込むことは独立した文に分け、理由のように読めないようにした |

Q5 の生徒は、復元の話を理由の一部として書いた。要点の「参照が古いテンソルに残る」は満たしているので正解とした。

直さなかった指摘:
- Leaky の係数の打ち切り: orchestrator の指示で、examples を直すときにまとめて扱う
- 枝に送った値の読み方: 文書に「値そのものは読めない」とあるので足りていると判断した
- まだ送られていないときに read が返す Timestamp: 日本語版にもなく、実物で確かめていないので書いていない
- Timer の API: time.md の担当範囲
- compose の表と箇条書きの重複: 箇条書きとコード片に書いてあり、Q15 は正解だった

次は保留問題 H1–H3 を出す。

### H（保留問題、497 → 497 行）

H1–H3 すべて正解。生徒の「読みにくかった箇所」から running.md の 2 点を直した。

- `join` の表の行が「別のスレッドが shutdown を呼ぶのを待つ」だけを書いていた。shutdown が時間切れになった後にも使うことが、表の下の段落とつながらなかった（埋没）。表の行に「after `shutdown` times out」を足した
- subscribe の反復を別のスレッドから止める理由が書いていなかった（欠落）。反復が、反復しているスレッドをふさぐことを書いた

直さなかった指摘: 「正常に終了」の定義は errors.md の範囲。「最初から一緒に走らせる」はサンプルの compose.py に `mujou.run(first, second)` がある。「終了中」という状態の名前は付けない（要らない用語を増やさない）。
