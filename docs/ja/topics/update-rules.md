# 🧮 更新則

更新則（学習則）は、ノードが `on_update(dt)` に書く更新そのもの。mujou はノード・Port・スケジューラに更新則を組み込まないので、ノードごとに選べ、同じネットワークのまま差し替えられる。このページでは、ノードをまたいで学ぶ則の書き方を feedback alignment を例に説明する。1 つのノードの内側の誤差逆伝播法（BP）は [4. ノードの中で学習する](../tutorial/4-learn-in-a-node.md)、`mujou.pc` の則は [PC-ALM](pc.md) にある。

## ノードの内側と、ノードの間

ノードの内側では、BP を含め PyTorch で計算できることは何でも使える。ノードの間では計算グラフをつなげない。Port が送るのは切り離したコピーである。そのため、ノードをまたいで学ぶ則は、各ノードが要るものをメッセージで送る。活動を前へ、誤差（やほかの credit）を後ろへ、目標は活動と一緒に。各ノードは自分の状態と隣のメッセージだけから学ぶ。

## feedback alignment

`examples/feedback_alignment.py` は $\hat y = W_2 \tanh(W_1 x)$ を 2 つのノードで学ぶ。`hidden` が活動を前へ送り、`output` が誤差 $e = \hat y - y$ を後ろへ返す。BP なら `hidden` は credit $W_2^\top e$ を受け取るが、`hidden` は `W2` を知らない。代わりに自分の固定の乱数行列 $B$ で $B e$ を使う。それでも $W_2$ が $B^\top$ にそろっていき、学べる（[Lillicrap et al., 2016](https://doi.org/10.1038/ncomms13276)）。

```sh
uv run python examples/feedback_alignment.py
```

```python
--8<-- "examples/feedback_alignment.py:hidden"
```

```python
--8<-- "examples/feedback_alignment.py:output"
```

プログラムは各バッチから学ぶ前の `output` の損失を表示する。損失は 6 秒で約 1.5 から約 0.02 まで下がる。`hidden` を固定すると 0.15 を下回らない。

## 両方向の Port

`hidden.next` は `Port(send=ACTIVITY, recv=ERROR)`、`output.prev` は `Port(send=ERROR, recv=ACTIVITY)`。1 本の接続で活動を前へ、誤差を後ろへ運ぶ。循環してつないでもよいので、誤差を別の Port で返してもよい。

## 目標を中継する

活動はそのサンプルの目標 `y` と、元になった書き込みの時刻を運ぶ。2 つの Port の値は別々に届くので、目標を別の Port で送ると、予測を別のサンプルの目標と組んでしまうことがある。1 つのメッセージにすれば、`output` はいつも予測を自分の目標と比べ、`hidden` は持っている入力と同じ `time` の誤差だけから学ぶ。`mujou.pc` のノードも同じように、`y`・`mask`・`time` を `pc.Activity` で運ぶ。

## 新しいかどうかを判定する

`recv()` は新しいかどうかによらず最新の値を返す。各サンプルから 1 回だけ学ぶノードは、時刻で新しいサンプルを見分ける。`recv_with_time()` が返す時刻か、メッセージが運ぶ時刻を使う。誰も送る前の時刻は `None`（メッセージが運ぶ時刻なら、初期値のゼロが表すもの）になる。

## `pc.LearnClock`

どちらのノードも、いつ学ぶかを `pc.LearnClock` で決める。`mujou.pc` の部品だが、どの則でも使える。更新ごとに `tick(dt, time_ns, now_ns)` に `dt`、入力が書かれた時刻、今の時刻を渡すと、`pc.Tick` が返る。

- `new`: 入力が新しい（時刻が前回見たものと違う）
- `learn`: この更新で学ぶ。`learn_interval` を渡したなら、各書き込みから `learn_interval` 後の最初の更新（入力が変わらない間は以後 `learn_interval` ごと）。ただし入力を初めて見た更新では学ばない。`learn_interval=None` なら毎回
- `elapsed`: 前に学んでからの時間。各 `dt` は `max_dt` で打ち切る。刻みはこれで重み付けする。$W \leftarrow W - \frac{\text{elapsed}}{\tau}\,\nabla$ は勾配流 $\tau \dot W = -\nabla$ をたどるので、どれだけ学ぶかはノードの更新回数ではなく時間で決まる
- `dt`: `max_dt` で打ち切った `dt`

クロックは `state_dict` と `load_state_dict` を持つので、属性に置くだけでノードの状態に入る。経路上のノードは実行の 1 本のタイマーを読むので、深さによらず各書き込みからほぼ同じ時刻に学ぶ。
