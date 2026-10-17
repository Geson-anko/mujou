# ノードを書く

ノードは `mujou.Node` を継承したクラスとして書く。このページは、その書き方を手順の順に説明する。ノードとは何かは[Mujou の仕組み](../concepts.md#%E3%83%8E%E3%83%BC%E3%83%89)にある。

## 例: リーク積分器

入力 $u$ を時定数 $\tau$ で追いかけるノードである。状態 $x$ は $\frac{dx}{dt} = \frac{u - x}{\tau}$ に従う。

```python
--8<-- "examples/leaky_chain/leaky.py"
```

このノードは、次の 3 つを書いている。

1. `__init__` で Port を宣言する（`prev` と `next`）
2. `on_start` で状態 `x` を確保する
3. `on_update(dt)` で、入力を読み、`dt` に比例して状態を更新し、結果を送る

以下、この順に説明する。

どのサブクラスでもするように、`__init__` では最初に `super().__init__()` を呼び、override した hook でも最初に `super()` の同じ hook を呼ぶ。`mujou.Node` の hook は、`state_dict` と `load_state_dict` を除いて何もしない（[状態](state.md)）。それでも呼んでおく。そうすれば、後で hook が何かをする自作のノードのクラスを継承しても、正しく動き続ける。

## 1. `__init__` で Port を宣言する

Port は `__init__` の中で、インスタンスの属性として代入する。クラス属性として宣言すると、ネットワークの構築（`build()`）で `ValueError` になる。

Port の種類は、ネットワークの中での役割で選ぶ。

| 役割                                              | 宣言       |
| ------------------------------------------------- | ---------- |
| 前のノードから受け取る側（`a.connect(b)` の `b`） | `PrevPort` |
| 次のノードへ送る側（`a.connect(b)` の `a`）       | `NextPort` |
| それ以外（教師信号の入力、観測用の出力など）      | `Port`     |

1 つのノードが持てる `PrevPort` と `NextPort` は、それぞれ 1 つまでである。`Port` はいくつでも持てる。

各 Port には、送る値の型を `send=`、受け取る値の型を `recv=` で渡す。型は `TensorSpec(shape, dtype)` で書く。dtype を省くと、spec を作った時点の `torch.get_default_dtype()`（普通は `float32`）になる。片方向の Port では、使わない方の引数を省く（`None` を渡さない）。

```python
self.prev = PrevPort(recv=TensorSpec((32, 784)))                       # 受け取るだけ
self.next = NextPort(send=TensorSpec((32, 256)), recv=TensorSpec((32, 256)))  # 両方向
```

相手の数は `multi=` で決まる。既定は `PrevPort` と `Port` が 1 つ、`NextPort` が複数である。相手が複数の Port の `recv()` は、値そのものではなく dict を返す（[ネットワークを組む](building-a-network.md#%E5%88%86%E5%B2%90%E3%81%A8%E5%90%88%E6%B5%81)の「分岐と合流」）。

送る値は spec と shape も dtype も一致しなければならない。dtype は自動で変換されない。上の `Leaky` が `torch.float64` の spec を使っているので、送る値も `float64` で作っている。

Port が最初に送るまで、相手が受け取る値と、ホストがその Port から読む値はゼロである。別の値から始めたいときは、`initial=` で初期値を与える。初期値は `send` と shape も dtype も一致しなければならず、Port を作った時点でコピーされる。状態を復元した実行（[状態](state.md)）は、初期値ではなく状態の中の値から始まる。

```python
self.next = NextPort(send=TensorSpec((32, 256)), initial=torch.full((32, 256), 0.5))
```

2 つの Port の値は別々に届く。読む側は、一方の Port の新しい値と、もう一方の Port の古い値を同時に読むことがある。組になる値（入力とその教師など）は 1 つのメッセージにして送る。`TypedDict` と、キーごとの `TensorSpec` を持つ `DictSpec` で型を書く。dict 全体が一度に届き、キーごとの型は型検査器にも分かる。

```python
class Sample(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


SAMPLE = DictSpec(Sample, x=TensorSpec((32, 4)), y=TensorSpec((32, 2)))

self.next = NextPort(send=SAMPLE)  # 送る側: self.next.send({"x": x, "y": y})
self.prev = PrevPort(recv=SAMPLE)  # 受け取る側: sample = self.prev.recv(); sample["x"]
```

## 2. `on_start` で状態を確保する

状態のテンソルは `on_start` で作る。`on_update` では新しいテンソルを代入し直さず、`+=` や `copy_` などでその場で更新する。

代入し直してもエラーにはならず、新しいテンソルが同じ名前で状態に入る。困るのは、`on_start` で作ったテンソルへのほかの場所（optimizer や別の属性など）からの参照である。その参照は古いテンソルを指したまま残り、以後の更新が届かない。その場で更新すれば、どの参照も同じ 1 つのテンソルを指し続ける。状態を復元するときも、mujou はそのテンソルにその場で書き込む。

属性に代入した `torch.Tensor` と `nn.Module` は、状態として自動で集まる。何が集まり、どう保存するかは[状態の保存と復元](state.md)にある。

ノードを GPU で計算するときは、テンソルを `device=self.device` で作る。

```python
@override
def on_start(self) -> None:
    self.w = torch.zeros(256, 256, device=self.device)
```

設定値（時定数、層の大きさなど）は `__init__` で数値として持たせてよい。数値は状態に集まらない。

`nn.Module` は `__init__` で作っても `on_start` で作ってもよく、どちらでも状態に集まる。乱数による初期化は、`on_start` 以降に使える `self.generator` で行う。そうして初期化するモジュールは `on_start` で作る。

## 3. `on_update(dt)` で `dt` に比例して更新する

`on_update(dt)` は、mujou が繰り返し呼ぶ。ノードの作者はループを書かず、1 回分の処理だけを書く。1 回の中でする処理は次の順である。

1. `recv()` で隣の最新の値を読む。新しい値が来るのを待つことはない
2. 状態を、`dt` に比例した量だけ更新する
3. `send()` で結果を送る

`dt` は、このノードの前回の更新からの経過時間（秒）である。最初の `dt` は、全ノードの `on_start` が終わって実行のタイマーが動き出した時点から数える（復元した実行では、状態を取る前の最後の更新から数える）。微分方程式 $\frac{dx}{dt} = f(x)$ なら `x += dt * f(x)` と書く（陽的 Euler 法）。更新量を `dt` に比例させないと、速く回るノードほど速く変化してしまう（[グローバル時間](../concepts.md#%E3%82%B0%E3%83%AD%E3%83%BC%E3%83%90%E3%83%AB%E6%99%82%E9%96%93)）。

`dt` は一定ではない。マシンが混むと長くなる。`on_update` は、起こりうる最も長い `dt` でも発散しないように書く。`dt` 自体は切り捨てない。切り捨てた分の経過時間が失われ、連続時間の解からずれる。長い `dt` でも安定に更新する方法は 2 つある。

- 厳密解がある系では、厳密解で `dt` だけ進める。上の `Leaky` は陽的 Euler 法ではなく、$x$ を $u$ までの距離の `1 - exp(-dt / tau)` 倍だけ動かす。この係数は、入力が一定の間は厳密である。どんな `dt` でも 1 未満なので、$x$ は $u$ を行き過ぎない
- 厳密解がない系では、1 回の `on_update` の中で刻みを分割する。1 回の刻み `dt / n` が安定に更新できる長さ以下になるように `n` を決め、`for` ループで `n` 回更新する（[ノードの中の PyTorch](torch.md) の例）

`recv()` は必ず値を返す。`on_update` はたいてい相手が送るより頻繁に呼ばれるので、同じメッセージを何度も読む。持続する値（センサーの読み、活動など）ならそれでよく、そのたびに `dt` だけその値へ向けて更新する。連続時間の更新に雑音を足すときは、標準偏差を `dt` ではなく $\sqrt{dt}$ に比例させる。そうすれば、結果がノードの更新の頻度によらない。

メッセージが 1 回だけ扱う出来事（学習のサンプルなど）のときは、`recv_with_time()` で時刻も読み、時刻が変わっていなければ更新を飛ばす。`on_start` で時刻を読んでおけば、相手が最初に送る前からある値を飛ばせる。その値は、Port の初期値（送る側が `initial=` を宣言していなければゼロ）か、復元した状態の値である。復元した値は、状態を取る前にすでに扱っている。（ホストが書く Port の `inputs=` の値も同じ時刻を持つので、これも飛ばされる。）

```python
@override
def on_start(self) -> None:
    super().on_start()
    _, self.last = self.prev.recv_with_time()  # いま届いている値の時刻


@override
def on_update(self, dt: float) -> None:
    super().on_update(dt)
    sample, t = self.prev.recv_with_time()
    if t == self.last:  # 前回の更新から新しいメッセージが無い
        return
    self.last = t
    ...  # サンプルから 1 回だけ学習する
```

## 実行の周期を決める

既定では、`on_update` が戻るとすぐに次の `on_update` が呼ばれ、ノードはできるだけ速く回る。このとき、ノード 1 つが CPU のコアを 1 つ使い切る。センサーの周期に合わせたいときや、CPU を空けたいときは、周期 `period`（秒）を渡す。

```python
class Sensor(mujou.Node):
    def __init__(self) -> None:
        super().__init__(period=0.01)  # 平均 10 ms に 1 回更新する
        self.next = mujou.NextPort(send=mujou.TensorSpec((3,)))
```

周期は、更新と更新の間の平均の間隔である。`on_update` にかかった時間も周期に含まれる。遅れたノードが、取り戻すために続けて何回も更新することはない。

待つ時間は実行のタイマーで測る。時間の速さを変えると、周期も実時間に対して同じ比で変わる。

周期を決めても、`dt` は `period` ちょうどにはならない。`dt` は実際の経過時間で、遅れると `period` より長く、その次は短くなる。周期があっても、`dt` に比例して更新する。

周期を持つノード（センサーなど）から入力を受けるパイプラインでは、後ろのノードは次の入力が来るまで同じ入力を読む。自分の周期が無いと同じ計算を繰り返し、それぞれが CPU のコアを 1 つ使い切る。後ろのノードにも、入力の周期以下の周期を与える。

`period` は `super().__init__` の引数でだけ渡す。クラス属性 `period = 0.01` で既定値を置くと、実行の開始で `TypeError` になる。手動の実行（[テスト](testing.md)）では周期は使われない。

## シードとデバイスを決める

`super().__init__` は次の引数を取る。すべてキーワード引数で、省略できる。

| 引数     | 意味                                                                                 | 既定                     |
| -------- | ------------------------------------------------------------------------------------ | ------------------------ |
| `device` | ノードが計算するデバイス。`"cpu"`、CUDA のデバイス、`"mps"` などほかのアクセラレータ | `"cpu"`                  |
| `seed`   | ノード専用の乱数生成器 `self.generator` のシード                                     | `None`（毎回違う）       |
| `period` | 実行の周期（秒）                                                                     | `None`（できるだけ速く） |

乱数は `self.generator` から引く（`torch.randn(..., generator=self.generator)`）。`torch.manual_seed` はノードの中で呼ばない。全ノードが共有する乱数生成器を変えてしまうからである。`self.generator` は `on_start` 以降の hook で使える。PyTorch との組み合わせの詳細は[ノードの中の PyTorch](torch.md)にある。

`device`・`generator`・`period`・`timer` は予約された名前である。これらの名前の属性を代入しない。

## hook とその順序

mujou は、ノードの hook を次の順に呼ぶ。override するのは必要なものだけでよい。

1. `on_thread_start()`: `on_start` の前に 1 回。スレッドごとの設定を書く（[ノードの中の PyTorch](torch.md)）
2. `on_start()`: 1 回。状態を作る
3. `load_state_dict(state)`: 復元するときだけ 1 回（[状態の保存と復元](state.md)）
4. `on_update(dt)`: 実行が終わるまで繰り返し
5. `on_pause()` と `on_resume()`: 実行の状態を取る前と後。その間に `state_dict()` が呼ばれる
6. `on_exception()`: どれかのノードのどれかの hook が例外を投げたとき、全ノードで 1 回（[例外と終了](errors.md)）
7. `on_stop()`: 終了のときに 1 回。正常な終了でも例外による終了でも呼ばれる

## hook の中でしてはいけないこと

| してはいけないこと                                                          | 理由・結果                                                                                                     |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| ほかのノードの値が新しくなるのを待つ（ループで `recv()` を繰り返すなど）    | ノードは互いを待たない。値が変わらなければ同じ値を使って進む                                                   |
| `mujou.run` や `Run` のメソッド（`write`・`read`・`state_dict` など）を呼ぶ | `RuntimeError` になる。ノードは自分の Port だけでやり取りする                                                  |
| `mujou.time.sleep`・`set_time_scale` など、時間を動かす操作                 | `RuntimeError` になる。時間を動かすのはホストだけである                                                        |
| ノードをまたいで `backward()` する                                          | 送った値は計算グラフから切り離されるので、勾配は Port を通らない                                               |
| `on_update` から戻らないループを書く                                        | 一時停止・終了・状態の取得ができなくなる。ループは mujou が持つ。刻みを分割する有限の `for` ループは使ってよい |

時刻を知りたいときは、`self.timer.now()` で実行のタイマーの時刻を読む。読むだけなら hook の中で使える。

どの hook が例外を投げても、ネットワーク全体が終了する（[例外と終了](errors.md)）。
