# 🔌 Port と Channel

Port はノードの接続口で、`connect` が 2 つの Port を 1 本の Channel でつなぐ。このページでは Port の種類、spec、つなぎ方、分岐と合流、誰も送る前に Port が持つ値を説明する。

## Port の種類

| 種類        | 相手                                          | `recv()` が返すもの                  |
| ----------- | --------------------------------------------- | ------------------------------------ |
| `Port`      | 受け取るなら 1 つまで。送るだけならいくつでも | 値                                   |
| `MultiPort` | 同じ種類の相手をいくつでも                    | `connect` の順の `dict[Channel, 値]` |

Port は `__init__` の中で `super().__init__()` の後に、ノードの属性として宣言する。実行は走り始めるときにインスタンスの属性を読む。リスト・タプル・dict に入れた Port には実行が、クラス属性として宣言した Port にはクラス文が `TypeError` を送出する。Port は 1 つずつ属性に代入するか、`MultiPort` を使う。

入力とその教師信号、予測と誤差のように役割の違う入力は、役割がわかる名前の別々の `Port` で受ける。`MultiPort` は、ノードが同じように束ねる同じ種類の相手のためのもの。

## spec

Port には送るものの spec を `send=`、受け取るものの spec を `recv=` でキーワード引数として渡す。片方向の Port では使わない方を省く。

```python
self.prev = Port(recv=TensorSpec((32, 784)))  # 受け取るだけ
self.next = MultiPort(send=TensorSpec((32, 256)), recv=TensorSpec((32, 256)))  # 両方向
```

`TensorSpec(shape, dtype)` はテンソル 1 つ。dtype を省くと、`torch.get_default_dtype()` の設定によらず `torch.float32` になる。送る値は shape と dtype の両方が spec と一致しなければならず、dtype は変換しない。spec が取れる dtype は `TensorSpec.dtypes` にある。spec はその shape と dtype の値も作れるので、ノードの状態やテストに使える。`spec.zeros()` と `spec.empty()`（`DictSpec` ならその dict）、`TensorSpec.rand(generator=...)` がある。

回数・フラグ・時刻は、テンソルではなく Python のスカラーで運ぶ。受け取る側は `int(...)` で取り出さずに済む。

| spec                            | 値                            | 初期値            |
| ------------------------------- | ----------------------------- | ----------------- |
| `IntSpec(bits=64, signed=True)` | 8 / 16 / 32 / 64 bit の `int` | `0`               |
| `FloatSpec()`                   | `float`（64 bit）             | `0.0`             |
| `BoolSpec()`                    | `bool`                        | `False`           |
| `TimestampSpec()`               | `mujou.time.Timestamp`        | `Timestamp(ns=0)` |

メッセージの大きさは固定なので、`IntSpec` は bit 数を持つ。範囲外の `int` は送るときに `OverflowError` になり、折り返さない。スカラーの spec は shape を持たず、値も作らない。`0` や `False` と書けばよい。

2 つの Port の値は別々に届くので、読む側は一方の新しい値ともう一方の古い値を同時に読むことがある。入力とその目標のように組になる値は 1 つのメッセージで送る。`DictSpec` は `TypedDict` のキーごとに spec を与えるもので、dict 全体が一度に届き、キーごとの型は型検査器にも伝わる。1 つのキーの spec を使うときは、種別を指定して取り出す（`SAMPLE.field("x", TensorSpec).shape`、種別が違えば `TypeError`）。

```python
class Sample(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


SAMPLE = DictSpec(Sample, x=TensorSpec((32, 4)), y=TensorSpec((32, 2)))

self.next = Port(send=SAMPLE)  # 送る側: self.next.send({"x": x, "y": y})
self.prev = Port(recv=SAMPLE)  # 受け取る側: sample = self.prev.recv(); sample["x"]
```

Port の型は型検査器が spec から推論する。注釈が必要なとき（補助関数の引数など）は `Port[S, R]` か `MultiPort[S, R]` と書く。`S` は送る型、`R` は受け取る型。省いた向きは `Never` になり、何も送らない Port の `send` は型検査器が拒む。任意の spec を受け取る関数は `MessageSpec[T]` で注釈する。

```python
from typing import Never

from mujou import MessageSpec, MultiPort, Port


def mean_of(port: MultiPort[Never, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(port.recv().values())).mean(0)


def sender[T](spec: MessageSpec[T]) -> Port[T]:
    return Port(send=spec)
```

## つなぐ

`a.next.connect(b.prev)` は 2 つの Port をつなぎ、`Channel` を返す。ノードとネットワークは `connect` を持たない。ノードの入出力は複数ありうるので、Port を 2 つとも書けばどの経路をつなぐかが読み取れる。

- 互いの送る spec が相手の受け取る spec と、両方向で一致しなければならない。一致しなければ `connect` が `TypeError` を送出する
- 受け取る `Port` の相手は 1 つだけで、2 つ目をつなぐと `ValueError` になる。複数の相手から受け取るなら `MultiPort` を使う
- 同じ 2 つの Port をもう一度つなぐと、すでにある Channel が返る。自分のノードの Port 同士をつないでもよい
- 循環してつないでもよい。送る相手から受け取ってもよく、`mujou.pc` のノードは両方向の Port でそうしている
- つなげるのは同じ実行のノードの Port だけ。木の外の相手があると `mujou.run` が `ValueError` を送出する

## fan-out と合流

送るだけの `Port` は相手をいくつでも取り、送った値をそれぞれに届ける。合流には `MultiPort` を使う。その `recv()` は Channel ごとの最新の値を返す。まだ一度も送っていない相手の値は、相手の Port の初期値になる。

```python
--8<-- "examples/branching.py:fan-out"
```

値をどう束ねるか（足す、平均する、相手ごとに重みを掛ける）は受け取るノードが決める。相手ごとに扱いを変えるときは、`connect` が返した `Channel` を受け取るノードの属性に持たせ（`self.sum.from_double = self.double.next.connect(self.sum.prev)`）、それで引く。`self.prev.recv()[self.from_double]`。

## 1 つの Channel に送る

`MultiPort` の送り方は 2 つある。

| 送り方                    | 送り先                   | `Run.read` が返すもの |
| ------------------------- | ------------------------ | --------------------- |
| `send(value)`             | すべての相手に同じ値     | この値                |
| `send_to(channel, value)` | `channel` の先の相手だけ | 変わらない            |

```python
--8<-- "examples/branching.py:send-to"
```

ネットワークは `connect` が返した Channel をノードに渡す（`self.split.plus = ...`）。`self.next.channels` から順番で取り出さないこと。`connect` を書く順を入れ替えると、エラーも出ずに送り先が入れ替わる。`Channel` を持つ属性は状態に入らない。

ホストが `Run.read` で読めるのは、Port が `send` で送ったもの（それまでは初期値）だけ。`send_to` の値も、受け取るだけの Port も読めない。

## 誰も送る前

Port が最初に送るまでは、相手も、その Port を読むホストも、Port の初期値を受け取る。初期値はゼロ（スカラーは上の表の初期値）か、`initial=` で与えた値。`initial` は `send` と一致しなければならず、Port を作った時点でコピーされる。その時刻は `None`。

```python
self.next = Port(send=TensorSpec((32, 256)), initial=torch.full((32, 256), 0.5))
```

`recv_with_time()` は値を送られた時刻（`mujou.time.Timestamp`）と一緒に返し、最初に送られるまでは時刻が `None`。前回読んだ値の時刻と比べれば、新しい値か同じ値の読み直しかがわかる。ホストが書く Port は、実行の `inputs=` かホストの最初の `write` から送られたものとして扱う。[ホストの入出力](host-io.md#%E5%88%9D%E6%9C%9F%E5%80%A4)を参照。

## 公開した Port

ネットワークは子の Port を自分の属性に代入して公開する（`self.prev = self.l1.prev`）。その属性はただの参照で、Port はノードのもののまま。外からはノードの Port と同じようにつなぐ（`x.next.connect(net.prev)`）。慣習では最初のノードの Port を `prev`、最後のノードの Port を `next` として公開し、ネットワークもノードと同じようにつなげるようにする。
