# ネットワークを組む

ネットワークは `mujou.Network` を継承したクラスとして書く。このページは、ノードのつなぎ方、外に公開する Port、入れ子、分岐と合流、構築した構造の確かめ方を説明する。ネットワークの考え方は[ネットワーク](../concepts.md#%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF)と [Port と Channel](../concepts.md#port-%E3%81%A8-channel) にある。

## 基本の形

`__init__` で、次の 3 つを順に書く。

1. ノード（または別のネットワーク）を属性に代入する
2. それらを `connect` でつなぐ
3. 外からつなぎたい・読み書きしたい Port を、ネットワークの属性に代入して公開する

```python
--8<-- "examples/quickstart.py:network"
```

ネットワークがたどるのは、直接の属性だけである。ノードをリストや dict に入れて持たせても、ネットワークの一部にならない。ノードは 1 つずつ属性に代入する。

## つなぐ

| 書き方                 | つなぐもの                                                    |
| ---------------------- | ------------------------------------------------------------- |
| `a.connect(b)`         | `a` の `NextPort` と `b` の `PrevPort`                        |
| `a.out.connect(b.inp)` | 指定した 2 つの Port。`Port` で宣言した Port はこちらでつなぐ |

`a.connect(b)` と `b.connect(a)` は別の接続である。`a` が前、`b` が次になる。

つなげるのは、片方の `send` の spec がもう片方の `recv` の spec と一致し、逆向きも一致するときだけである。一致しなければ、`connect` を呼んだ時点で `TypeError` になる。相手が 1 つの Port に 2 本目をつなぐと `ValueError` になる。

`connect` は、できた接続（`Channel`）を返す。分岐で相手ごとに違う値を送るときに使う（このページの[分岐と合流](#%E5%88%86%E5%B2%90%E3%81%A8%E5%90%88%E6%B5%81)）。

## Port を公開する

ネットワークの属性に子の Port を代入すると、その Port を公開できる。公開した Port の役割は、Port の種類で決まる。

| 公開した Port | できるようになること                                |
| ------------- | --------------------------------------------------- |
| `PrevPort`    | ネットワークが `x.connect(net)` の `net` 側になれる |
| `NextPort`    | ネットワークが `net.connect(x)` の `net` 側になれる |
| `Port`        | 外から `net.teacher` のような短い名前で指せる       |

ネットワークが公開できる `PrevPort` と `NextPort` は、ノードと同じく 1 つずつまでである。公開できるのは、そのネットワークの中のノードが持つ Port に限る。

ホストは `run.write` で、受け取る Port のうちつながっていないものに値を書ける。公開しているかどうかは問わない。`run.write({net.head.teacher: y})` のように、子の属性をたどって Port を指す。公開は、指す名前を短くするだけである。

## 入れ子にする

ネットワークは、別のネットワークを属性に持てる。公開した `PrevPort` と `NextPort` があれば、子のネットワークもノードと同じように `connect` でつなぐ。

```python
class Stack(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Hidden(784, 256)
        self.l2 = Hidden(256, 256)
        self.l1.connect(self.l2)
        self.prev = self.l1.prev
        self.next = self.l2.next


class Classifier(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.body = Stack()
        self.head = Head()
        self.body.connect(self.head)   # body.next（実体は l2.next）と head.prev をつなぐ
        self.prev = self.body.prev     # 子が公開した Port を、さらに公開する
        self.teacher = self.head.teacher
```

`Hidden` と `Head` は、それぞれ `PrevPort` と `NextPort` を持つノードとする。`Head` は教師信号を受け取る `Port`（`teacher`）も持つ。

ノードは、ルートからの属性名を `.` でつないだパスで区別される。この例のノードのパスは `"body.l1"`、`"body.l2"`、`"head"` である。属性名を変えるとパスが変わり、変える前に保存した状態を復元できなくなる（[状態の保存と復元](state.md)）。

ノードは、ほかのノードやネットワークを属性に持てない。持たせると構築で `ValueError` になる。並べたいノードは、ネットワークの属性に並べる。

## 分岐と合流

1 つの Port に複数の Channel をつなぐと、分岐と合流になる。次の例は、`split` の出力を 2 つに分け、`sum` で合わせる。

```python
--8<-- "examples/branching.py:network"
```

### 分岐: 送る側

`NextPort` は既定で複数の相手を取れるので、`a.connect(b)` と `a.connect(c)` を続けて書けば分岐する。送り方は 2 つある。

| 送り方                    | 送り先                   | `Run.read` で読める値 |
| ------------------------- | ------------------------ | --------------------- |
| `send(value)`             | すべての相手に同じ値     | 更新される            |
| `send_to(channel, value)` | `channel` の先の相手だけ | 更新されない          |

`send_to` だけを使う Port をホストが `Run.read` で読むと、その Port の初期値（`initial`。無ければゼロ）が返る。`send_to` で送った値そのものをホストが読む方法は無い。`Run.read` は送る型を持つ Port しか読まないので、受け取るだけの Port（各枝の `prev`）も読めない。ホストに読めるのは、枝のノードが受け取った値から計算して `send` した結果（上の例なら `net.double.next` の `2 x`）である。

`send_to` に渡す `Channel` は、送り先ごとに区別して持つ。上の例のように、ネットワークで `connect` が返す `Channel` を、送る側のノードの属性に代入する（`self.split.plus = self.split.connect(self.double)`）。`self.next.channels`（つないだ順に並ぶ）から取り出すと、`connect` を書く順を入れ替えたときに、値の送り先が気づかないうちに入れ替わる。`Channel` を代入した属性は状態に集まらない。

### 合流: 受け取る側

`PrevPort(multi=True)` は複数の相手から受け取る。相手が複数の Port の `recv()` は、`Channel` をキーにした dict を、つないだ順に返す。まだ一度も送っていない相手の Channel の値は、相手の Port の初期値である（`initial` を宣言していなければゼロ）。

複数の値をどう束ねるか（足す、平均する、相手ごとに重みを掛ける）は、受け取るノードが決める。mujou は束ねない。上の例の `Sum` は足している。

`NextPort` も既定で複数の相手を取るので、その `recv()` は dict を返す。次のノードが 1 つだけと決まっているなら、`NextPort(send=..., multi=False)` にすると `recv()` が値そのものを返す。

## 構築した構造を確かめる

`net.build()` は、構造と型を検査して凍結し、構築した構造（`Graph`）を返す。`mujou.run` も中で `build()` を呼ぶので、走らせるだけなら呼ばなくてよい。走らせる前に構造を確かめたいときに呼ぶ。

```python
--8<-- "examples/branching.py:build"
```

`build()` はネットワークを凍結する。その後は木の中のどの Port にも `connect` できず（`RuntimeError`）、より大きなネットワークの一部にもできない。そのため、呼ぶのは走らせるネットワークだけにする。構造が不正なら、`build()` が理由を添えて `ValueError` を送出する。`Graph` と不正の原因の一覧は API リファレンスにある。
