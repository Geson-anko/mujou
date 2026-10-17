# 🕸️ ネットワークを組む

ネットワークは `mujou.Network` を継承したクラスとして書く。このページは次のことを説明する。

- ノードのつなぎ方
- 外に公開する Port
- 入れ子
- 分岐と合流
- 構築した構造の確かめ方

ネットワークの考え方は[ネットワーク](../concepts.md#%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF)と [Port と Channel](../concepts.md#port-%E3%81%A8-channel) にある。

## 基本の形

`__init__` で、次の 3 つを順に書く。

1. ノード（または別のネットワーク）を属性に代入する
2. それらの Port を `connect` でつなぐ
3. 外からつなぎたい・読み書きしたい Port を、ネットワークの属性に代入して公開する

```python
--8<-- "examples/quickstart.py:network"
```

`super().__init__()` は最初に呼ぶ。ネットワークは、`nn.Module` がサブモジュールを登録するのと同じく、属性に代入されたノード・ネットワーク・Port をその時点で登録する。`super().__init__()` より前に代入すると `AttributeError` になる。

!!! warning "ノードは 1 つずつ属性に代入する"

    ノードをリストや dict に入れて持たせても登録されず、ネットワークの一部にならない。ノードは 1 つずつ属性に代入する。

## つなぐ

2 つの Port を `Port.connect` でつなぐ。`a.next.connect(b.prev)` のように書く。ノードとネットワーク自体は `connect` を持たない。ノードの入出力は複数ありうるので、Port を 2 つとも書けば、どの経路をつなぐかがいつも読み取れる。

慣習では、前のノードの側の Port を `PrevPort` にして `prev` と名付け、次のノードの側の Port を `NextPort` にして `next` と名付ける。こうすると、つないだ列は `a.next.connect(b.prev)` と読める。

名前と種類は向きを読みやすくするだけのものである。`PrevPort` と `NextPort` が `Port` と違うのは既定で取る相手の数（`PrevPort` は 1 つ、`NextPort` は複数）だけで、1 つのノードがそれぞれをいくつ持ってもよい。

つなげるのは、片方の `send` の spec がもう片方の `recv` の spec と一致し、逆向きも一致するときだけである。一致しなければ、`connect` を呼んだ時点で `TypeError` になる。相手を 1 つしか取らない Port に 2 本目の Channel をつなぐと `ValueError` になる。

`connect` は、できた接続（`Channel`）を返す。分岐で相手ごとに違う値を送るときに使う（このページの[分岐と合流](#%E5%88%86%E5%B2%90%E3%81%A8%E5%90%88%E6%B5%81)）。

## Port を公開する

ネットワークの属性に子の Port を代入すると、その Port を公開できる。

外からは短い名前で Port を指せるようになる。慣習では、最初のノードの Port を `prev`、最後のノードの Port を `next` として公開する（`self.prev = self.l1.prev`）。するとネットワークも、`x.next.connect(net.prev)`、`net.next.connect(y.prev)` とノードと同じようにつなげる。ほかの Port は `net.teacher` のように、それぞれの名前で公開する。

公開できるのは、そのネットワークの中のノードが持つ Port に限る。

ホストは `run.write` で、受け取る Port のうちつながっていないものに値を書ける。公開しているかどうかは問わない。`run.write({net.head.teacher: y})` のように、子の属性をたどって Port を指す。公開は、指す名前を短くするだけである。

## 入れ子にする

ネットワークは、別のネットワークを属性に持てる。子のネットワークが `prev` と `next` を公開していれば、ノードの Port と同じようにつなぐ。

```python
class Stack(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Hidden(784, 256)
        self.l2 = Hidden(256, 256)
        self.l1.next.connect(self.l2.prev)
        self.prev = self.l1.prev
        self.next = self.l2.next


class Classifier(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.body = Stack()
        self.head = Head()
        self.body.next.connect(self.head.prev)  # body.next の実体は l2.next
        self.prev = self.body.prev     # 子が公開した Port を、さらに公開する
        self.teacher = self.head.teacher
```

`Hidden` と `Head` は、それぞれ `PrevPort` と `NextPort` を持つノードとする。`Head` は教師信号を受け取る `Port`（`teacher`）も持つ。

ノードは、ルートからの属性名を `.` でつないだパスで区別される。この例のノードのパスは `"body.l1"`、`"body.l2"`、`"head"` である。

!!! warning "属性名を変えるとパスが変わる"

    属性名を変えるとパスが変わり、変える前に保存した状態を復元できなくなる（[状態の保存と復元](state.md)）。

ノードは、ほかのノードやネットワークを属性に持てない。代入すると `ValueError` になる。並べたいノードは、ネットワークの属性に並べる。

## 分岐と合流

1 つの Port に複数の Channel をつなぐと、分岐と合流になる。次の例は、`split` の出力を 2 つに分け、`sum` で合わせる。

```python
--8<-- "examples/branching.py:network"
```

### 分岐: 送る側

`NextPort` は既定で複数の相手を取れるので、`a.next.connect(b.prev)` と `a.next.connect(c.prev)` を続けて書けば分岐する。送り方は 2 つある。

| 送り方                    | 送り先                   | `Run.read` で読める値 |
| ------------------------- | ------------------------ | --------------------- |
| `send(value)`             | すべての相手に同じ値     | 更新される            |
| `send_to(channel, value)` | `channel` の先の相手だけ | 更新されない          |

`send_to` だけを使う Port をホストが `Run.read` で読むと、その Port の初期値（`initial`。無ければゼロ）が返る。`send_to` で送った値そのものをホストが読む方法は無い。

`Run.read` は送る型を持つ Port しか読まないので、受け取るだけの Port（各枝の `prev`）も読めない。ホストに読めるのは、枝のノードが受け取った値から計算して `send` した結果（上の例なら `net.double.next` の `2 x`）である。

`send_to` に渡す `Channel` は、送り先ごとに区別して持つ。上の例のように、ネットワークで `connect` が返す `Channel` を、送る側のノードの属性に代入する（`self.split.plus = self.split.next.connect(self.double.prev)`）。

!!! warning "Channel を順番で取り出さない"

    `self.next.channels`（つないだ順に並ぶ）から取り出すと、`connect` を書く順を入れ替えたときに、エラーも出ずに値の送り先が入れ替わる。

`Channel` を代入した属性は状態に集まらない。

### 合流: 受け取る側

`PrevPort(multi=True)` は複数の相手から受け取る。相手が複数の Port の `recv()` は、`Channel` をキーにした dict を、つないだ順に返す。まだ一度も送っていない相手の Channel の値は、相手の Port の初期値である（`initial` を宣言していなければゼロ）。

複数の値をどう束ねるか（足す、平均する、相手ごとに重みを掛ける）は、受け取るノードが決める。mujou は束ねない。上の例の `Sum` は足している。

相手ごとに扱いを変えるときは、`send_to` のときと同じく、相手ごとの `Channel` を受け取るノードに持たせる。`self.sum.from_double = self.double.next.connect(self.sum.prev)` とし、ノードの中で `self.prev.recv()[self.from_double]` と読む。

`NextPort` も既定で複数の相手を取るので、その `recv()` は dict を返す。次のノードが 1 つだけと決まっているなら、`NextPort(send=..., multi=False)` にすると `recv()` が値そのものを返す。

## 構築した構造を確かめる

`net.build()` は、構造と型を検査して凍結し、構築した構造（`Graph`）を返す。`mujou.run` も中で `build()` を呼ぶので、走らせるだけなら呼ばなくてよい。走らせる前に構造を確かめたいときに呼ぶ。

```python
--8<-- "examples/branching.py:build"
```

!!! warning "`build()` はネットワークを凍結する"

    `build()` の後は、次のことができない。

    - 木の中のどの Port にも `connect` する
    - 木のノードとネットワークに Port・ノード・ネットワークを代入する、または削除する（どちらも `RuntimeError`）
    - より大きなネットワークの一部にする

    そのため、呼ぶのは走らせるネットワークだけにする。

構造が不正なら、`build()` が理由を添えて `ValueError` を送出する。`Graph` と不正の原因の一覧は API リファレンスにある。
