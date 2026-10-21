# 🕸️ ネットワークを組む

ネットワークは `mujou.Network` を継承したクラスとして書く。このページでは次のことを説明する。

- ノードのつなぎ方
- 外に公開する Port
- 入れ子
- 分岐と合流
- 経路に沿ったデータの運び方
- 構造の確かめ方

ネットワークの考え方は[ネットワーク](../concepts.md#%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF)と [Port と Channel](../concepts.md#port-%E3%81%A8-channel) を参照。

## 基本の形

`__init__` では次の 3 つを順に書く。

1. ノード（または別のネットワーク）を属性に代入する
2. それらの Port を `connect` でつなぐ
3. 外からつなぎたい Port や読み書きしたい Port をネットワークの属性に代入して公開する

```python
--8<-- "examples/quickstart.py:network"
```

`mujou.run` は走り始めるときに、ネットワークの属性を代入の順に読む。`nn.Module` がサブモジュールを見つけるのと同じく、そのうちのノードとネットワークが子になる。

!!! warning "ノードは 1 つずつ属性に代入する"

    実行はリスト・タプル・dict の中を見ない。そこにノード・ネットワーク・Port を入れると、`mujou.run` が `TypeError` を送出する。ノードは 1 つずつ属性に代入する。

## つなぐ

2 つの Port は `a.next.connect(b.prev)` のように `Port.connect` でつなぐ。ノードとネットワークは `connect` を持たない。ノードの入出力は複数ありうるので、Port を 2 つとも書くことでどの経路をつなぐかが常に読み取れる。

慣習では前のノード側の Port を `prev`、次のノード側の Port を `next` と名付ける。こうするとつないだ列が `a.next.connect(b.prev)` と読める。名前は向きを読みやすくするためだけのもので、1 つのノードが Port をいくつ持ってもよい。

接続できるのは、片方の `send` の spec がもう片方の `recv` の spec と一致し、逆向きも一致するときだけ。一致しなければ `connect` を呼んだ時点で `TypeError` になる。受け取る `Port` の相手は 1 つだけで、2 つ目の相手をつなぐと `ValueError` になる（複数の相手から受け取るには `MultiPort` を使う）。同じ 2 つの Port をもう一度つなぐと、すでにある Channel が返る。

`connect` は作った接続（`Channel`）を返す。これは分岐で相手ごとに違う値を送るときに使う（このページの[分岐と合流](#%E5%88%86%E5%B2%90%E3%81%A8%E5%90%88%E6%B5%81)）。

## Port を公開する

子の Port をネットワークの属性に代入すると、その Port を公開できる。

公開すると外から短い名前で Port を指せる。慣習では最初のノードの Port を `prev`、最後のノードの Port を `next` として公開する（`self.prev = self.l1.prev`）。するとネットワークもノードと同じように `x.next.connect(net.prev)`、`net.next.connect(y.prev)` とつなげる。ほかの Port は `net.teacher` のようにそれぞれの名前で公開する。

公開できるのはそのネットワークの中のノードが持つ Port だけ。

ホストは `run.write` で、受け取る Port のうちつながっていないものに値を書ける。公開しているかどうかは問わず、`run.write({net.head.teacher: y})` のように子の属性をたどって Port を指せばよい。公開は指す名前を短くするだけ。

## 入れ子にする

ネットワークは別のネットワークを属性に持てる。子のネットワークが `prev` と `next` を公開していれば、ノードの Port と同じようにつなぐ。

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

`Hidden` と `Head` はそれぞれ `prev` と `next` の Port を持つノードとする。`Head` は教師信号を受け取る Port `teacher` も持つ。

ノードはルートからの属性名を `.` でつないだパスで区別される。この例のノードのパスは `"body.l1"`、`"body.l2"`、`"head"` になる。

!!! warning "属性名を変えるとパスが変わる"

    属性名を変えるとパスが変わり、変える前に保存した状態を復元できなくなる（[状態の保存と復元](state.md)）。

ノードはほかのノードやネットワークを属性に持てず、代入すると `ValueError` になる。並べたいノードはネットワークの属性に並べる。

## 分岐と合流

1 つの Port に複数の Channel をつなぐと分岐と合流になる。次の例では `split` の出力を 2 つに分け、`sum` で合わせる。

```python
--8<-- "examples/branching.py:network"
```

### 分岐: 送る側

送るだけの `Port` は相手をいくつでも取れるので、`a.next.connect(b.prev)` と `a.next.connect(c.prev)` を続けて書けば分岐する。`MultiPort` も相手をいくつでも取れ、送り方が 2 つある。

| 送り方                    | 送り先                   | `Run.read` で読める値 |
| ------------------------- | ------------------------ | --------------------- |
| `send(value)`             | すべての相手に同じ値     | 更新される            |
| `send_to(channel, value)` | `channel` の先の相手だけ | 更新されない          |

`Port` の送り方は `send` だけ。

`send_to` だけを使う Port をホストが `Run.read` で読むと、その Port の初期値（`initial`。無ければゼロ）が返る。`send_to` で送った値をホストが読む方法は無い。

`Run.read` は送る型を持つ Port しか読まないので、受け取るだけの Port（各枝の `prev`）も読めない。ホストが読めるのは、枝のノードが受け取った値から計算して `send` した結果（上の例なら `net.double.next` の `2 x`）になる。

`send_to` に渡す `Channel` は送り先ごとに区別して持つ。上の例のように、ネットワークで `connect` が返す `Channel` を送る側のノードの属性に代入する（`self.split.plus = self.split.next.connect(self.double.prev)`）。

!!! warning "Channel を順番で取り出さない"

    `self.next.channels`（つないだ順に並ぶ）から取り出すと、`connect` を書く順を入れ替えたときにエラーも出ずに値の送り先が入れ替わってしまう。

`Channel` を代入した属性は状態に集まらない。

### 合流: 受け取る側

`MultiPort` は複数の相手から受け取る。その `recv()` は `Channel` をキーにした dict をつないだ順に返す。まだ一度も送っていない相手の Channel の値は、相手の Port の初期値になる（`initial` を宣言していなければゼロ）。

複数の値をどう束ねるか（足す、平均する、相手ごとに重みを掛ける）は受け取るノードが決め、mujou は束ねない。上の例の `Sum` は足している。

相手ごとに扱いを変えるときは、`send_to` と同じく相手ごとの `Channel` を受け取るノードに持たせる。`self.sum.from_double = self.double.next.connect(self.sum.prev)` とし、ノードの中で `self.prev.recv()[self.from_double]` と読む。役割の違う入力（予測と誤差など）は、代わりに入力ごとに `Port` を宣言して受け取る。

前へ送りつつ複数の次のノードからフィードバックを受け取るノードは、`next` も `MultiPort` にする。その `recv()` は dict を返す。

## 経路に沿ってデータを運ぶ

値と一緒にデータ（サンプルの ID やターゲットなど）を経路に沿って渡すときは、それをメッセージのフィールド（`DictSpec`）にし、各ノードが送るメッセージにそのフィールドを写す。経路の先のどのノードも、値を、その値の元になったサンプルのデータと組で受け取る。データを別の Port で送ると、こうはならない。2 つの Port を読むノードは、それぞれを別々の瞬間に読むので、値を別のサンプルのデータと組んでしまうことがある。`mujou.pc` は、サンプルごとのターゲットをこの形で `pc.Activity` に入れて渡す。

## 構造を確かめる

`mujou.run` は、どのノードも起動する前に構造を検査する。`super().__init__()` を呼んでいないノード、ノードを持つノード、リストに入れた部品、2 か所にある同じノードや Port、ネットワークの外の Port とつないだ Port、同じテンソルや module を持つ 2 つのノードを見つけると、属性のパス（`"body.l1.prev"`）を添えて `TypeError` か `ValueError` を送出し、何も起動しない。ネットワークを直せば、同じインスタンスをそのまま走らせられる。ノードのパスは状態の木のキーになる。

```python
--8<-- "examples/branching.py:paths"
```
