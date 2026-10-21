# 💾 状態の保存と復元

実行中のネットワークの状態を取得・保存し、後でその状態から再開する手順を説明する。

状態の木の中身と、取得中に時間が止まる仕組みは[状態](../concepts.md#%E7%8A%B6%E6%85%8B)を参照。

## 基本の手順

次のサンプルは、ネットワークを 0.2 秒走らせて状態を取得し、`torch.save` で保存してから、新しいインスタンスに復元して続きを走らせる。

```python
--8<-- "examples/save_and_restore.py"
```

手順は次のとおり。

1. ノードは状態にする値を `torch.Tensor` か `nn.Module` の属性に持つ。`mujou.Node` が、乱数生成器の状態と一緒に自動で状態に集める（サンプルの `self.x` と `self.readout`）
2. 状態のテンソルは `__init__` で作り、`on_update` ではその場で更新する（`+=`、`add_`、`copy_` など）
3. ホストが `run.state_dict()` で状態の木を取得する。実行中でも、with 文を抜けた後（正常に終了した後）でも取得できる
4. 木を `torch.save` などで保存する。保存方法はホストが決める
5. 復元するときは同じ構成の**新しいインスタンス**を作り、`mujou.run(net, state=tree)` に渡す

既定の `load_state_dict` はノードが持つテンソルへその場でコピーして状態を戻すので、手順 2 では更新もその場で行う。そうすれば、ほかの属性やローカル変数に取っておいた同じテンソルへの参照も復元した値を指す。

1 つのネットワークは 1 回しか走らせられないので、手順 5 では新しいインスタンスを作る。木を取得したネットワークをもう一度 `mujou.run` に渡すと `RuntimeError` になる。

## 復元するときの注意

- **復元した実行は専用のタイマーで走る。** 木の時刻と時間の速さから始まる新しい `Timer`（`run.timer`）で走るので、グローバルタイマーやほかの実行の時刻は戻らない。`state=` と一緒に `timer=` を渡すと `ValueError` になる。復元した実行は `mujou.time.global_timer.pause()` では止まらないので、その実行の `run.timer` を止める（[時間の制御](time.md)）
- 復元するネットワークは、ノードのパス・Port・接続が木を取得したネットワークと同じでなければならない。違えば `mujou.run` は、足りない項目と余分な項目を列挙して `ValueError` を送出する。各ノードの状態も合っていなければならない。キーや shape が違えば、既定の `load_state_dict` が `ValueError` を送出する
- `state=` と `inputs=` を両方渡すと、`inputs` の値が木の中の入力の値を上書きする

## 複数の実行の状態を同じ時刻で取る

`run.state_dict()` は 1 つの実行の状態を 1 つの時刻で取得する。複数の実行が 1 つのタイマーを共有しているとき（既定ではどの実行もグローバルタイマーで走る）、状態を 1 つずつ取得するとその間にも時間が進むので、後で取得した状態ほど時刻が遅れる。

すべてを同じ時刻で取得するには、先にタイマーを止めて状態を取得し、それから再開する。

```python
--8<-- "examples/pause.py:pause"
```

タイマーが止まっている間も `run.state_dict()` は取得でき、どの木も止めた時刻を持つ。止まっている間はノードが更新せず、止まっていた時間は `resume` 後の `dt` に入らない（[時間を止める](time.md#%E6%99%82%E9%96%93%E3%82%92%E6%AD%A2%E3%82%81%E3%82%8B)）。

!!! tip "resume は finally に書く"

    例外で時間が止まったままにならないよう `resume` は `finally` に書く。

復元した実行は、それぞれ専用のタイマーで木の時刻から続く。そのため、複数の木を同時に復元できる。ただし、復元した実行どうしはもう時間軸を共有しない。それぞれのタイマーは少しずつずれていき、1 つを止めてもほかは止まらない。復元した後も 1 本の時間軸に載せたい実行は、1 つの実行（`mujou.run(Group(a=a, b=b))`）として走らせる。その 1 つの木は 1 つのタイマーに復元される。

## 例外で終了したときは保存できない

例外で終了した実行では `run.state_dict()` は `NetworkError` を送出し、状態を返さない（[例外と終了](errors.md)）。

!!! tip "長い実行では定期的に保存する"

    長く走らせる実行で状態を失いたくなければ、実行中に定期的に `run.state_dict()` で取得して保存する。

## 自動で集まらないものを状態に入れる

自動で集まるのは、属性に直接代入した `torch.Tensor` と、`state_dict` と `load_state_dict` を持つオブジェクト（`nn.Module` など）、そして `self.generator` の状態である（[Mujou の仕組み](../concepts.md#%E7%8A%B6%E6%85%8B)）。次のものは自動では集まらない。

- 数値の属性（ステップ数のカウンタなど）
- テンソルのリストや dict
- `__init__` が引数から計算する設定（時定数など）。別の引数で作ったノードに復元しても、そのノードの設定のまま

実行中に変わる値（数値・文字列・`None` とそのリストや dict）は、属性に代入してから `register_state` で名前を登録すれば状態に入る。後で何を代入しても状態に残る。

```python
self.elapsed = 0.0
self.register_state("elapsed")
```

既定の `load_state_dict` は保存した値のコピーを属性に代入する。登録できるのは代入済みの属性だけで（そうでなければ `AttributeError`）、登録は `__init__` で行う。復元では新しいインスタンスを使い、その状態は同じでなければならないからである。既定の `torch.load` で読むなら値は基本的な型に限る（[保存の形式](#%E4%BF%9D%E5%AD%98%E3%81%AE%E5%BD%A2%E5%BC%8F)）。

それ以外のものは `state_dict` と `load_state_dict` の両方を override する。`super().state_dict()` の戻り値を拡張し、`super().load_state_dict` には自分で足したキーを除いて渡す。

```python
from collections.abc import Mapping
from typing import Any, override

from mujou import Node


class Recent(Node):
    """Keeps the last few inputs in a list."""

    @override
    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "recent": list(self.recent)}

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.recent = [x.clone() for x in state["recent"]]  # 木の値は写して持つ
        super().load_state_dict({k: v for k, v in state.items() if k != "recent"})
```

override するときは次を守る。

| hook                     | 守ること                                                                                                                                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `state_dict()`           | pickle できる dict を返す。この中で Port の `recv` と `send` を呼ばない（スロットは木に別に入り、送ると状態を取る最中の実行を変えてしまう）。更新中のテンソルをそのまま入れてよい（mujou が戻り値をコピーする） |
| `load_state_dict(state)` | `state` は `state_dict()` が返したもので、木から読んだもの。持ち続けるものは写す。`on_start` の後、最初の `on_update` の前に呼ばれる                                                                            |

!!! warning "on_stop で状態を変えない"

    `on_stop` では状態を変えない。正常に終了した後の `run.state_dict()` は `on_stop` の後にノードが残した状態を写し、時刻は `shutdown` が終了を求めた時点にする。`on_stop` で変えた値は木に入ってしまう（理由は[状態](../concepts.md#%E7%8A%B6%E6%85%8B)）。

## 保存の形式

`run.state_dict()` が返す木は素の `dict` で、テンソルはどれも CPU 上のコピーである（CUDA で計算するノードの状態も）。dict と同じようにキーで引け、pickle でも `torch.save` でも保存できる。木の形は API リファレンスの `Run.state_dict` にある。

自分で使うファイルならどちらでもよい。木を他人と共有するなら `torch.save` で保存し、相手が既定の `torch.load` で読めるようにする。

読むときは保存時と同じ方法を使う（`pickle.dumps` なら `pickle.loads`、`torch.save` なら `torch.load`）。保存方法が分からないファイルは、まず既定の `torch.load` で読む。それが失敗したら、送り手を信頼できるときだけ pickle で読む。

- `torch.load` は既定（`weights_only=True`）で、テンソル・数値・文字列・dict・list・tuple だけからなる木を読める。登録した属性や `state_dict` の override から独自クラスのオブジェクトが木に入ると、既定の `torch.load` では読めない
- `pickle.load` と `torch.load(..., weights_only=False)` は、読むときにファイルの中のコードを実行できる。信頼できないファイルをこれらで読まない
- 既定の `load_state_dict` はテンソルをノードのデバイスへコピーする。ただし `generator` の状態は CPU と CUDA で形式が違うので、木は保存したノードと同じ種類のデバイスのノードに復元する。CUDA のノードから保存した木は、別の GPU のマシンでも CUDA のノードに復元できる

```python
torch.save(run.state_dict(), "state.pt")
tree = torch.load("state.pt")  # 素の dict
```
