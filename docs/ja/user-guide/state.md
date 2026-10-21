# 💾 状態の保存と復元

実行中のネットワークの状態を取得・保存し、後でその状態から再開する手順を説明する。

状態の木の中身と、取得中に時間が止まる仕組みは[状態](../concepts.md#%E7%8A%B6%E6%85%8B)を参照。

## 基本の手順

次のサンプルは、ネットワークを 0.2 秒走らせて状態を取得し、`torch.save` で保存してから、新しいインスタンスに復元して続きを走らせる。

```python
--8<-- "examples/save_and_restore.py"
```

手順は次のとおり。

1. ノードは状態にする値を `torch.Tensor` か `nn.Module` の属性に持つ。`mujou.Node` が自動で状態に集める（サンプルの `self.x` と `self.readout`）
2. 状態のテンソルは `on_start` で作り、`on_update` ではその場で更新する（`+=`、`add_`、`copy_` など）
3. ホストが `run.state_dict()` で状態の木を取得する。実行中でも、with 文を抜けた後（正常に終了した後）でも取得できる
4. 木を `torch.save` などで保存する。保存方法はホストが決める
5. 復元するときは同じ構成の**新しいインスタンス**を作り、`mujou.run(net, state=tree)` に渡す

既定の `load_state_dict` は `on_start` で作ったテンソルへその場でコピーして状態を戻すので、手順 2 では更新もその場で行う。そうすれば、ほかの属性やローカル変数に取っておいた同じテンソルへの参照も復元した値を指す。

1 つのネットワークは 1 回しか走らせられないので、手順 5 では新しいインスタンスを作る。木を取得したネットワークをもう一度 `mujou.run` に渡すと `RuntimeError` になる。

## 復元するときの注意

- **復元した実行は専用のタイマーで走る。** 木の時刻と時間の速さから始まる新しい `Timer`（`run.timer`）で走るので、グローバルタイマーやほかの実行の時刻は戻らない。`state=` と一緒に `timer=` を渡すと `ValueError` になる。復元した実行は `mujou.time.global_timer.pause()` では止まらないので、その実行の `run.timer` を止める（[時間の制御](time.md)）
- 復元するネットワークは、ノードのパスと接続が木を取得したネットワークと同じでなければならない。違えば `mujou.run` は `ValueError` を送出する
- `state=` と `inputs=` を両方渡すと、`inputs` の値が木の中の入力の値を上書きする
- 経路に沿って運ぶ付随データ（[経路に沿ってデータを運ぶ](building-a-network.md#%E7%B5%8C%E8%B7%AF%E3%81%AB%E6%B2%BF%E3%81%A3%E3%81%A6%E3%83%87%E3%83%BC%E3%82%BF%E3%82%92%E9%81%8B%E3%81%B6)）も木に入る。それを運ぶメッセージと一緒に入り、`recv_carry` を持つ Port ごとに、最後に受け取ったメッセージの付随データも入る。復元したノードは、止めずに回した場合と同じデータを中継する

## 複数の実行の状態を同じ時刻で取る

`run.state_dict()` は 1 つの実行の状態を 1 つの時刻で取得する。複数の実行が 1 つのタイマーを共有しているとき（既定ではどの実行もグローバルタイマーで走る）、状態を 1 つずつ取得するとその間にも時間が進むので、後で取得した状態ほど時刻が遅れる。

すべてを同じ時刻で取得するには、先にタイマーを止めて状態を取得し、それから再開する。

```python
--8<-- "examples/pause.py:pause"
```

タイマーが止まっている間も `run.state_dict()` は取得でき、どの木も止めた時刻を持つ。止まっている間はノードが更新せず、止まっていた時間は `resume` 後の `dt` に入らない（[時間を止める](time.md#%E6%99%82%E9%96%93%E3%82%92%E6%AD%A2%E3%82%81%E3%82%8B)）。

!!! tip "resume は finally に書く"

    例外で時間が止まったままにならないよう `resume` は `finally` に書く。

復元した実行は、それぞれ専用のタイマーで木の時刻から続く。そのため、複数の木を同時に復元できる。ただし、復元した実行どうしはもう時間軸を共有しない。それぞれのタイマーは少しずつずれていき、1 つを止めてもほかは止まらない。復元した後も 1 本の時間軸に載せたい実行は、複数の部品を持つ 1 つの実行（`mujou.run(a, b)`）として走らせる。その 1 つの木は 1 つのタイマーに復元される。

## 例外で終了したときは保存できない

例外で終了した実行では `run.state_dict()` は `NetworkError` を送出し、状態を返さない（[例外と終了](errors.md)）。

!!! tip "長い実行では定期的に保存する"

    長く走らせる実行で状態を失いたくなければ、実行中に定期的に `run.state_dict()` で取得して保存する。

## 自動で集まらないものを状態に入れる

自動で集まるのは属性に直接代入した `torch.Tensor` と `nn.Module` だけである（[Mujou の仕組み](../concepts.md#%E7%8A%B6%E6%85%8B)）。次のものは自動では集まらない。

- 数値の属性（ステップ数のカウンタなど）
- テンソルのリストや dict
- 乱数生成器 `self.generator` の状態
- optimizer の状態（モーメントなど）と、`.grad` に溜めた勾配

数値・文字列・`None` とそのリストや dict は、属性に代入してから `register_state` で名前を登録すれば状態に入る。後で何を代入しても `del` するまで状態に残る。

```python
self.elapsed = 0.0
self.register_state("elapsed")
```

既定の `load_state_dict` は保存した値のコピーを属性に代入する。登録できるのは代入済みの属性だけで、登録は `__init__` か `on_start` で行う。復元では新しいインスタンスを使うので、それより後（`on_update` の中など）で初めて登録した名前は復元先でまだ登録されておらず、復元が失敗する。既定の `torch.load` で読むなら値は基本的な型に限る（[保存の形式](#%E4%BF%9D%E5%AD%98%E3%81%AE%E5%BD%A2%E5%BC%8F)）。

それ以外のものは `state_dict` と `load_state_dict` の両方を override する。`super()` を呼べば、自動で集めた分と登録した分はそのまま扱える。

次の例では乱数生成器の状態を足す。復元では新しいインスタンスを使い生成器も `seed` から作り直すので、生成器の状態を足さないと、復元後の乱数列が止めずに続けたときと違う列になる。

```python
from typing import Any, override

from mujou import Node


class Noisy(Node):
    @override
    def state_dict(self) -> Any:
        state = super().state_dict()
        state["generator"] = self.generator.get_state()
        return state

    @override
    def load_state_dict(self, state: Any) -> None:
        state = dict(state)  # 渡された木を書き換えないよう、写してから取り出す
        self.generator.set_state(state.pop("generator"))
        super().load_state_dict(state)
```

override するときは次を守る。

| hook                     | 守ること                                                                                                                                                                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `state_dict()`           | pickle できるオブジェクトを返す。この中で Port の `recv` と `send` を呼ばない（スロットは木に別に入り、送ると状態を取る最中の実行を変えてしまう）。更新中のテンソルをそのまま入れてよい（mujou が木をコピーする） |
| `load_state_dict(state)` | `state` は `state_dict()` が返したもののコピー。`on_start` の後、最初の `on_update` の前に呼ばれる                                                                                                                |

!!! warning "on_stop で状態を変えない"

    `on_stop` では状態を変えない。正常に終了した後の `run.state_dict()` は `on_stop` の後にノードが残した状態を写し、時刻は `shutdown` が終了を求めた時点にする。`on_stop` で変えた値は木に入ってしまう（理由は[状態](../concepts.md#%E7%8A%B6%E6%85%8B)）。

## 保存の形式

`run.state_dict()` が返す木は `mujou.StateDict` である。`dict` の派生なので dict と同じようにキーで引け、pickle でも `torch.save` でも保存できる。読み込んだ木も `mujou.StateDict` に戻る。

自分で使うファイルならどちらでもよい。木を他人と共有するなら `torch.save` で保存し、相手が既定の `torch.load` で読めるようにする。`mujou.run(..., state=)` には `mujou.StateDict` のほか、同じ中身の素の dict も渡せる。

読むときは保存時と同じ方法を使う（`pickle.dumps` なら `pickle.loads`、`torch.save` なら `torch.load`）。保存方法が分からないファイルは、まず既定の `torch.load` で読む。それが失敗したら、送り手を信頼できるときだけ pickle で読む。

- `torch.load` は既定（`weights_only=True`）で、テンソル・数値・文字列・dict・list・tuple だけからなる木を読める。`mujou.StateDict` も `import mujou` した後なら読める。登録した属性や `state_dict` の override から独自クラスのオブジェクトが木に入ると、既定の `torch.load` では読めない
- `pickle.load` と `torch.load(..., weights_only=False)` は、読むときにファイルの中のコードを実行できる。信頼できないファイルをこれらで読まない
- CUDA で計算するノードの状態は CUDA のテンソルのまま木に入る。`tree.cpu()` は木の中のテンソルをすべて CPU に移した新しい木を返す（元の木は変えない）。保存前に CPU に移せば GPU の無いマシンでも読める。CUDA のまま保存したファイルは `torch.load(..., map_location="cpu")` で読む。既定の `load_state_dict` はノードのデバイスへコピーするので、CPU のテンソルのまま `state=` に渡してよい

```python
torch.save(run.state_dict().cpu(), "state.pt")  # GPU のノードの状態も CPU のテンソルで保存する
tree = torch.load("state.pt")  # mujou.StateDict
```

`tree.to(device)` は木のテンソルをすべて `device` に移す。`cpu()` は `to("cpu")` と同じ。

辿るのは木の中の dict（`nn.Module` の `state_dict` の `OrderedDict` も含む）・list・tuple で、これらの入れ物は新しく作る。テンソル以外の値は元の木と共有する。`torch.Tensor.to` と同じく、すでに `device` にあるテンソルはコピーせず元の木と共有する。
