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

- **タイマーの時刻が木の時刻に戻る。** 既定のグローバルタイマーで走らせた場合、復元するとグローバル時刻（`mujou.time.now()`）が木を取得した時刻に戻る。`mujou.run(..., state=tree)` が戻った時点で時刻はすでに戻っている。復元前に読んだ `mujou.time.now()` の値と復元後の時刻を比べてはならない。経過時間は復元後に読んだ時刻同士の差で測る
- **同じタイマーでほかの実行が走っている間は復元できない。** このとき `mujou.run` は何も起動せずに `ValueError` を送出する。先にほかの実行を終了させるか、復元する実行に専用の `mujou.time.Timer()` を渡す（[時間の制御](time.md)）
- 復元するネットワークは、ノードのパスと接続が木を取得したネットワークと同じでなければならない。違えば `mujou.run` は `ValueError` を送出する
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

同じタイマーでほかの実行が走っている間は木を復元できない点に注意する。前の実行が終わってから 1 つずつ復元するか、復元する実行それぞれに専用のタイマーを渡す。どのタイマーも木の同じ時刻から始まる。

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

これらを状態に入れるには `state_dict` と `load_state_dict` の両方を override する。`super()` を呼べば、自動で集めた分はそのまま扱える。

数値なら override せずに、`on_start` で 0 次元のテンソルとして作り（`self.count = torch.zeros((), device=self.device)`）、その場で更新してもよい（`self.count += 1`）。そうすれば自動で集まる。

次の例では乱数生成器の状態を足す。復元した実行は新しいインスタンスで走り、生成器も `seed` から作り直されるので、生成器の状態を足さないと、復元後の乱数列が止めずに続けたときと違う列になる。

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

| hook                     | 守ること                                                                                                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `state_dict()`           | pickle できるオブジェクトを返す。この中で Port の `recv` と `send` を呼ばない（`RuntimeError` になる）。更新中のテンソルをそのまま入れてよい（mujou が木をコピーする） |
| `load_state_dict(state)` | `state` は `state_dict()` が返したもののコピー。`on_start` の後、最初の `on_update` の前にノードのスレッドで呼ばれる                                                   |

!!! warning "on_stop で状態を変えない"

    `on_stop` では状態を変えない。正常に終了した後の `run.state_dict()` は `on_stop` の後にノードが残した状態を写し、時刻は `shutdown` が終了を求めた時点にする。`on_stop` で変えた値は木に入ってしまう（理由は[状態](../concepts.md#%E7%8A%B6%E6%85%8B)）。

## 保存の形式

`run.state_dict()` が返す木は `mujou.StateDict` である。`dict` の派生なので dict と同じようにキーで引け、pickle でも `torch.save` でも保存できる。読み込んだ木も `mujou.StateDict` に戻る。

自分で使うファイルならどちらでもよい。木を他人と共有するなら `torch.save` で保存し、相手が既定の `torch.load` で読めるようにする。`mujou.run(..., state=)` には `mujou.StateDict` のほか、同じ中身の素の dict も渡せる。

読むときは保存時と同じ方法を使う（`pickle.dumps` なら `pickle.loads`、`torch.save` なら `torch.load`）。保存方法が分からないファイルは、まず既定の `torch.load` で読む。それが失敗したら、送り手を信頼できるときだけ pickle で読む。

- `torch.load` は既定（`weights_only=True`）で、テンソル・数値・文字列・dict・list・tuple だけからなる木を読める。`mujou.StateDict` も `import mujou` した後なら読める。`state_dict` を override して独自クラスのオブジェクトを木に入れると、既定の `torch.load` では読めない
- `pickle.load` と `torch.load(..., weights_only=False)` は、読むときにファイルの中のコードを実行できる。信頼できないファイルをこれらで読まない
- CUDA で計算するノードの状態は CUDA のテンソルのまま木に入る。`tree.cpu()` は木の中のテンソルをすべて CPU に移した新しい木を返す（元の木は変えない）。保存前に CPU に移せば GPU の無いマシンでも読める。CUDA のまま保存したファイルは `torch.load(..., map_location="cpu")` で読む。既定の `load_state_dict` はノードのデバイスへコピーするので、CPU のテンソルのまま `state=` に渡してよい

```python
torch.save(run.state_dict().cpu(), "state.pt")  # GPU のノードの状態も CPU のテンソルで保存する
tree = torch.load("state.pt")  # mujou.StateDict
```

`tree.to(device)` は木のテンソルをすべて `device` に移す。`cpu()` は `to("cpu")` と同じ。

辿るのは木の中の dict（`nn.Module` の `state_dict` の `OrderedDict` も含む）・list・tuple で、これらの入れ物は新しく作る。テンソル以外の値は元の木と共有する。`torch.Tensor.to` と同じく、すでに `device` にあるテンソルはコピーせず元の木と共有する。
