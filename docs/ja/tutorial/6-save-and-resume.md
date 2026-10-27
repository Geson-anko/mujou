# 💾 6. 保存と再開

実行の状態を取得して保存し、新しい実行でその続きから再開する。状態が何を持つかは [Mujou の仕組み](../concepts.md#7-%E7%8A%B6%E6%85%8B)を参照。

## 手順

```python
--8<-- "examples/save_and_restore.py"
```

1. ノードは状態を属性に持つ。`self.x`（テンソル）と `self.readout`（モジュール）は、ノードの乱数生成器の状態と一緒に自動で集まる
2. 状態は `__init__` で作り、`on_update` ではその場で更新する。こうすれば、optimizer などのほかの参照も状態を指したままになる（[2. ノードを書く](2-node.md)）
3. ホストは `run.state_dict()` で木を取得する。実行中でも、正常に終了した後（ここでは `with` ブロックの後）でもよい
4. 木を `torch.save` で保存する
5. 再開するときは同じ構成の**新しいインスタンス**を作り、木を `mujou.run(net, state=tree)` に渡す。ネットワークは 1 回だけ走るので、木を取得したネットワークは使えない（[ノードが守る規則](../concepts.md#%E3%83%8E%E3%83%BC%E3%83%89%E3%81%8C%E5%AE%88%E3%82%8B%E8%A6%8F%E5%89%87)）

## 状態の木

木は CPU 上のコピーからなる素の `dict` で、ノードのパスをキーにする。各ノードの状態、各 Channel の最新のメッセージとその時刻、各ノードの最後の更新の時刻、タイマーの時刻を持つ。正確な形は API リファレンスの [`Run.state_dict`](../api/run.md#mujou.Run.state_dict) にある。

取得する間は実行が一時停止する。各ノードは更新の途中では止まらず、2 回の更新の間で止まるので、木の中の値はすべて同じ時刻のものになる。止まっていた時間は以後の `dt` に入らない。この一時停止の前後で各ノードの `on_pause` と `on_resume` が呼ばれ、その間の I/O を止められる。タイマーの一時停止（`run.timer.pause()`）ではどちらも呼ばれない。

| 取得するとき         | 得られるもの                                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| ネットワークの実行中 | その時刻の状態                                                                                                    |
| 正常に終了した後     | `on_stop` の後にノードが残した値。時刻は `shutdown` が終了を求めた時点                                            |
| 例外で終了した後     | 取得できず、`state_dict` は `NetworkError` を送出する。残したい状態は実行中に取得し、長い実行では定期的に取得する |

## 再開するときの注意

- **再開した実行は専用のタイマーで走る。** 木の時刻と時間の速さから始まる新しい `Timer`（`run.timer`）で走るので、`mujou.time.global_timer` やほかの実行の時刻は戻らない。`state=` と一緒に `timer=` を渡すと `ValueError` になる。`global_timer.pause()` では再開した実行は止まらないので、その実行の `run.timer` を止める
- **同じネットワークを作る。** 再開するネットワークは、ノードのパス・Port・接続が木を取得したネットワークと同じでなければならない。違えば `mujou.run` は、足りない項目と余分な項目を列挙して `ValueError` を送出する。属性名を変えるとパスが変わるので、変える前に保存した木では再開できない。どちらもまだ何も起動していないので、木を直せば同じインスタンスで再開できる
- **ノードごとの状態をそろえる。** ノードの状態のキーや shape が違うと、ノードの起動中に既定の `load_state_dict` が `ValueError` を送出する。そのため `mujou.run` は `NetworkError`（`__cause__` がその `ValueError`）を送出し、そのインスタンスはもう走らない。新しく作り直す
- **入力。** `state=` と `inputs=` を両方渡すと、`inputs` の値が木の値を置き換える

## 自分の値を状態に入れる

自動で集まるのは、テンソル、`mujou.Stateful` のオブジェクト（どの `nn.Module` も、`pc.clock.LearnClock` も、`mujou.Stateful` を継承した自作のクラスも）、`self.generator` の状態。数値・文字列、テンソルのリストや dict は集まらない。`state_dict` と `load_state_dict` という名前のメソッドを持つだけのオブジェクトも集まらないので、`mujou.Stateful` を継承する。実行中に変わる素の値は、`__init__` で属性に代入して登録する。

```python
self.elapsed = 0.0
self.register_state("elapsed")
```

値はそのまま状態に入り、後で属性に何を代入しても状態に残る。登録できるのは代入済みの属性だけ（そうでなければ `AttributeError`）。値は既定の `torch.load` が読める型（`int`・`float`・`bool`・`str`・`None` とそのリスト・タプル・dict、テンソル）にする。

`deque` などそれ以外のものは、`state_dict` と `load_state_dict` の両方を override する。`super().state_dict()` の戻り値を拡張し、`super().load_state_dict` には自分で足したキーを除いて渡す。

```python
from collections import deque
from collections.abc import Mapping
from typing import Any, override

import torch

from mujou import Node


class Recent(Node):
    """Keeps the last 8 inputs in a deque."""

    def __init__(self) -> None:
        super().__init__()
        self.recent: deque[torch.Tensor] = deque(maxlen=8)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "recent": list(self.recent)}

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.recent = deque((x.clone() for x in state["recent"]), maxlen=8)  # 木の値は写して持つ
        super().load_state_dict({k: v for k, v in state.items() if k != "recent"})
```

`state_dict` はノードが一時停止している間に呼ばれる。状態の取得がほかのノードから見えないよう、その中では Port が受信も送信も断る。戻り値は mujou がコピーする。`load_state_dict` は `on_start` の後、最初の `on_update` の前に 1 回呼ばれる。

## ファイル

- `torch.save` と、既定の `weights_only=True` の `torch.load` は、テンソル・数値・文字列・dict・list・tuple からなる木を読み書きできる。mujou が木に入れるのはこれだけ。独自クラスのオブジェクトが入った木（`register_state` や override から）は、この方法では読めない
- `pickle.load` と `torch.load(..., weights_only=False)` は、ファイルの中のコードを実行できる。信頼できないファイルをこれらで読まない
- 既定の `load_state_dict` は各テンソルをノードのデバイスへコピーするので、CUDA のノードから保存した木も GPU の無いマシンで読める。ただし乱数生成器の状態は CPU と CUDA で違うので、木を復元できるのは保存したノードと同じ種類のデバイスのノードだけ

## 複数の実行を同じ時刻で

`run.state_dict()` は 1 つの実行の状態を 1 つの時刻で取得する。1 つのタイマーで走る複数の実行の状態を同じ時刻で取得するには、先にタイマーを止める。手順は[時間](../topics/time.md#%E6%99%82%E9%96%93%E3%82%92%E6%AD%A2%E3%82%81%E3%82%8B)。

## 次へ

- 各部分の詳細はトピックへ。まずは [Port と Channel](../topics/ports.md)
