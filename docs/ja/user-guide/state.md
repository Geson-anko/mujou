# 状態の保存と復元

このページでは、走っているネットワークの状態を取って保存し、後でその状態から走らせ直す手順を書く。状態の木の中身と、取得の間に時間が止まる仕組みは[状態](../concepts.md#%E7%8A%B6%E6%85%8B)にある。

## 基本の手順

次のサンプルは、ネットワークを 0.2 秒走らせて状態を取り、pickle で保存し、新しいインスタンスに復元して続きを走らせる。

```python
--8<-- "examples/save_and_restore.py"
```

手順は次のとおりである。

1. ノードは、状態にする値を `torch.Tensor` か `nn.Module` の属性に持つ。`mujou.Node` が自動で状態に集める（サンプルの `self.x` と `self.readout`）
2. 状態のテンソルは `on_start` で作り、`on_update` ではその場で更新する（`+=`、`add_`、`copy_` など）
3. ホストが `run.state_dict()` で状態の木を取る。実行中でも、with 文を抜けた後（正常に終了した後）でも取れる
4. 木を pickle などで保存する。保存の方法はホストが決める
5. 復元するときは、同じ構成の**新しいインスタンス**を作り、`mujou.run(net, state=木)` に渡す

手順 2 でその場で更新するのは、既定の `load_state_dict` が、`on_start` で作ったテンソルにその場でコピーして状態を戻すからである。その場で更新すれば、ほかの属性やローカル変数に取っておいた同じテンソルへの参照も、復元した値を指す。

手順 5 で新しいインスタンスを作るのは、1 つのネットワークは 1 回しか走らせられないからである。木を取ったネットワークをもう一度 `mujou.run` に渡すと `RuntimeError` になる。

## 復元するときの注意

- **タイマーの時刻が木の時刻に戻る。** 既定のグローバルタイマーで走らせた場合、復元するとグローバル時刻（`mujou.time.time()`）が木を取った時刻に戻る。`mujou.run(..., state=tree)` が戻った時点で、時刻はすでに戻っている。復元の前に読んだ `mujou.time.time()` の値と、復元の後の時刻を比べてはならない。経過時間は、復元の後に読んだ時刻同士の差で測る
- **同じタイマーでほかの実行が走っている間は、復元できない。** そのとき `mujou.run` は何も起動せずに `ValueError` を送出する。先にほかの実行を終了させるか、復元する実行に自分の `mujou.time.Timer()` を渡す（[時間の制御](time.md)）
- 復元するネットワークは、木を取ったネットワークとノードのパスと接続が同じでなければならない。違えば `mujou.run` は `ValueError` を送出する
- `state=` と `inputs=` を両方渡すと、`inputs` の値が木の中の入力の値を上書きする

## 例外で終了したときは保存できない

例外で終了した実行では、`run.state_dict()` は `NetworkError` を送出し、状態を返さない（[例外と終了](errors.md)）。長く走らせる実行で状態を失いたくなければ、実行中に定期的に `run.state_dict()` を取って保存する。

## 自動で集まらないものを状態に入れる

自動で集まるのは、属性に直接代入した `torch.Tensor` と `nn.Module` だけである（[Mujou の仕組み](../concepts.md#%E7%8A%B6%E6%85%8B)）。次のものは自動では集まらない。

- 数値の属性（ステップ数のカウンタなど）
- Tensor のリストや dict
- 乱数生成器 `self.generator` の状態
- optimizer の状態（モーメントなど）と、`.grad` に溜めた勾配

これらを状態に入れるには、`state_dict` と `load_state_dict` の両方を override する。`super()` を呼べば、自動で集めた分はそのまま扱える。数値なら override せずに、`on_start` で 0 次元の Tensor として作り（`self.count = torch.zeros((), device=self.device)`）、その場で更新してもよい（`self.count += 1`）。そうすれば自動で集まる。

次の例では、乱数生成器の状態を足す。生成器は起動のたびに `seed` から作り直される。そのため生成器の状態を足さないと、復元した後の乱数列は止めずに続けたときと違う列になる。

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

| hook                     | 守ること                                                                                                                                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `state_dict()`           | pickle できるオブジェクトを返す。ノードのスレッドではなく別のスレッドで呼ばれるので、Port の `recv` と `send` を呼ばない（`RuntimeError` になる）。更新中のテンソルをそのまま入れてよい。mujou が木をコピーする |
| `load_state_dict(state)` | `state` は `state_dict()` が返したもののコピーである。`on_start` の後、最初の `on_update` の前に、ノードのスレッドで呼ばれる                                                                                    |

`on_stop` では状態を変えない。正常に終了した後の `run.state_dict()` は、`on_stop` が終わった後のノードから状態を取るからである（理由は[状態](../concepts.md#%E7%8A%B6%E6%85%8B)）。

## 保存の形式

木は入れ子の dict で、pickle でも `torch.save` でも保存できる。自分で使うファイルなら、どちらでもよい（上の例は pickle を使っている）。木を他人と共有するなら `torch.save` で保存し、相手が既定の `torch.load` で読めるようにする。

読むときは、保存したときと同じ方法を使う（`pickle.dumps` なら `pickle.loads`、`torch.save` なら `torch.load`）。保存の方法が分からないファイルは、まず既定の `torch.load` で読む。それが失敗したら、送り手を信頼できるときだけ pickle で読む。

- `torch.load` は既定（`weights_only=True`）で、Tensor・数値・文字列・dict・list・tuple だけからなる木を読める。`state_dict` を override して独自のクラスのオブジェクトを木に入れると、既定の `torch.load` では読めない
- `pickle.load` と `torch.load(..., weights_only=False)` は、読むときにファイルの中のコードを実行できる。信頼できないファイルをこれらで読まない
- CUDA で計算するノードの状態は、CUDA の Tensor のまま木に入る。GPU の無いマシンで読むときは `torch.load(..., map_location="cpu")` を使う。既定の `load_state_dict` はノードのデバイスへコピーするので、CPU の Tensor のまま `state=` に渡してよい
