# ノード

ノードは mujou の計算の単位である。利用者は `mujou.Node` を継承したクラスとしてノードを書く。

## 内側は同期、外側は非同期

ノードの内側では、普通の PyTorch のコードを同期的に実行する。行列演算でも、ノードの中に閉じた BP でもよい。

ノード同士は、[Port](ports.md) を通して値を読み書きするだけでやり取りする。ノードは隣のノードを待たない。読むのはそのときの最新の値である。隣のノードが新しく書いていなければ前と同じ値を読み、まだ一度も書いていなければ初期値を読む。

ノードは次の 3 つを守る。

- ノードをまたいで計算グラフをつなげない。ノードをまたぐ `backward()` は書けない。送った値は送るときに計算グラフから切り離されるので、勾配は Port を通らない
- ほかのノードやホストとは、自分の Port だけでやり取りする。hook の中から `mujou.run` や `Run` のメソッド（`write`・`read`・`state_dict` など）を呼ぶと `RuntimeError` になる
- 時間は読むだけにする。時刻は `self.timer.now()` で読める。hook の中から時間を動かす操作（`mujou.time.sleep`・`mujou.time.set_time_scale` など）を呼ぶと `RuntimeError` になる（[グローバル時間](time.md)）

## ノードの大きさ

1 つのノードにどれだけの計算を入れるかは、ノードのクラスの書き方で決まる。設定を変えてノードを分割したり統合したりはできない。

| 大きさ   | 例                                           |
| -------- | -------------------------------------------- |
| レイヤー | 全結合層 1 つを 1 ノードにする。既定の考え方 |
| スタック | 複数の層、またはモデル全体を 1 ノードにする  |

非同期の境界はノードの境界と一致する。複数の層を互いに待ち合わせて同期的に計算したいなら、それらを 1 つのノードのクラスにまとめる。

## ライフサイクル hook

ノードは、mujou が呼ぶメソッド（hook）を override して書く。ループは mujou が持ち、ノードの作者は 1 ステップ分の処理を `on_update(dt)` に書く。自分で `while` ループを書かない。

hook は次の順に呼ばれる。

1. `on_thread_start()`: 実行の開始時に 1 回。このノードのスレッドだけに効く設定（例: torch のスレッドごとの設定）をする（[ノードの中の PyTorch](../user-guide/torch.md)）
2. `on_start()`: 実行の開始時に 1 回。状態のテンソルを確保し、初期化する
3. `load_state_dict(state)`: 状態を復元して走らせるときだけ、`on_start` の後に 1 回（[状態](state.md)）
4. `on_update(dt)`: 繰り返し呼ばれる。Port から隣の最新値を読み、`dt` 秒ぶん状態を更新し、結果を Port に書く
5. `on_exception()`: どれかのノードの hook が例外を投げたときだけ、全ノードに 1 回。アクチュエータの安全停止などを書く
6. `on_stop()`: 実行の終了時に 1 回。例外で終わったときも呼ばれる。I/O などの資源を解放する

次の hook は、上の順とは別に、必要なときに呼ばれる。

- `on_pause()`: ネットワークが停止したとき。実行中の `on_update` が終わった後に呼ばれる。停止は状態の取得などのときに起きる（[状態](state.md)）
- `on_resume()`: 停止から再開したとき。次の `on_update` の前に呼ばれる
- `state_dict()`: 状態を取得するとき

`on_update` は、既定ではできるだけ速く繰り返される。`super().__init__(period=0.01)` のように周期を渡すと、平均してその秒数に 1 回呼ばれる。周期を渡しても、`dt` は前回の更新から実際に経った時間である（[グローバル時間](time.md)）。

どのノードのどの hook が例外を投げても、ネットワーク全体が終了する。例外を投げたノードだけを止めて、ほかのノードを動かし続けることはない。

`__init__` では Port を宣言する。状態のテンソルは `on_start` で作り、その後は同じテンソルをその場で更新するのが基本である。

## 持たない hook

エポックの開始・終了の hook、学習と推論のモードを切り替える hook は無い。メッセージを受け取るたびに呼ばれる hook も無い。ノードは受け取りを待たず、`on_update` のたびに最新の値を読みに行く。

## 例

この例のノードは、前のノードから `x` を受け取り、状態 `h` を `x @ w` に向けて時間とともに近づけ、次のノードへ送る。

```python
import torch

import mujou


class Leaky(mujou.Node):
    def __init__(self, n_in: int, n_out: int) -> None:
        self.prev = mujou.PrevPort(recv=mujou.TensorSpec((32, n_in)))   # 受け取るだけ
        self.next = mujou.NextPort(send=mujou.TensorSpec((32, n_out)))  # 送るだけ
        self.n_in, self.n_out = n_in, n_out

    def on_start(self) -> None:
        # 状態のテンソルは on_start で作る。Tensor の属性は状態として自動で集まる
        self.w = torch.randn(self.n_in, self.n_out) * 0.1
        self.h = torch.zeros(32, self.n_out)

    def on_update(self, dt: float) -> None:
        x = self.prev.recv()                  # 前のノードの最新値。待たない
        self.h += dt * (x @ self.w - self.h)  # dt に比例して、その場で更新する
        self.next.send(self.h)
```

ノードの書き方の詳細は[ノードを書く](../user-guide/writing-a-node.md)にある。
