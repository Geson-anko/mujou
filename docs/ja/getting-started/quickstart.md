# 🚀 クイックスタート

最小のプログラムを走らせてみる。2 つのノードをつないで $y = 2x + 1$ を計算するプログラムである。

```mermaid
flowchart LR
    H1["ホスト"] -->|"x を書く"| S["scale（2 倍）"]
    S --> F["shift（+1）"]
    F -->|"y を読む"| H2["ホスト"]
```

## プログラム

```python title="examples/quickstart.py"
--8<-- "examples/quickstart.py"
```

リポジトリのルートで次のように実行する。

```sh
uv run python examples/quickstart.py
```

次のような 1 行が表示される。

```text
y = tensor([3., 5., 7.]) at t = 0.017 s
```

`y` は `x = [1, 2, 3]` に対する $2x + 1$ である。`t` は `shift` が `y` を送った時刻で、単位は mujou のグローバル時間の秒。`t` の値は実行ごとに変わる。

## 何が起きたか

- `Scale` と `Shift` は `mujou.Node` を継承したノード。それぞれ `__init__` で Port を宣言し、1 回分の処理を `on_update(dt)` に書く。どのサブクラスとも同じく、最初に `super().__init__()` と `super().on_update(dt)` を呼ぶ
- `Affine` はネットワーク。ノードを属性として持ち、その Port 同士を `connect` でつなぐ。どこにも接続していない `scale` の `prev` を入力として、`shift` の `next` を出力として公開する
- `mujou.run(net)` はノードを動かし始め、ノードが動き出すと戻る。各ノードは自分のスレッドで `on_update` を繰り返す。`on_update` の中の `recv()` は待たずに最新の値を読むので、`x` が `scale` を通り抜けるまで `shift` は前の入力（最初はゼロ）から計算した値を送り続ける
- そのため `run.write` の直後に読んだ出力はまだ古いことがある。`run.write` で `x` を書き、出力が条件を満たすまで `run.wait_for` で待つ。実時間で 5 秒たっても満たさなければ `TimeoutError` になる。`with` ブロックを抜けるとネットワークが止まる

## 次に読むもの

- ノード・Port・ネットワーク・時間・状態の考え方: [Mujou の仕組み](../concepts.md)
- 自分のノードを書く: [ノードを書く](../user-guide/writing-a-node.md)。走らせ方と `Run` のほかのメソッド: [ネットワークを走らせる](../user-guide/running.md)
- ほかのプログラム: [サンプル](../examples/index.md)
