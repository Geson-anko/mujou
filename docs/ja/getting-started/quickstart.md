# クイックスタート

最小のプログラムを走らせ、出力を読む。このプログラムは 2 つのノードをつないで $y = 2x + 1$ を計算する。

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

リポジトリのルートで次のように走らせる。

```sh
uv run python examples/quickstart.py
```

次のような 1 行が表示される。

```text
y = tensor([3., 5., 7.]) at t = 0.017 s
```

`y` は `x = [1, 2, 3]` に対する $2x + 1$ である。`t` は `shift` が `y` を送った時刻で、単位は mujou のグローバル時間の秒である。`t` の値は実行ごとに変わる。

## 何が起きたか

- `Scale` と `Shift` はノードで、`mujou.Node` を継承したクラスである。それぞれ `__init__` で Port を宣言し、1 回分の処理を `on_update(dt)` に書く。どのサブクラスでもするように、最初に `super().__init__()` と `super().on_update(dt)` を呼ぶ
- `Affine` はネットワークである。ノードを属性として持ち、`connect` でつなぐ。どこにもつながっていない `scale` の `prev` を入力として、`shift` の `next` を出力として公開する
- `mujou.run(net)` はノードを動かし始める。各ノードは自分のスレッドで `on_update` を繰り返す。ノードが動き出すと `mujou.run` は戻る。`on_update` の中の `recv()` は待たずに最新の値を読むので、`x` が `scale` を通り抜けるまで、`shift` は前の入力（最初はゼロ）から計算した値を送り続ける
- そのため、`run.write` の直後に読んだ出力は、まだ古いことがある。`run.write` で `x` を書き、`run.wait_for` で出力が条件を満たすまで待つ。実時間で 5 秒たっても満たさなければ `TimeoutError` になる。`with` ブロックを抜けると、ネットワークが止まる

## 次に読むもの

- ノード・Port・ネットワーク・時間・状態の考え方: [Mujou の仕組み](../concepts.md)
- 自分のノードを書く: [ノードを書く](../user-guide/writing-a-node.md)。走らせ方と `Run` のほかのメソッド: [ネットワークを走らせる](../user-guide/running.md)
- ほかのプログラム: [サンプル](../examples/index.md)
