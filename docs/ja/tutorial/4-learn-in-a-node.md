# 🔥 4. ノードの中で学習する

ノードの内側の誤差逆伝播法（BP）で学ぶノードを書く。学ぶのは、実行中に目標が変わっていくストリームから。学習の段階も推論の段階も無く、ノードは新しいバッチが来るたびに予測し、学ぶ。

## プログラム

教師のノードが、ホストの書く `a` を使って $y = a x + b$ のバッチを 10 ms ごとに送る。ホストは実行の途中で `a` を変える。学習するノードの損失は下がり、目標が変わると跳ね上がり、また下がる。

```sh
uv run python examples/online_learning.py
```

```text
t = 0.48 s: loss = 0.000090  (learned a = 2.0)
t = 0.49 s: loss = 3.405589  (a changed at 0.48 s)
t = 0.97 s: loss = 0.000095  (learned a = -1.0)
the learner's weight = -0.988, bias = 0.498
```

## 学習するノード

```python
--8<-- "examples/online_learning.py:node"
```

- **モジュールは状態。** `self.model` はどの `nn.Module` とも同じく `mujou.Stateful` なので、テンソルの属性と同じく自動で状態に入る
- **seed。** `seed=0` はノード専用の乱数生成器 `self.generator` の seed。`nn.Linear` の既定の初期化は全ノードが共有する生成器を使うので、重みは `self.generator` で初期化し直す
- **素の値は `register_state` で状態に加える。** `last` と `elapsed` は実行中に変わるが、テンソルでもモジュールでもない。登録しておけばほかと一緒に保存・復元される
- **新しいかどうか。** `recv_with_time()` はバッチを送られた時刻と一緒に返す。教師が最初に送るまで時刻は `None`、次に送るまで時刻は同じなので、学習するノードは各バッチから 1 回だけ学ぶ
- **勾配は通常の PyTorch と同じ。** hook は通常の PyTorch と同じく勾配を記録する。`loss.backward()` が `p.grad` を埋め、パラメータは `torch.no_grad()` の中でその場で更新する。送るためだけの値（学習前の損失）もグラフは要らない
- **バッチの数ではなく時間で。** バッチごとの固定の刻みではなく、前に学んでからの時間（`elapsed`）だけ勾配流 $\frac{dW}{dt} = -\text{rate}\,\frac{\partial L}{\partial W}$ をたどる。どれだけ学ぶかは、ノードが何回回ったかではなく時間で決まる。長い `elapsed` は `MAX_STEP` 以下の刻みに分け、陽的 Euler 法が発散しないようにする

BP はノードの中に閉じる。Port は計算グラフから切り離したコピーを送るので、勾配がノードをまたぐことはない。

## torch の設定

```python
--8<-- "examples/online_learning.py:threads"
```

torch のスレッド数のようなプロセス全体の設定は、ホストが実行の前に決める。ノードが変えてよいのは自分のスレッドの設定だけで、hook を抜ける前に戻す（[スレッドとデバイス](../topics/threads-and-devices.md)）。

## 実行

```python
--8<-- "examples/online_learning.py:run"
```

`run.wait_for` は条件を満たすまで損失の Port を追う。終わった後は、ノードの属性をホストが読んでよい。`learner.model` が学習したモジュールである。

## 次へ

- ノードをまたいで学ぶ則と、経路に沿って運ぶ目標: [更新則](../topics/update-rules.md)
- PC-ALM、`mujou.pc` のノード: [5. PC-ALM で学習する](5-learn-with-pc.md)
