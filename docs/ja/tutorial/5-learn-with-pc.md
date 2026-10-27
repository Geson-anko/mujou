# 🧠 5. PC-ALM で学習する

`mujou.pc` は PyTorch のモデルをノードに分け、PC-ALM（Augmented Lagrangian Predictive Coding、<https://pub.sakana.ai/pc-alm/>）で学習させるモジュール。各ノードはモデルの一部を持ち、隣のノードのメッセージだけから学ぶ。平衡点では、その勾配が誤差逆伝播法のものと一致する。このページでは回帰を学習させる。決めるのは時定数、重みの optimizer、各バッチを見せる時間だけ。

## プログラム

```sh
uv run python examples/pc_regression.py
```

```text
batch   1: loss 0.4157 (the mean of the last 1)
batch  50: loss 0.0476 (the mean of the last 50)
...
batch 300: loss 0.0059 (the mean of the last 50)
```

## 鎖

```python
--8<-- "examples/pc_regression.py:chain"
```

`pc.Chain` はモジュールごとにノードを 1 つ作り、`nn.Sequential` が計算する順につなぐ。サンプルを受け取る `pc.Input`、最後以外の各モジュールの `pc.Layer`（`layer0`、`layer1`）、損失を持つ最後の `pc.Output` である。鎖の子は `input`・`layer0`・`layer1`・`output` で、これがそのまま状態のパスになる。層は `net.layers` で順に取れる（`net.layers[0]` が `layer0`）。

- `x` と `y` は入力と目標の spec。`batch` はその先頭の shape で、ここでは `B` 個のサンプル。`batch` を省くと鎖はサンプルを 1 つずつ受け取り、`x` は `TensorSpec((2,))`、`y` は `TensorSpec((1,))` になる
- `loss` はサンプルごとの損失をバッチの shape で返す。`pc.losses.MSELoss(B)`（サンプルごとの平均二乗誤差。`nn.MSELoss` と同じくサンプルの次元で平均する）か `pc.losses.CrossEntropyLoss(B)`、または同じ shape を返す自作の関数。損失には鎖と同じバッチを渡す（[バッチ](../topics/pc.md#%E3%83%90%E3%83%83%E3%83%81)）
- `tau` は活動と双対変数の時定数で、実行のタイマーの秒で表す。`h` は活動（鎖のどのノードも `tau.period`、つまり `tau.h / 4` ごとに更新する）、`lam` は双対変数の時定数。重みの時定数は optimizer が持つ
- `optimizer` はノードが学ぶときに重みを動かす。`pc.optim.SGD(tau=...)` は勾配降下で、`tau` は重みの時定数。1 回の学習ではバッチの平均の勾配に（そのノードが前に学んでからの時間）/ `tau` を掛けた分だけ重みを動かす。ここではノードがバッチごとに 1 回学ぶので、1 バッチあたりの学習率は `SHOWN / optimizer.tau` = 0.3 になる（`SHOWN` は各バッチを見せる時間）。もう 1 つの選択肢は `pc.optim.AdamW`（[optimizer を選ぶ](../topics/pc.md#optimizer-%E3%82%92%E9%81%B8%E3%81%B6)）。各ノードは optimizer を自分の重みに結び付けるので、1 つの `pc.optim.SGD` を鎖全体で使える。`optimizer` に既定値は無い
- `learn_interval` はノードが学ぶ時刻で、各書き込みから数える。ここではバッチごとに 1 回、見せる時間の 3/4 の時点。既定値は無い。`None` は毎回の更新で学ぶモードで、フレームごとに入力が変わるストリーム向け。このモードのよい値はまだ測っていない

ノードの内側では BP と同じくモジュールを autograd で微分する。重みはモジュールを作るときに初期化しておく（ここでは実行の前にホストの `torch.manual_seed` で）。ノードは初期化しない。

## 見せて学ぶ

```python
--8<-- "examples/pc_regression.py:run"
```

`pc.show(run, net, sample, seconds=SHOWN)` はバッチを書き、鎖がそのバッチに対して出す最初の予測を待つ。そしてバッチを実行のタイマーで `SHOWN` 秒見せ終えたところで、その予測を返す。その間ノードは目標に向かって緩和し、学ぶ。この予測はどのノードもまだそのバッチから学んでいない時点のものなので、その損失は、まだ見ていないデータをどれだけ当てられるかを表す（prequential な損失）。

- `mask` はどのサンプルに目標があるかを表す。`mask` が False のサンプルは損失に何も足さない。`mask` がすべて False なら重みは動かない（活動は動く）。走っている同じ鎖を評価するにはこれを使う。切り替える推論モードは無い
- `pc.messages.Prediction` は `y_hat` を、その元になったサンプルの `y`・`mask`・`time` と組にして持つ。予測はいつも自分の目標と対になる
- `run.state_dict()["nodes"]["layer0"]["state"]` は最初の層の状態。活動 `h`、双対変数 `lam`（平衡点では誤差逆伝播法の credit）、モジュール、クロック、optimizer と、予測が跳んだときに `h` を動かすための値（`mu`・`jump`・`jumped`・`jumps`・`seen_jumps`。説明は [mujou.pc の仕組み](../topics/pc-concepts.md#jumps)）を持つ

## 次へ

- 部品のつながりと、ノードが 1 回の更新で行うこと: [mujou.pc の仕組み](../topics/pc-concepts.md)
- `pc.Chain` で MNIST を学ぶ 784-256-256-10 のネットワークと BP の比較: `examples/mnist/`（`uv run python examples/mnist/main.py`）。説明は [PC-ALM](../topics/pc.md#mnist)
- GPU の上の `pc.Chain` で CIFAR-10 を学ぶ ResNet-20 と BP の比較: `examples/cifar10/`（`uv run python examples/cifar10/main.py`）。説明は [PC-ALM](../topics/pc.md#cifar-10)
- 時定数・optimizer・`learn_interval` の決め方、落とし穴、自作のノード: [PC-ALM](../topics/pc.md)
