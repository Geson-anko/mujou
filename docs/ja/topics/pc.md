# 🧠 PC-ALM

`mujou.pc` は PyTorch のモデルをノードに分け、PC-ALM（Augmented Lagrangian Predictive Coding、<https://pub.sakana.ai/pc-alm/>）で学習させるモジュール。[5. PC-ALM で学習する](../tutorial/5-learn-with-pc.md)では `pc.Chain` で鎖を組み、`pc.show` でバッチを見せた。このページでは時定数・optimizer・relaxer の決め方、BP との違い、落とし穴、MNIST と CIFAR-10、自作のノードを説明する。

## BP との違い

| 項目           | 普通の PyTorch の BP                         | `mujou.pc`                                                                                                               |
| -------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 推論と学習     | forward、backward、`optimizer.step()` を順に | 同時に進む。各ノードが自分の周期で更新し、入力を見せている間に学習する                                                   |
| 1 回の学習     | 1 バッチ                                     | ホストがバッチを一定の時間（`S` 周期）見せる。ノードは目標に向かって緩和し、その後に重みを更新する                       |
| 目標           | バッチと一緒に損失関数に渡す                 | 入力と一緒に書き込み、活動と組にしてノードからノードへ出力ノードまで運ぶ                                                 |
| 学習とテスト   | `model.train()` / `model.eval()`             | モードは無い。`mask` が False のサンプルからは学習しないので、テストのバッチも走っている同じノードに流す                 |
| 損失           | 普通はバッチの平均                           | サンプルごとの損失で、shape はバッチの shape。出力ノードが合計する                                                       |
| 勾配を取る範囲 | モデル全体を通して `.backward()`             | 各ノードの内側だけ。隣のノードとは活動とエネルギーの勾配をメッセージでやり取りする                                       |
| 計算量         | 1 バッチに forward と backward が 1 回ずつ   | 1 周期にノードごとに forward と VJP が 1 回なので、1 バッチで `S` 回。MNIST では CPU 上で 1 バッチに BP の約 40 倍かかる |

## ノード

| ノード      | 持つもの                                       | Port                                                                                                                                                |
| ----------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pc.Input`  | 書き込みの回数                                 | `prev`: ホストが `pc.msg.Sample`（`x`、`y`、`mask`）を書く。`next`: `x` を `pc.msg.Activity` の `h` として、`y`・`mask`・書き込みの時刻と一緒に送る |
| `pc.Layer`  | モジュール、活動 `h`、双対変数 `lam`、クロック | `prev` と `next`。`next` は `h` を、自分の `jumps` と入力と一緒に届いた `y`・`mask`・`time` と組にして `pc.msg.Activity` で送る                     |
| `pc.Output` | 最後のモジュール、損失、クロック               | `prev`。`next` はホストが読む `pc.msg.Prediction`（`y_hat`、`y`、`mask`、`time`）を送る                                                             |

`pc.msg.Activity` は `h`・`jumps`・`y`・`mask`・`time` を運ぶ。`jumps` は送り手の `h` が緩和でなく跳んだ回数。`pc.Input` はホストの書き込みを数え、`pc.Layer` は新しいサンプルや受け取る `h` の跳びに合わせて動いた回数と、間隔を置いて学ぶときに自分の重みが起こした跳びを数える。受け手は `jumps` の変化を見たら自分の `h` を予測と一緒に 1 回動かし、跳びを偏差 `h - mu` に入れない。

`pc.Chain(*modules, ...)` は `Input` → `Layer` → ... → `Output` をつなぎ、周期はどれも `tau.period`（`tau.h / 4`）にする。別の周期や別の形にするなら、ノードを 1 つずつ組む。`pc.Input` には spec の `x` と `y`、`period` を渡す。各 `pc.Layer` と `pc.Output` には前のノードの `act_spec`（受け取る `pc.msg.Activity` の spec）を渡し、鎖に沿って `next` を `prev` につなぐ。`mujou.Network` とあわせて `pc.Shown` を継承し、`Input` の `prev` と `Output` の `next` を自分の `prev` と `next` として公開すれば、そのネットワークにも `pc.show` でサンプルを見せられる。

ノードにはバッチの各サンプルを独立に扱う `nn.Module` なら何でも入れられる。1 層でも、層を重ねたものでも、残差ブロックでもよい。`Dropout` と `BatchNorm` は training mode でサンプルを落としたり混ぜたりするので使わない。`LayerNorm` は使える。重みはモジュールを作るときに初期化しておく。ノードは初期化しない。`requires_grad=False` のパラメータは学習しない。

## 時定数を決める

`mujou.pc` の速さはすべて実行のタイマーの秒で指定する。次の表の順に決める。

| 値               | 決めるもの                                   | 決め方                                                                                                                                                                              | MNIST                    |
| ---------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `tau.h`          | 各活動 `h` が緩和する速さ                    | 最初に決める。ほかはここから決まる                                                                                                                                                  | 4 ms                     |
| 周期             | 各ノードが更新する間隔                       | `tau.period`（`tau.h / 4`）。`Layer`・`Output`・`pc.Chain`（`period=`）の既定。硬いモジュールでは短くする（[落とし穴](#%E8%90%BD%E3%81%A8%E3%81%97%E7%A9%B4)）                      | 1 ms                     |
| `tau.lam`        | 双対変数 `lam` が追従する速さ                | `2 * tau.h`                                                                                                                                                                         | 8 ms                     |
| `tau.r`          | ずれ `r = h - mu` が減衰する速さ             | 既定は `math.inf` で減衰しない。深い鎖では `tau.h / 3`。[CIFAR-10](#cifar-10) を見る                                                                                                |                          |
| `relaxer`        | `h` が自分に掛かる力に沿ってどう動くか       | 既定は `pc.relax.Euler()`。[relaxer を選ぶ](#relaxer-%E3%82%92%E9%81%B8%E3%81%B6)を見る                                                                                             |                          |
| 見せる時間       | ホストが各バッチを見せる長さ                 | `S` 周期。`S = 32`                                                                                                                                                                  | 32 ms                    |
| `learn_interval` | ホストの書き込みから重みを更新するまでの時間 | 見せる時間の半分より長く、見せる時間より短く、credit の往復より長くする。MNIST では見せる時間の 3/4                                                                                 | 24 ms                    |
| `optimizer`      | 重みが学ぶ速さ                               | `pc.optim.SGD(tau=見せる時間 / lr)`。`lr` は 1 バッチあたりの学習率。[optimizer を選ぶ](#optimizer-%E3%82%92%E9%81%B8%E3%81%B6)を参照                                               | `pc.optim.SGD(tau=0.32)` |
| `max_dt`         | 1 回の更新で数える `dt` の上限               | 既定の `tau.max_dt`（`tau.r` が無ければ `min(tau.h / 2, 3 * tau.lam)`、ここでは 2 周期）。周期より短い `max_dt` は `ValueError` になる。`tau.lam < tau.h / 12` なら既定でもそうなる | 2 ms                     |

- **学習率**: `pc.optim.SGD` では、重みは 1 回の更新で `elapsed / tau` にバッチの勾配の平均を掛けた分だけ動く。`elapsed` はそのノードが前に学んでからの時間。1 バッチに 1 回学ぶなら `elapsed` は見せる時間なので、`pc.optim.SGD(tau=見せる時間 / lr)` とすれば 1 バッチあたりの学習率が `lr` になる。`elapsed` はその間の各更新の `dt`（1 回ごとに `max_dt` で打ち切ったもの）の和で、周期どおりの更新なら `dt` は 1 周期なのでそのまま数えられる
- **勾配の平均**: サンプルごとの勾配の和を、バッチのサンプルの数 `B`（バッチが無ければ 1）で割ったもの。`mask` が False のサンプルは和に何も足さないが、`B` には数える
- **`learn_interval` を見せる時間の半分から全体の間にする理由**: `learn_interval` はホストが書き込んだ時刻（活動の `time`）から数え、ノードが受け取った時刻からは数えない。そのため、ノードは深さによらずほぼ同じ時刻に学ぶ。ノードは書き込みから `learn_interval` 後に学び、バッチを見せている間は以後 `learn_interval` ごとに学ぶ。半分より短いと 1 バッチで 2 回以上学ぶ。見せる時間以上だと先に次のバッチが書かれ、ノードは一度も学ばない
- **credit の往復**: ノードは入力を初めて見た更新では学ばず、credit が経路を往復するにも時間が要る。往復はおよそ `2 × ホップ数 × 周期`（ホップ数は経路上の接続の数。MNIST では `Input` から `Output` まで 3 本で 6 周期）。`learn_interval` はこれより長くすることを勧める。短いと、落ち着く前の credit から重みが学ぶ。設計時の測定では、往復が 18 周期の深い鎖が、16 周期では収束し、4 周期では有界だが約 4 倍ずれ、2 周期と 1 周期では発散した
- **入力がフレームごとに変わるストリーム**: `learn_interval=None` を渡す。ノードは入力が変わったかどうかによらず毎回の更新で学び、1 秒あたりの学習率は `pc.optim.SGD` の `1 / tau` になる。ストリームに適した値はまだ測っていない
- **`r` の減衰**: `tau.r` は `tau.h * dh/dt` に `-(tau.h / tau.r) * r` を足し、`h` を `mu` へ引き寄せる。`lam` が `r` を積分するので、`tau.r` によらず平衡点では `r = 0` で、`lam` は BP の credit のまま。深い鎖では `lam` が早く credit に近づく（[CIFAR-10](#cifar-10)）。短すぎると `r` が小さくなりすぎ、それを積分する `lam` が遅くなる（その ResNet では、`tau.h / 10` だと `tau.h / 3` より `lam` が落ち着かなかった）。`h` を引く力が `1 + tau.h / tau.r` 倍になるので、`tau.max_dt` もその倍率で割った長さになる
- **安定性**: `tau.h` に対して周期が短いほど安定する。設計時の測定では、ReLU で `period / tau.h` が 0.5 までは安定し、`period / tau.lam` は 0.1〜0.3 が最もよかった。`S = 32` で BP と同じ精度が出た。それより短い見せる時間は確かめていない

## optimizer を選ぶ

`pc.Layer`・`pc.Output`・`pc.Chain` は `optimizer=` が必須。optimizer はノードが学ぶときに、バッチの勾配の平均 `g` で重みをどう動かすかを決める。状態を持たない設定なので、`pc.Chain` は同じものを全ノードに渡す。各ノードはそれを自分の重みに bind し、重みの状態を自分で持つ。

| optimizer                                                                    | `elapsed` 秒の 1 回の学習                                                                    | 状態                   |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------- |
| `pc.optim.SGD(tau=...)`                                                      | `p -= (elapsed / tau) * g`                                                                   | 無し                   |
| `pc.optim.AdamW(tau=..., tau_m=..., tau_v=..., eps=1e-8, weight_decay=1e-2)` | 学習率 `elapsed / tau` の AdamW。moment の `m` と `v` は時定数 `tau_m` と `tau_v` で減衰する | `m`、`v`、`c_m`、`c_v` |

- **ステップでなく時間**: `tau` はどれも秒。`elapsed` はノードが前に学んでからの時間なので、1 バッチの間に何回学んでも、各歩みの `elapsed` を合わせると見せる時間になる。そのため `pc.optim.SGD` では 1 バッチに 1 回学んでも 2 回学んでも、1 バッチで重みが動く量は同じ
- **SGD** は MNIST が使うもの。`tau` は時定数の表のとおり、見せる時間を 1 バッチあたりの学習率で割って決める
- **AdamW** は勾配の大きさによらず、各重みを 1 回におよそ `elapsed / tau` 動かす。`mujou.pc` に適した AdamW の値はまだ測っていない
- **`torch.optim.AdamW` からの換算**: `Δt` は 2 回の学習の間の時間で、1 バッチに 1 回学ぶなら見せる時間。`torch.optim.AdamW` の `lr`・`betas` の値を使うなら、`tau = Δt / lr`、`tau_m = -Δt / ln(beta1)`、`tau_v = -Δt / ln(beta2)` とする。`Δt` が一定なら、歩みは `torch.optim.AdamW` と同じになる。32 ms に 1 回学ぶなら、`beta1 = 0.9` は `tau_m` ≈ 0.30 s、`beta2 = 0.999` は `tau_v` ≈ 32 s
- **`weight_decay`** は `torch.optim.AdamW` と同じく単位を持たない。重みを時定数 `tau / weight_decay` で減衰させる
- **不正な値**: `tau` と `eps` はどれも有限で > 0、`weight_decay` は有限で >= 0 でなければ、optimizer が `ValueError` を送出する。ノードに学ばせないときは、その `pc.Layer` か `pc.Output` に無限大の `tau` でなく `learn_interval=math.inf` を渡す（`pc.Chain` は 1 つの `learn_interval` を全ノードに渡す）
- **学ぶ重みだけ**: ノードは `requires_grad=True` のパラメータだけを bind する。凍結したパラメータには weight decay も掛からない
- **状態**: ノードの optimizer の状態は `run.state_dict()["nodes"][path]["state"]["optimizer"]` にある。`pc.optim.AdamW` ならキー `"m"`・`"v"`・`"c_m"`・`"c_v"` の dict、`pc.optim.SGD` なら空の dict。実行のほかの状態と一緒に保存・復元される

```python
# torch の lr=1e-3, betas=(0.9, 0.999) を、32 ms のバッチごとに 1 回学ぶ場合に換算して丸めたもの。
optimizer = pc.optim.AdamW(tau=0.032 / 1e-3, tau_m=0.3, tau_v=32.0)
```

??? question "`torch.optim` を使わないのはなぜ？"

    `torch.optim` の optimizer は時間でなくステップを数えるので、`betas` が与える時定数がノードの学ぶ頻度で変わる。また、作るだけで free-threaded の Python に GIL が戻る。

## relaxer を選ぶ

`pc.Layer` と `pc.Chain` は `relaxer=` を取る。relaxer は、活動 `h` が自分に掛かる力 `F = r + lam + g` に沿ってどう動くかを決める。optimizer が重みに対して行うことの、`h` 版である。省略でき、既定は `pc.relax.Euler()`。状態を持たない設定なので、`pc.Chain` は同じものを全 `pc.Layer` に渡す。各 `pc.Layer` はそれを自分の `h` に bind し、`h` の状態を自分で持つ。`pc.Output` は `h` を持たないので、relaxer を取らない。

| relaxer                       | `dt` 秒の 1 回の動き                                         | 状態       |
| ----------------------------- | ------------------------------------------------------------ | ---------- |
| `pc.relax.Euler()`            | `h -= (dt / tau.h) * F`                                      | 無し       |
| `pc.relax.HeavyBall(tau=...)` | `v = exp(-dt / tau) * v + F`、続けて `h -= (dt / tau.h) * v` | `velocity` |

- relaxer が変えるのは `h` が不動点に着くまでの道で、不動点ではない。不動点では `F = 0` で、`lam` は BP の credit になる。どの relaxer が `h` を動かしても、`lam` はこれまでどおり動かした後の `h` で進む
- 深い鎖には HeavyBall を使う。速度が前の更新の力を覚える（heavy ball）。力が続くと `h` が速く動くので、多くのノードの鎖を credit が少ない周期で渡る。効くのは credit の先頭を速めることで、緩和の遅い裾は速くならない。[CIFAR-10](#cifar-10) の ResNet を周期 `tau.h / 4` で学ばせると、慣性ありの 64 周期は 3 epoch 目で慣性なしの 128 周期に並んだが、10 epoch 目には遅れた（62.77% と 68.73%）。いまの例は慣性を使わず、短い周期・速い双対・`r` の減衰で緩和させ、どちらよりもよかった。これらと `pc.relax.HeavyBall` の組み合わせは測っていない。浅い鎖には要らない。MNIST は既定のまま使う
- `tau` は周期に合わせて選ぶ。力は `dt` ではなく 1 回の更新ごとに掛かるので、利得は 1 更新あたりの係数 `exp(-dt / tau)` で決まる。試した値は周期 `tau.period` に対する `tau = tau.h / 4` で、係数は 0.37（`tau` が周期と同じ）になる。ResNet の 11 ノードでは 64 周期が `pc.relax.Euler()` の 128 周期とほぼ同じ精度になり、実時間は半分で済んだ。ResNet では係数 0.47（`tau.h / 3`）が 3 回の実行のうち 1 回で崩れ、0.8 は発散した。約 0.78 という限界は恒等写像の線形な鎖のもので、実際の限界はそれより低く、更新が遅れるとさらに低くなる。周期を変えるときは、係数を 0.37 の近くに保つ
- 新しいサンプルのとき、また読んでいる `h` が跳んだときには、速度を 0 に戻す。`r` と `lam` は保たれるが、速度が覚えているのは前の入力の力だからである。ノード自身の重みが跳んだときは戻さない
- `pc.relax.HeavyBall` の `tau` は有限で > 0 でなければ、`ValueError` を送出する
- ノードの relaxer の状態は `run.state_dict()["nodes"][path]["state"]["relaxer"]` にある。`pc.relax.HeavyBall` ならキー `"velocity"` の dict、`pc.relax.Euler` なら空の dict。実行のほかの状態と一緒に保存・復元される
- relaxer を自作するには、optimizer を自作するときと同じく、`pc.relax.Relaxer` と `pc.relax.BoundRelaxer` を継承して書く（[mujou.pc の仕組み](pc-concepts.md#7-relaxer)）。不動点は `F = 0` のところでなければならない。`lam` が credit になるのはそこだけだからである

## MNIST

`examples/mnist/` は 784-256-256-10 の ReLU ネットワークを `pc.Chain` で MNIST に学習させ、同じネットワークを同じ初期重みと同じバッチで BP でも学習させる。3 epoch 後のテスト精度は PC-ALM が 97.60%、BP が 97.54% だった（seed 0）。

```sh
uv run python examples/mnist/main.py            # 3 epoch、約 2 分
uv run python examples/mnist/main.py --epochs 1 --watch
```

```python
--8<-- "examples/mnist/main.py:chain"
```

```python
--8<-- "examples/mnist/main.py:run"
```

- **prequential 精度**（学習中）: 各バッチへの最初の予測で測る。`pc.show` が返すもので、どのノードもまだそのバッチから学んでいない時点のもの。まだ見ていないデータをどれだけ当てられるかを表す
- **テスト精度**: ホストはテストのバッチを `mask` をすべて False にして見せるので、どのノードもそこから学ばない。採点には元の `mask` を取っておく。切り替える推論モードは無い
- **最後のバッチ**: `data.batches` は、どのバッチも spec の大きさになるよう、最後のバッチの不足分を `mask` が False のゼロで埋める。正解を数えるのは、元の `mask` が True のサンプルだけ
- **`--watch`** は各ノードの `dt` を数え（`watch.py`）、更新が中央値で 1.5 周期、99% 点で 2 周期より遅れると警告する
- `data.py`・`metrics.py`・`bp.py` は `mujou.pc` に依存しない。データは素のテンソルで、BP の基準は素の PyTorch

## CIFAR-10

`examples/cifar10/` は、CUDA のデバイスに載せた ResNet-20 を残差ブロックごとのノードにして CIFAR-10 に学習させる。同じネットワークを、同じ初期重み・同じバッチ・同じ `pc.optim.AdamW` で、GPU 上の BP でも学習させる。モジュールは `pc.Chain` の `Input` の後ろのノードになる。stem と最初のブロック、残り 8 つのブロック、head の 11 ノードである。3 epoch 目の prequential 精度は PC-ALM が 54.83%、BP が 54.45%、その後のテスト精度は 59.05% と 59.02% だった（seed 0）。10 epoch 目は prequential 精度が 73.35% と 73.27%、テスト精度が 73.79% と 72.61% だった。同じ設定の別の実行では 10 epoch 目が 71.56% と 71.41% で、PC-ALM は BP に並ぶか、2 ポイントほど下にとどまる。1 epoch に実時間で約 750 秒かかり、BP は 2.8 秒。学習率を減衰させないので、テスト精度はどちらも epoch ごとに数ポイント動く。GPU の計算も実行ごとにビット単位では一致しない。

```sh
uv run python examples/cifar10/main.py            # 3 epoch、RTX 4090 で約 40 分
uv run python examples/cifar10/main.py --epochs 10  # 約 2 時間
```

```python
--8<-- "examples/cifar10/main.py:chain"
```

- **残差ブロックごとのノード**: ノードの内側では autograd がモジュール（ブロックのスキップ接続と活性化を含む）を微分し（BP）、ノードの間には PC-ALM のメッセージだけが流れる
- **周期は `tau.h / 16`**: ブロックが緩和を硬くするので、`pc.Chain` の既定の `tau.h / 4` では `h` の陽的な歩みが BP の credit に届かなかった（[落とし穴](#%E8%90%BD%E3%81%A8%E3%81%97%E7%A9%B4)）。`tau.h / 16` では `tau.h / 64` と同じ credit に届いた。これより短くしても更新が増えるだけ
- **速い双対と `r` の減衰**: 残る問題は、11 ノードを渡る credit が落ち着くまでの時間。`tau.lam = tau.h / 4` で `lam` が `r` に早く追いつき、`tau.r = tau.h / 3` でずれが減衰する。どちらも平衡点は変えない。10 epoch 学んだ重みで、最初のノードの `lam` と credit の cos は、`24 * tau.h` 緩和したあと、速い双対だけなら 0.84、両方なら 0.94 だった。`tau.r = tau.h` では、3 epoch 目の prequential 精度が減衰なしの 51.75% から 54.24% に上がった
- **1 バッチに `24 * tau.h`**（`S = 384` 周期）、学ぶのはその 0.9 の時点: `16 * tau.h`・学習 0.75・`tau.r = tau.h` でも最初の数 epoch は BP に並んだが、10 epoch 目の prequential 精度は 67.28% だった。ネットワークが学ぶにつれて緩和が遅くなるからである
- **GPU で実時間の 4 分の 1**: `device="cuda"` で、すべてのノードが GPU で計算し、ノードの間のメッセージも GPU に置かれたまま渡る（[デバイス](threads-and-devices.md#%E3%83%87%E3%83%90%E3%82%A4%E3%82%B9)）。ノードの 1 回の更新は RTX 4090 で約 1.2 ms かかり、サンプルが 1 でも 128 でも変わらない（カーネルを起動する CPU が限界になる）ので、バッチを 128 にした。ワーカー 11 本なら実時間 5 ms の周期に追いつく。周期はタイマーで 1.25 ms なので、タイマーを実時間の 4 分の 1 で進める（`--time-scale` 0.25）。最初のノードの更新の 99% が前の更新から 1.01 周期以内に来た
- **AdamW**: [optimizer を選ぶ](#optimizer-%E3%82%92%E9%81%B8%E3%81%B6)のとおり、Adam の `lr` から作る。BP の基準も同じ `pc.optim.AdamW` をモデル全体に束ね、1 バッチに 1 回、見せる時間だけ進める。ノードが学ぶ時間は前に学んでからの時間で、見せる時間に近い。そのため BP の基準は同じ勾配に PC-ALM とほぼ同じステップを踏む
- **GroupNorm**: BatchNorm の代わりに使う。移動統計も切り替えるモードも無い。詰め物も含め、バッチの中のサンプルは互いに影響しない

## バッチ

バッチはメッセージのどのテンソルも先頭に持つ shape。`pc.Chain`・`pc.Input`・`pc.msg.activity_spec` に `batch` で渡し、spec の `x` と `y` もこの shape で始める。`pc.Layer` と `pc.Output` は受け取る `act_spec` から読む。

| `batch`      | `x`（入力 784） | `y`（クラス 1 つ） | `mask`   | 用途                                         |
| ------------ | --------------- | ------------------ | -------- | -------------------------------------------- |
| `()`（既定） | `(784,)`        | `()`               | `()`     | センサーのように 1 サンプルずつ受け取る      |
| `64`         | `(64, 784)`     | `(64,)`            | `(64,)`  | 64 サンプル                                  |
| `(8, 4)`     | `(8, 4, 784)`   | `(8, 4)`           | `(8, 4)` | たとえば 8 個の環境それぞれに 4 エージェント |

- サンプルは互いに独立であること。module はバッチのどの次元についても、各サンプルを別々に計算する。系列の時間のように module が混ぜる次元はバッチでなくサンプルの次元。`T` ステップの系列 64 本なら `batch=64` で `x` は `(64, T, 784)`
- 多次元のバッチは同じサンプルを 1 次元に並べたものと同じように学ぶ。`(8, 4)` は `32` と同じ
- 損失もバッチを受け取る。`pc.loss.MSELoss(batch)` と `pc.loss.CrossEntropyLoss(batch)`。`pc.loss.CrossEntropyLoss` の `y` はクラス（int64 で、`y_hat` からクラスの次元を除いた shape）かクラスごとの確率（`y_hat` と同じ shape）。クラスの次元は `nn.CrossEntropyLoss` と同じく `y_hat` のバッチの直後。その後ろの次元（画像の画素など）は平均する

## 落とし穴

!!! warning "バッチは鎖と損失の両方に渡す"

    `batch` の既定は `()`（1 サンプル）で、spec の shape からは読まない。64 サンプルの spec でどこにも `batch` を渡さないと、鎖は 64 サンプルを 1 つのサンプルとして扱い `pc.loss.MSELoss()` はそれらを平均する。それでも動くが、各サンプルの credit は本来の 1/64 になる。`pc.Chain`（か `pc.Input`）と損失に同じ `batch` を渡す。片方だけに渡すと `pc.Output` がエラーを送出する。

!!! warning "損失関数はサンプルごとの損失を返す"

    `pc.loss.MSELoss(batch)`、`pc.loss.CrossEntropyLoss(batch)`、またはバッチの shape を返す `(y_hat, y)` の自作の関数を渡す。`nn.MSELoss()` のようにバッチで平均する損失を渡すと、`pc.Output` が `ValueError` を送出する。`pc.loss.MSELoss` は `nn.MSELoss` と同じく各サンプルのすべての次元（バッチ以外）で平均し、`0.5` は付けない。重みはバッチに対する `nn.MSELoss()` の勾配に沿って動く。

!!! warning "予測は出力ノードから読む"

    予測を送るのは `pc.Output` の `next`（`pc.Chain` なら `net.next`）だけ。`Layer` の `next` が送るのは活動 `h` と、中継する `y`・`mask`・`time` で、予測は含まない。層の `lam` を見るときは `run.state_dict()["nodes"]["layer0"]["state"]["lam"]` を読む。走っている間にノードの属性を読むと、その更新と競合する。

!!! warning "最初のサンプルが届くまで活動はゼロ"

    ホストの最初の書き込みが届くまで、ノードが受け取る活動は `time` が 0 で `mask` がすべて False。このとき `run.read(net.next)` は `mask` がすべて False の予測を返し、ノードは学ばない。書いたバッチの予測を得るには、`pc.show` のように `write` が返した時刻と `time` を照合する。

!!! warning "負荷が高いとネットワークが発散する"

    ノードはそれぞれ更新しながらメッセージをやり取りする。あるノードが遅れると、隣のノードはその古いメッセージから学び続け、ネットワークが発散するか精度が落ちる。MNIST は `y_hat` が有限でないと止まる。予測が有限のまま発散したときは、精度の低下としてしか見えない。わかっている原因は 2 つ。

    - **時間の速さが速すぎる**: MNIST は `--time-scale` で実行のタイマーを速める。1 でなく 2 にすると、最も遅いノードが周期の大半を計算に使い、その更新の 1% が 2 周期より遅れた。`--watch` で 1 epoch 回し、警告が出ず発散もしない最も速い時間の速さを選ぶ
    - **ほかのプロセスがコアを使っている**: コアを待つノードがあると、隣のノードが先に進む。32 コアの機械で時間の速さ 1 なら、busy loop 26 本と並べて 1 epoch を完走し、32 本では発散した。0.5 なら 32 本でも完走した。ほかに何も動いていない機械で回すか、時間の速さを下げる

    `max_dt` ではこれを防げない。`max_dt` は 1 回の更新で数える時間の上限で、超えた分は緩和からも次の重みの更新からも捨てられる。

!!! warning "ヤコビアンの大きいモジュールには短い周期が要る"

    ノードは、次のノードが送り返す値で活動 `h` を緩和する。この値は次のノードのモジュールのヤコビアン `J` を通る。`h` の陽的な歩みが安定なのは `(1 + |J|^2) * period / tau.h < 2` の間だけで、`|J|` は `J` の最大特異値。`pc.Chain` の周期 `tau.h / 4` なら、`|J|` がおよそ 2.6 未満の間になる。これはどの更新も遅れないときの境界である。遅れた更新は `max_dt` までの長い 1 歩を進め、ノードは次のノードが 1 周期前に送った値を読むこともある（線形の鎖では境界が 1.2〜1.8 程度まで下がった）。次のノードが `Output` なら、損失の曲率（交差エントロピーなら 1/2 以下）が `|J|^2` に掛かり、境界は緩む。

    ResNet のブロックはこの範囲を超える。`examples/cifar10/` では、BP と PC-ALM のどちらで学んでも、数 epoch 後のブロックの最大特異値が 2〜9 になった。範囲を超えると、`lam` は BP の credit に落ち着かない。BP が 3 epoch で学んだ重みでは、`lam` と credit の cos は周期 `tau.h / 4` だと 0.06〜0.3 にとどまり、`tau.h / 16` だと `tau.h / 64` と同じ 0.95〜0.99 に届いた。`tau.h / 4` で学ぶ PC-ALM は、ブロックをほぼ恒等写像のままにして（最大特異値 1.0〜1.7、`tau.h / 16` では 2〜6）、学ぶ量も少なかった。

    `pc.Chain` に短い `period` を渡し、`max_dt` は 2 周期程度にする（既定の `tau.max_dt` は `tau.h / 16` の 8 周期になる）。手元の重みで確かめるには、重みを固定して 1 バッチを緩和させ、autograd の勾配と `lam` を比べる。半分の周期と同じ `lam` になる周期なら十分に短い。残りは多くのノードを渡る credit が落ち着くまでの時間で、速い `tau.lam` と `r` の減衰（`tau.r`）で縮む（[CIFAR-10](#cifar-10)）。

    効かなかったものが 2 つある。1 つ目は残差の枝を小さく始めること（枝の 0.1 倍や 0.25 倍）。AdamW の下で GroupNorm のスケールが育ち、最大特異値は下がらなかった。BP も悪くなった（10 epoch 目のテスト精度が 72.6% から 68.5〜69.2% に下がった）。2 つ目は次のノードが硬いところで刻みを縮めること（最大の曲率をべき乗法で見積もった）。固定した重みでは `lam` が正しくなったが、学習は遅くなった。

## 自分のノードを書く

`mujou.pc.functional` にノードを構成する計算がある。`vjp` と `loss_vjp`（ノードのエネルギーの勾配）、`relax_`（`h` と `lam` の 1 ステップ）。重みは optimizer が、`h` は relaxer が動かす。複数の入力を持つノードのように `pc.Layer` で書けないノードは、これらで書く。ここに出てくる名前（`mu`・`lam`・`g_in`・`jumps`・`Tick`）は [mujou.pc の仕組み](pc-concepts.md)で説明している。

1. `__init__` で module を device に載せた後、クロックと bind した optimizer（`h` を `pc.relax.HeavyBall` などで動かすなら、bind した relaxer も）を属性に置く。どれもそれだけでノードの状態に入る
    - `self.clock = pc.clock.LearnClock.of(tau, period=self.period, learn_interval=...)`。`max_dt` は `pc.Layer` と同じく `tau` の既定になる
    - `self.optimizer = optimizer.bind(params)`。`params` は学ぶパラメータ。bind はノードごとに行う。bind した同じ optimizer を 2 つのノードに持たせてはいけない
    - `self.relaxer = relaxer.bind(h)`。`h` はノードが持つ活動で、同じくノードごとに bind する。置かなければ、`relax_` は素の 1 歩を進める
2. 毎回の更新で `t = self.clock.tick(dt, act["time"], self.timer.now())` を呼ぶ
3. bind した `params` を同じ順で `pc.functional.vjp` に渡し、予測 `mu`、`g_in`、学ぶなら `g_theta` を求めて、`g_in` を前のノードへ送る
4. `t.new` が True か、活動の `jumps` が前に読んだものと違えば、`h` を予測と一緒に 1 回動かす。前回の更新の `mu_last` で `h <- mu + (h - mu_last)` とし、自分の `jumps` に 1 を足す。`mu`、前に読んだ `jumps`、自分の `jumps` は状態に入れる（素の数は `register_state`）
5. `pc.functional.relax_(..., t.dt, tau, self.relaxer)` で活動を動かす。手順 4 で付け替えたときは、先に `self.relaxer.reset_()` を呼ぶ。relaxer が覚えているのは前の入力の力だからである
6. `h` を `pc.msg.Activity` で送る。自分の `jumps` と、受け取った活動の `y`・`mask`・`time` を組にする
7. `t.learn` が True で `mask` が True のサンプルがあれば、`self.optimizer.step_([g / B for g in g_theta], t.elapsed)` で学ぶ。`B` はバッチのサンプルの数。optimizer はバッチを知らないので、`B` で割るのはノード

手順 4 を抜くと、ノードは `pc.Layer` と違うダイナミクスになる。新しいサンプルや下の跳びが偏差 `h - mu` に入り、送り返す credit にも入る。間隔を置いて学ぶノードは、重みの更新が起こした `mu` の跳びの分も `h` を動かし、`jumps` に 1 を足す。`pc.Layer` はどちらも 1 回後の更新で行う（docstring に説明がある）。

`mujou.pc` のノードとつなぐには、Port の spec に前のノードの `act_spec` か `pc.msg.activity_spec(h, y, batch)` を使う。引数は [API リファレンス](../api/pc.md)にある。PC 以外の、ノードをまたいで学ぶ則は[更新則](update-rules.md)を参照。

### 例: 入力が 2 つのノード

`examples/pc_merge.py` の `Merge` は 2 本の塔（`pc.Layer`）の活動をつないだものから予測する。毎回の更新で学ぶ（`learn_interval=None`）ので、重みの跳びの処理は要らない。

```python
--8<-- "examples/pc_merge.py:merge"
```

`pc.Layer` との違いは 2 つだけ。入力ごとに Port を持って `g_in` を分けて送り返すことと、どちらかの入力の `jumps` が変われば `h` を動かすこと（手順 4）。
