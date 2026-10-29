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

| ノード      | 持つもの                                       | Port                                                                                                                                                                  |
| ----------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pc.Input`  | 何も持たない                                   | `prev`: ホストが `pc.msg.Sample`（`x`、`y`、`mask`、`reset`）を書く。`next`: `x` を `pc.msg.Activity` の `h` として、`y`・`mask`・`reset`・書き込みの時刻と一緒に送る |
| `pc.Layer`  | モジュール、ずれ `r`、双対変数 `lam`、クロック | `prev` と `next`。`next` は予測 `mu` を `h` として、入力と一緒に届いた `y`・`mask`・`reset`・`time` と組にして `pc.msg.Activity` で送る                               |
| `pc.Output` | 最後のモジュール、損失、クロック               | `prev`。`next` はホストが読む `pc.msg.Prediction`（`y_hat`、`y`、`mask`、`time`）を送る                                                                               |

`pc.msg.Activity` は `h`・`y`・`mask`・`reset`・`time` を運ぶ。`pc.Layer` は活動の予測 `mu = module(h_in)` からのずれ `r = h - mu` と双対変数 `lam` を持ち、`mu` を送る。前へ流れるのは順伝播の値で、credit は後ろへだけ流れる。何が `mu` を動かしても（新しいサンプル、下のノードの学習、自分の学習）`r` と `lam` は保たれて緩和する。ただし、新しく始まるサンプルでは消す（[新しく始めるサンプル](#reset)）。

`pc.Chain(*modules, ...)` は `Input` → `Layer` → ... → `Output` をつなぎ、周期はどれも `tau.period`（`tau.r = tau.h` なら `tau.h / 2`）にする。別の周期や別の形にするなら、ノードを 1 つずつ組む。`pc.Input` には spec の `x` と `y`、`period` を渡す。各 `pc.Layer` と `pc.Output` には前のノードの `act_spec`（受け取る `pc.msg.Activity` の spec）を渡し、鎖に沿って `next` を `prev` につなぐ。`mujou.Network` とあわせて `pc.Shown` を継承し、`Input` の `prev` と `Output` の `next` を自分の `prev` と `next` として公開すれば、そのネットワークにも `pc.show` でサンプルを見せられる。

ノードにはバッチの各サンプルを独立に扱う `nn.Module` なら何でも入れられる。1 層でも、層を重ねたものでも、残差ブロックでもよい。`Dropout` と `BatchNorm` は training mode でサンプルを落としたり混ぜたりするので使わない。`LayerNorm` は使える。重みはモジュールを作るときに初期化しておく。ノードは初期化しない。`requires_grad=False` のパラメータは学習しない。

## 新しく始めるサンプル: `reset` {#reset}

`r` と `lam` が持つのは、ノードが見てきたサンプルの credit である。ホストは書くサンプルごとに、新しく始まるかどうかを `reset`（サンプルごとの bool で、`mask` と同じ shape）で伝える。

| データ                               | `reset`                                    |
| ------------------------------------ | ------------------------------------------ |
| シャッフルしたバッチ、独立なサンプル | すべて True                                |
| 各フレームが前に続くストリーム       | False。エピソードの最初のフレームだけ True |
| テストのバッチ                       | 学習と同じく True                          |

ノードはその書き込みを初めて見た更新で 1 回だけ、そのサンプルの `r` と `lam` を 0 にする。関係のないサンプルの credit は残らず、新しいサンプルの credit は出力から戻ってくるにつれて 0 から育つ。`reset` が False のサンプルは `r` と `lam` を保つので、credit がフレームからフレームへ引き継がれる。ノードが見なかった書き込み（ノードが更新する前にホストが次を書いた）ではリセットされない。

```python
new = torch.ones_like(mask)  # シャッフルしたデータ: どのサンプルも新しく始まる
p = pc.show(run, net, pc.msg.Sample(x=x, y=y, mask=mask, reset=new), seconds=shown)
```

## 時定数を決める

`mujou.pc` の速さはすべて、実行のタイマーの秒で時定数として指定する。時定数が短いほど、その値は速く動く。次の表の順に決める。

| 値               | 決めるもの                                   | 決め方                                                                                                                                                     | MNIST                    |
| ---------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `tau.h`          | 各ずれ `r` が緩和する速さ                    | 最初に決める。ほかはここから決まる                                                                                                                         | 4 ms                     |
| `tau.r`          | ずれ `r = h - mu` が減衰する速さ             | `tau.h`。必須で、既定値は無い。`math.inf` で減衰を切る                                                                                                     | 4 ms                     |
| `tau.lam`        | 双対変数 `lam` が追従する速さ                | `tau.h / 2`                                                                                                                                                | 2 ms                     |
| 周期             | 各ノードが更新する間隔                       | `tau.period`（`tau.r = tau.h` なら `tau.h / 2`）。`Layer`・`Output`・`pc.Chain`（`period=`）の既定。上の 2 つと組むと、credit は 1 周期に 1 ノードずつ戻る | 2 ms                     |
| `relaxer`        | `r` が自分に掛かる力に沿ってどう動くか       | 既定は `pc.relax.Euler()`。[relaxer を選ぶ](#relaxer-%E3%82%92%E9%81%B8%E3%81%B6)を見る                                                                    |                          |
| 見せる時間       | ホストが各バッチを見せる長さ                 | `S` 周期。`S` は credit の往復 `2 × ホップ数` より数周期長く                                                                                               | 16 周期、32 ms           |
| `learn_interval` | ホストの書き込みから重みを更新するまでの時間 | 見せる時間の半分より長く、見せる時間より短く、credit の往復より長くする。MNIST では見せる時間の 3/4                                                        | 24 ms                    |
| `optimizer`      | 重みが学ぶ速さ                               | `pc.optim.SGD(tau=見せる時間 / lr)`。`lr` は 1 バッチあたりの学習率。[optimizer を選ぶ](#optimizer-%E3%82%92%E9%81%B8%E3%81%B6)を参照                      | `pc.optim.SGD(tau=0.32)` |
| `max_dt`         | 1 回の更新で数える `dt` の上限               | 既定の `tau.max_dt`。ここでは 1 周期。下の「周期と `max_dt`」を見る                                                                                        | 2 ms                     |

- **周期と `max_dt`**: 既定の `tau.period` は `tau.h / (1 + tau.h / tau.r)`、既定の `tau.max_dt` は `tau.period` と `2 * tau.lam` の短いほう。周期より短い `max_dt` は、ノードを作るときに `ValueError` になる。`tau.lam >= tau.period / 2` なら、既定の `max_dt` は既定の周期と同じ
- **学習率**: `pc.optim.SGD` では、重みは 1 回の更新で `elapsed / tau` にバッチの勾配の平均を掛けた分だけ動く。`elapsed` はそのノードが前に学んでからの時間。1 バッチに 1 回学ぶなら `elapsed` は見せる時間なので、`pc.optim.SGD(tau=見せる時間 / lr)` とすれば 1 バッチあたりの学習率が `lr` になる。`elapsed` はその間の各更新の `dt`（1 回ごとに `max_dt` で打ち切ったもの）の和で、周期どおりの更新なら `dt` は 1 周期なのでそのまま数えられる
- **勾配の平均**: サンプルごとの勾配の和を、バッチのサンプルの数 `B`（バッチが無ければ 1）で割ったもの。`mask` が False のサンプルは和に何も足さないが、`B` には数える
- **`learn_interval` を見せる時間の半分から全体の間にする理由**: `learn_interval` はホストが書き込んだ時刻（活動の `time`）から数え、ノードが受け取った時刻からは数えない。そのため、ノードは深さによらずほぼ同じ時刻に学ぶ。ノードは書き込みから `learn_interval` 後に学び、バッチを見せている間は以後 `learn_interval` ごとに学ぶ。半分より短いと 1 バッチで 2 回以上学ぶ。見せる時間以上だと先に次のバッチが書かれ、ノードは一度も学ばない
- **credit の往復**: ノードは入力を初めて見た更新では学ばず、credit が経路を往復するにも時間が要る。往復はおよそ `2 × ホップ数 × 周期`（ホップ数は経路上の接続の数。MNIST では `Input` から `Output` まで 3 本で 6 周期）。`learn_interval` はこれより長くすることを勧める。credit が戻る前に学ぶと、入力に近いノードは credit の一部からしか学べない。どのノードも 1 周期ごとに forward と VJP を 1 回ずつ計算するので、大きいノードを少なく置くほど 1 バッチに要る周期は少ない
- **入力がフレームごとに変わるストリーム**: `learn_interval=None` を渡す。ノードは入力が変わったかどうかによらず毎回の更新で学び、1 秒あたりの学習率は `pc.optim.SGD` の `1 / tau` になる。ストリームに適した値はまだ測っていない
- **`r` の減衰**: `tau.r` は `tau.h * dr/dt` に `-(tau.h / tau.r) * r` を足す。`lam` が `r` を積分するので、`tau.r` によらず平衡点では `r = 0` で、`lam` は BP の credit のまま。`r` を引く力が `1 + tau.h / tau.r` 倍になるので、既定の周期もその倍率で割った長さになる
- **この値にする理由**: `tau.r = tau.h`、`tau.lam = tau.h / 2`、周期 `tau.h / 2` なら、1 回の更新で `delta = r + lam` がちょうど `-g` になる。credit は 1 周期に 1 ノードずつ戻り、行き過ぎない。ノードは `mu` を送るので credit は後ろへだけ流れ、モジュールのヤコビアンによらず、深い鎖でも 1 ノードのときと同じ刻みで安定する。長い刻み（`dt = tau.h = tau.lam`、減衰なし）は `delta` が 2 倍に行き過ぎ、その行き過ぎが通るノードごとに大きくなる。短い刻みは安定だが、周期が多く要る

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

`pc.Layer` と `pc.Chain` は `relaxer=` を取る。relaxer は、ずれ `r` が自分に掛かる力 `F = (1 + tau.h / tau.r) * r + lam + g` に沿ってどう動くかを決める。optimizer が重みに対して行うことの、`r` 版である。省略でき、既定は `pc.relax.Euler()`。状態を持たない設定なので、`pc.Chain` は同じものを全 `pc.Layer` に渡す。各 `pc.Layer` はそれを自分の `r` に bind し、`r` の状態を自分で持つ。`pc.Output` は `r` を持たないので、relaxer を取らない。

`dt` 秒の 1 回で、`pc.relax.Euler()` は `r -= (dt / tau.h) * F` と動かす。状態は持たない。

- relaxer が変えるのは `r` が不動点に着くまでの道で、不動点ではない。不動点では `F = 0` で、`lam` は BP の credit になる。どの relaxer が `r` を動かしても、`lam` は動かした後の `r` で進む
- ノードの relaxer の状態は `run.state_dict()["nodes"][path]["state"]["relaxer"]` にある。`pc.relax.Euler` なら空の dict で、自作の relaxer なら、それが持つもの。実行のほかの状態と一緒に保存・復元される。ノードは新しい入力のたびに `reset_(reset)` を呼ぶ。力を覚える relaxer は、新しく始まるサンプルの分だけそれを忘れる
- relaxer を自作するには、optimizer を自作するときと同じく、`pc.relax.Relaxer` と `pc.relax.BoundRelaxer` を継承して書く（[mujou.pc の仕組み](pc-concepts.md#7-relaxer)）。不動点は `F = 0` のところでなければならない。`lam` が credit になるのはそこだけだからである

## MNIST

`examples/mnist/` は 784-256-256-10 の ReLU ネットワークを `pc.Chain` で MNIST に学習させ、同じネットワークを同じ初期重みと同じバッチで BP でも学習させる。3 epoch 後のテスト精度は PC-ALM が 97.56%、BP が 97.54% だった（seed 0）。prequential 精度はどの epoch も BP との差が 0.03 ポイント以内だった。

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
- **`--watch`** は各ノードの `dt` を数え（`watch.py`）、更新が中央値で 1.5 周期、99% 点で 2 周期より遅れると警告する。`--time-scale 2` でも更新の 99% が 1.25 周期以内に来て、精度も同じだった
- `data.py`・`metrics.py`・`bp.py` は `mujou.pc` に依存しない。データは素のテンソルで、BP の基準は素の PyTorch

## CIFAR-10

`examples/cifar10/` は、CUDA のデバイスに載せた ResNet-20 をステージごとのノードにして CIFAR-10 に学習させる。同じネットワークを、同じ初期重み・同じバッチ・同じ `pc.optim.AdamW` で、GPU 上の BP でも学習させる。モジュールは `pc.Chain` の `Input` の後ろのノードになる。stem と最初のステージ、2 つ目のステージ、3 つ目のステージ、head の 5 ノードである。32 サンプルのバッチを 12 周期ずつ見せると、3 epoch 目の prequential 精度は PC-ALM が 57.11%、BP が 55.91%、その後のテスト精度は 59.47% と 61.76% だった（seed 0）。10 epoch 目は 73.05% と 73.20%、73.90% と 74.35%。1 epoch は 48 秒で、BP は 8〜9 秒。学習率を減衰させないので、テスト精度はどちらも epoch ごとに数ポイント動く。GPU の計算も実行ごとにビット単位では一致しない。

```sh
uv run python examples/cifar10/main.py            # 3 epoch、RTX 4090 で約 4 分
uv run python examples/cifar10/main.py --epochs 10  # 約 13 分
```

```python
--8<-- "examples/cifar10/main.py:chain"
```

- **ステージごとのノード**: ノードの内側では autograd がモジュール（ブロックとそのスキップ接続・活性化を含む）を微分し（BP）、ノードの間には PC-ALM のメッセージだけが流れる
- **1 バッチに 12 周期**、学ぶのはその 0.9 の時点: credit は 1 周期に 1 ノードずつ戻る。`Input` から `Output` までの往復は 8 周期
- **ノードが少ないほど周期も少ない**: GPU 1 枚では、全ノードの 1 周期のコストはネットワークの切り方によらずほぼ同じ。カーネルを起動する CPU が限界で、1 回の更新は約 1.7 ms かかる。残差ブロックごとに切る（11 ノード）と 1 バッチに 24 周期要り、実時間 2.5 ms の周期にノードが追いつかなかった。128 サンプルのバッチと 5 ms の周期なら、3 epoch 目は prequential 53.94%・テスト 57.67% で、時間は BP の 17 倍だった
- **GPU で実時間の 2 倍**: `device="cuda"` で、すべてのノードが GPU で計算し、ノードの間のメッセージも GPU に置かれたまま渡る（[デバイス](threads-and-devices.md#%E3%83%87%E3%83%90%E3%82%A4%E3%82%B9)）。`tau.h` は 10 ms なので周期はタイマーで 5 ms。タイマーを実時間の 2 倍で進める（`--time-scale` 2）ので、実時間では 2.5 ms。更新の間隔は平均 1.00 周期だった
- **バッチは 32**: 1 回の更新のコストは 32 サンプルでも 128 サンプルでもほぼ同じ。大きいバッチで得をするのは BP のほう
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

    予測を送るのは `pc.Output` の `next`（`pc.Chain` なら `net.next`）だけ。`Layer` の `next` が送るのは自分の予測 `mu`（`h` として）と、中継する `y`・`mask`・`reset`・`time` で、ネットワークの予測は含まない。層の `lam` を見るときは `run.state_dict()["nodes"]["layer0"]["state"]["lam"]` を読む。走っている間にノードの属性を読むと、その更新と競合する。

!!! warning "最初のサンプルが届くまで活動はゼロ"

    ホストの最初の書き込みが届くまで、ノードが受け取る活動は `time` が 0 で `mask` がすべて False。このとき `run.read(net.next)` は `mask` がすべて False の予測を返し、ノードは学ばない。書いたバッチの予測を得るには、`pc.show` のように `write` が返した時刻と `time` を照合する。

!!! warning "負荷が高いとネットワークが発散する"

    ノードはそれぞれ更新しながらメッセージをやり取りする。あるノードが遅れると、隣のノードはその古いメッセージから学び続け、ネットワークが発散するか精度が落ちる。MNIST は `y_hat` が有限でないと止まる。予測が有限のまま発散したときは、精度の低下としてしか見えない。わかっている原因は 2 つ。

    - **時間の速さが速すぎる**: 例は `--time-scale` で実行のタイマーを速める。速すぎると、最も遅いノードの計算が周期に収まらない。残差ブロックごとに切った CIFAR-10 では、周期を実時間 2 ms にすると、更新の間隔が平均 1.1〜1.3 周期になり、最大 100 周期ほど止まって、精度が約 20 ポイント落ちた。`--watch`（MNIST）で 1 epoch 回し、警告が出ず発散もしない最も速い時間の速さを選ぶ
    - **ほかのプロセスがコアを使っている**: コアを待つノードがあると、隣のノードが先に進む。32 コアの機械で時間の速さ 1 なら、busy loop 26 本と並べて 1 epoch を完走し、32 本では発散した。0.5 なら 32 本でも完走した。ほかに何も動いていない機械で回すか、時間の速さを下げる

    `max_dt` ではこれを防げない。`max_dt` は 1 回の更新で数える時間の上限で、超えた分は緩和からも次の重みの更新からも捨てられる。

!!! warning "シャッフルしたサンプルには `reset` を立てる"

    `reset` を False で書いたサンプルは、自分の credit が出力から戻るまで、各ノードで前のサンプルの `r` と `lam` を持ち続ける。シャッフルしたバッチでは、それは関係のないサンプルの credit である。それまでの間、入力に近いノードは誤った credit を持ち、早く学べばそこから学ぶ。前に続かないサンプルには、どれも `reset` を True にする。

## 自分のノードを書く

`mujou.pc.functional` にノードを構成する計算がある。`vjp` と `loss_vjp`（ノードのエネルギーの勾配）、`relax_`（`r` と `lam` の 1 ステップ）。重みは optimizer が、`r` は relaxer が動かす。複数の入力を持つノードのように `pc.Layer` で書けないノードは、これらで書く。ここに出てくる名前（`mu`・`r`・`lam`・`g_in`・`Tick`）は [mujou.pc の仕組み](pc-concepts.md)で説明している。

1. `__init__` で module を device に載せた後、クロック、bind した optimizer、ずれ `r` と双対変数 `lam`（`mu` と同じ shape のゼロ）、`r` を自作の relaxer で動かすなら bind した relaxer を属性に置く。どれもそれだけでノードの状態に入る
    - `self.clock = pc.clock.LearnClock.of(tau, period=self.period, learn_interval=...)`。`max_dt` は `pc.Layer` と同じく `tau` の既定になる
    - `self.optimizer = optimizer.bind(params)`。`params` は学ぶパラメータ。bind はノードごとに行う。bind した同じ optimizer を 2 つのノードに持たせてはいけない
    - `self.relaxer = relaxer.bind(self.r)`。同じくノードごとに bind する。置かなければ、`relax_` は素の 1 歩を進める
2. 毎回の更新で `t = self.clock.tick(dt, act["time"], self.timer.now())` を呼ぶ。`t.new` が True なら、`reset` が立ったサンプルの `r` と `lam` を 0 にし（`self.r.masked_fill_(rows, 0.0)`。`rows` は `reset` を `r` の shape に合わせたもの）、relaxer を持つなら `self.relaxer.reset_(act["reset"])` を呼ぶ
3. `delta = r + lam`（`mask` が False のサンプルは 0）と bind した `params` を同じ順で `pc.functional.vjp` に渡し、予測 `mu`、`g_in`、学ぶなら `g_theta` を求めて、`g_in` を前のノードへ送る
4. 次のノードたちが送った値の和を `g` として、`pc.functional.relax_(self.r, self.lam, g, t.dt, tau, self.relaxer)` で `r` と `lam` を動かす
5. `mu` を `h` として `pc.msg.Activity` で送る。受け取った活動の `y`・`mask`・`reset`・`time` を組にする
6. `t.learn` が True で `mask` が True のサンプルがあれば、`self.optimizer.step_([g / B for g in g_theta], t.elapsed)` で学ぶ。`B` はバッチのサンプルの数。optimizer はバッチを知らないので、`B` で割るのはノード

`pc.Layer` と同じく、`h` ではなく `r` を持ち、`mu + r` ではなく `mu` を送る。そうすれば何が `mu` を動かしても `r` と `lam` は引き継がれ、credit は後ろへだけ流れるので、どの鎖でも刻みが安定する（[mujou.pc の仕組み](pc-concepts.md#r-mu)）。

`mujou.pc` のノードとつなぐには、Port の spec に前のノードの `act_spec` か `pc.msg.activity_spec(h, y, batch)` を使う。引数は [API リファレンス](../api/pc.md)にある。PC 以外の、ノードをまたいで学ぶ則は[更新則](update-rules.md)を参照。

### 例: 入力が 2 つのノード

`examples/pc_merge.py` の `Merge` は 2 本の塔（`pc.Layer`）の活動をつないだものから予測する。毎回の更新で学ぶ（`learn_interval=None`）。

```python
--8<-- "examples/pc_merge.py:merge"
```

`pc.Layer` との違いは 1 つだけ。入力ごとに Port を持って `g_in` を分けて送り返すこと。
