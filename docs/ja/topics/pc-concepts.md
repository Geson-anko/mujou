# 🔩 mujou.pc の仕組み

`mujou.pc` の部品が何をし、どうつながり、1 回の更新で何を計算するかを説明する。読み終えれば [PC-ALM](pc.md) の「自分のノードを書く」と `examples/pc_merge.py` に出てくる名前が分かる。使い方は [5. PC-ALM で学習する](../tutorial/5-learn-with-pc.md) で。

!!! abstract "このページで分かること"

    - `mujou.pc` の部品の役割と、利用者が直接使う部品・自分のノードを書くときに使う部品
    - ノードの間を流れるメッセージと Port の向き
    - `pc.Layer` が 1 回の `on_update(dt)` で行うこと
    - `pc.clock.LearnClock` が返す `Tick` と `pc.TimeConstants` の意味
    - `pc.optim.SGD` と `pc.optim.AdamW` が重みを、`pc.relax.Euler` がずれ `r` を時間で動かす仕組み

    前提は PyTorch・誤差逆伝播法（BP）と、[Mujou の仕組み](../concepts.md)のノード・Port・`dt`。predictive coding の知識は要らない。

## ノードが持つ量

PC-ALM では各ノードに自分の**活動** `h` があり、前のノードの活動から自分の活動を**予測**する。`pc.Layer` は活動の予測からの**ずれ** `r` を持ち、予測 `mu` を次のノードへ送る。理解に要る量はこれだけ。

| 量      | 式                                   | 意味                                                                         |
| ------- | ------------------------------------ | ---------------------------------------------------------------------------- |
| `h_in`  |                                      | 前のノードから届いた活動                                                     |
| `mu`    | $\mu = f(h_{\mathrm{in}};\theta)$    | 予測。ノードの `module` の出力。$\theta$ は `module` の重み                  |
| `r`     | $r = h - \mu$                        | 予測からのずれ。ノードはこれを持つ                                           |
| `h`     | $h = \mu + r$                        | 自分の活動。ノードの中にとどまり、次のノードは `mu` を読む                   |
| `lam`   | $\tau_\lambda\dot\lambda = r$        | 双対変数。ずれを時間で溜めた値                                               |
| `delta` | $\delta = r + \lambda$               | ノードの誤差                                                                 |
| `g_in`  | $g_{\mathrm{in}} = -J_h^\top \delta$ | 前のノードへ送り返す勾配（$J_h$ は `module` の `h_in` についてのヤコビアン） |
| `g`     |                                      | 次のノードたちから届いた `g_in` の和                                         |

`r` は `delta` と `g` の和を打ち消す向きに動き、時定数 `tau.r` で減衰する。`lam` は `r` が残る間溜まり続ける。両方が止まると `r = 0`（`h` は forward の値）になり、`lam` は BP の credit $-\partial\ell/\partial h$ に一致する。そのとき重みの勾配も BP と一致する。理論の詳細は [PC-ALM の記事](https://pub.sakana.ai/pc-alm/)にある。記事のノードは `h` を持って送るが、mujou のノードは `r` を持ち、`mu` を送る（下で述べる）。

### ノードが r を持ち、mu を送る理由 {#r-mu}

予測 `mu` は、新しいサンプルが来たときや、下のノードか自分が学んだときに一度に跳ぶ。`h` を持つノードは `h` を置いたままにするので、$r = h - \mu$ が急に大きくなり、偽の誤差が `g_in` と重みの勾配に入る。`r` を持つノードは、`mu` がどう跳んでもずれを保つ。

送るのは $\mu + r$ でなく `mu` である。predictive coding の**固定予測の仮定**（fixed prediction assumption）にあたる。前へ流れる活動は順伝播の値になり、credit は後ろへだけ流れる。$\mu + r$ を送ると、あるノードの `r` の動きが、順伝播のヤコビアンの積を通って上のすべてのノードの活動を動かし、credit として自分に戻ってくる。順伝播が増幅する鎖では、このループが刻みをどれだけ短くしても発散する。`mu` を送れば平衡点は変わらずにループが切れるので、1 ノードで安定な刻みはどの鎖でも安定になる。

`pc.TimeConstants` の既定（[5 節](#5-timeconstants)）なら、1 回の更新で `delta` が `-g` になり、credit は 1 周期に 1 ノードずつ戻る。

### 新しいサンプル: リセットか持ち越しか

`r` と `lam` が持つのは、ノードが見てきたサンプルの credit である。シャッフルしたバッチのように次のサンプルが無関係なら、その credit は次のサンプルには誤り。そこでホストは書くサンプルに `reset` を立て、各ノードはその書き込みを初めて見たときに、そのサンプルの `r` と `lam` を 0 にする。ストリームの次のフレームのように前のサンプルに続くものは、`reset` を False で書いて持ち越す。

## 1. 部品

!!! abstract "要点"

    利用者が普段使うのは `Chain`・`show`・`TimeConstants`・optimizer・損失・`Sample`。自分のノードを書くときに `LearnClock`・`functional`・`Activity` を使う。

| 部品                                                    | 種類       | 役割                                                                                                                      | 主に使う場面    |
| ------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `pc.Input`                                              | ノード     | ホストが書いた `x` をそのまま活動 `h` として送る。何も持たない                                                            | 手で組む        |
| `pc.Layer`                                              | ノード     | `module` を 1 つ持ち、`r` と `lam` を動かし、重みを学ぶ                                                                   | 手で組む        |
| `pc.Output`                                             | ノード     | 最後の `module` と損失を持ち、予測 `y_hat` を出し、重みを学ぶ。`r` と `lam` は持たない                                    | 手で組む        |
| `pc.Chain`                                              | Network    | `Input → Layer → … → Output` をまとめて作りつなぐ                                                                         | 普段            |
| `pc.show`                                               | 関数       | ホストがサンプルを書き、一定時間見せ、学ぶ前の予測を返す                                                                  | 普段            |
| `pc.msg.Sample`・`pc.msg.Activity`・`pc.msg.Prediction` | メッセージ | 流れるデータの型（[2 節](#2-%E3%81%A4%E3%81%AA%E3%81%8C%E3%82%8A%E3%81%A8%E6%B5%81%E3%82%8C%E3%82%8B%E3%82%82%E3%81%AE)） | `Sample` は普段 |
| `pc.TimeConstants`                                      | 設定       | `r` と `lam` の速さを秒で持つ（[5 節](#5-timeconstants)）                                                                 | 普段            |
| `pc.optim.SGD`・`pc.optim.AdamW`                        | 設定       | optimizer。重みを時間でどう動かすか（[6 節](#6-optimizer)）                                                               | 普段            |
| `pc.loss.MSELoss`・`pc.loss.CrossEntropyLoss`           | 損失       | サンプルごとの損失を返す。`Output` に渡す                                                                                 | 普段            |
| `pc.clock.LearnClock`                                   | 部品       | いつ学ぶかを決める（[4 節](#4-learnclock)）                                                                               | 自分のノード    |
| `pc.functional`                                         | 関数群     | 状態を持たない式（`vjp`・`loss_vjp`・`relax_`）。`Layer` と `Output` もこれで書く                                         | 自分のノード    |
| `pc.msg.activity_spec`                                  | 関数       | `Activity` の spec を作る                                                                                                 | 自分のノード    |

`Layer` と `Output` は `LearnClock`・`functional`・optimizer を組み合わせたもの。同じ部品で入力が 2 つのノードのような `Layer` で書けないノードも作れる。

## 2. つながりと流れるもの

!!! abstract "要点"

    前へは活動（予測 `mu`）を `Activity` で、後ろへは勾配 `g_in` を送る。目標 `y`・`mask`・`reset`・時刻 `time` は活動と同じメッセージに載せて中継する。

```mermaid
flowchart LR
    H1["ホスト"] -->|"Sample"| I["Input"]
    I -->|"Activity"| L0["Layer"]
    L0 -->|"Activity"| L1["Layer"]
    L1 -->|"Activity"| O["Output"]
    O -->|"Prediction"| H2["ホスト"]
    O -.->|"g_in"| L1
    L1 -.->|"g_in"| L0
    L0 -.->|"g_in"| I
```

実線が前向き、点線が後ろ向き。`Input` は届いた `g_in` を読まない（`x` は固定）。

| メッセージ          | フィールド                        | 流れ                         |
| ------------------- | --------------------------------- | ---------------------------- |
| `pc.msg.Sample`     | `x`、`y`、`mask`、`reset`         | ホスト → `Input.prev`        |
| `pc.msg.Activity`   | `h`、`y`、`mask`、`reset`、`time` | `next` → 次のノードの `prev` |
| `g_in`              | テンソル（`h_in` と同じ shape）   | `prev` → 前のノードの `next` |
| `pc.msg.Prediction` | `y_hat`、`y`、`mask`、`time`      | `Output.next` → ホスト       |

- `mask`: サンプルごとに目標があるか。False のサンプルは損失に入らず、`Layer` はその行の `g_in` を 0 にする。`mask` がすべて False の入力でも `r` と `lam` は緩和し、重みだけ学ばない。評価はこれで行う
- `reset`: サンプルごとに新しく始まるか。`Layer` はその書き込みを初めて見た更新で、その行の `r` と `lam` を 0 にする。シャッフルしたサンプルでは立て、前のフレームに続くフレームでは False にする
- `time`: ホストがその入力を書いた時刻（`Timestamp`）。書き込みの前は `Timestamp(ns=0)`

### Port の向き

| ノード   | Port   | 種類        | 受け取る                             | 送る         |
| -------- | ------ | ----------- | ------------------------------------ | ------------ |
| `Input`  | `prev` | `Port`      | `Sample`（ホストが書く）             | —            |
| `Input`  | `next` | `MultiPort` | `g_in`（読まない）                   | `Activity`   |
| `Layer`  | `prev` | `Port`      | `Activity`                           | `g_in`       |
| `Layer`  | `next` | `MultiPort` | 相手ごとの `g_in`（和を `g` にする） | `Activity`   |
| `Output` | `prev` | `Port`      | `Activity`                           | `g_in`       |
| `Output` | `next` | `Port`      | —                                    | `Prediction` |

`next` は `MultiPort` なので 1 つのノードの活動を複数のノードが読める。

??? question "なぜ目標を活動と同じメッセージで運ぶの？"

    ノードは非同期で、Port は最新値だけを持つ。目標を別の Port で送ると、`Output` は活動と目標を別々の瞬間に読み、別のサンプルの組になりうる。同じメッセージなら `y` は `x` と同じ経路を同じ遅れで進む。ホストが伝播より速く次のサンプルを書いても予測はいつも自分の目標と組になる。

## 3. Layer の 1 回の更新

!!! abstract "要点"

    `Layer.on_update(dt)` は「読む → `tick` → `vjp` → `g_in` を送る → `relax_` → `Activity` を送る → `optimizer.step_`」の順に進む。

| 段  | 使う部品            | 計算                                                                                                                                                                                                                                            |
| --- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `LearnClock.tick`   | `prev` から `Activity` を読み、`Tick`（`dt`・`new`・`learn`・`elapsed`）を得る。新しい入力なら（`tick.new`）、`reset` が立ったサンプルの `r` と `lam` を 0 にする                                                                               |
| 2   | `functional.vjp`    | $\mu = f(h_{\mathrm{in}})$、$g_{\mathrm{in}} = -J_h^\top\delta$、学ぶなら $g_\theta = -J_\theta^\top\delta$ を 1 回の VJP で。`g_in` を `prev` から送る                                                                                         |
| 3   | `functional.relax_` | $r \leftarrow r - \frac{dt}{\tau_h}\big((1 + \frac{\tau_h}{\tau_r})\,r + \lambda + g\big)$（既定の relaxer の `pc.relax.Euler` が動かす形。[7 節](#7-relaxer)）、続けて動かした `r` で $\lambda \leftarrow \lambda + \frac{dt}{\tau_\lambda} r$ |
| 4   | `Activity`          | 予測 $\mu$ を `h` として、受け取った `y`・`mask`・`reset`・`time` を付けて `next` から送る                                                                                                                                                      |
| 5   | `optimizer.step_`   | $g_\theta$ をサンプルの数 $B$ で割り、重みを `tick.elapsed` 秒ぶん動かす（[6 節](#6-optimizer)）。`tick.learn` が True で、`mask` に True があるときだけ                                                                                        |

- 段 2 の `delta` は段 3 の前の `r`・`lam` で作る
- 新しい入力のとき（`tick.new`）、ノードは段 1 で `relaxer.reset_(reset)` も呼ぶ（[7 節](#7-relaxer)）
- 勾配はバッチで和を取り、$B$ で割るのは段 5 だけ。だから `lam` はサンプルごとの credit になり、optimizer はバッチの平均勾配を受け取る
- `vjp` の `delta` は値ではなく、`mu` から `delta` を作る関数（`mu` は `vjp` の中で計算するため）。`r` を持つノードは `lambda mu: r + lam` を渡す
- `vjp` は `h_in` を detach してから微分するので、計算グラフは前のノードに届かない

`Output` は `r` も `lam` も持たない。段 2 で `functional.loss_vjp` を使い、損失 $\ell = \sum_b \mathrm{mask}_b\,\ell_b$ の勾配 $\partial\ell/\partial h_{\mathrm{in}}$ を `g_in` として送り、`Prediction` を送る。学ぶ条件と `optimizer.step_` は `Layer` と同じ。

??? question "g_in を前のノードへ送るのは BP と何が違うの？"

    `g_in` の計算は BP の 1 層分と同じ VJP（$-J_h^\top\delta$）で、平衡点では `lam` が BP の credit と同じ漸化式を満たす。違うのは `g_in` の扱い方。

    1. 送る側は自分の `delta` から `g_in` を作る。`delta = r + lam` は自分の状態だけで決まる
    2. 受け取る側は届いた `g_in` を足して `g` にし、段 3 の `relax_` で自分の `r` を動かす力に使う。`g` をそのまま前のノードへ流さない
    3. `r` と `lam` が動くと自分の `delta` が変わり、次の更新で送る `g_in` も変わる。credit は 1 回の逆向きの掃引でなく、各ノードの緩和を通じて時間をかけて入力側へ伝わる

    forward と backward の段階も、全ノードの待ち合わせも無い。各ノードは手元にある最新の `g`（古いこともある）で更新する。`vjp` は `h_in` を detach するので、計算グラフはノードをまたがない。BP の勾配と一致するのは平衡点だけで、学ぶまで `learn_interval` だけ待つのはこのため。

## 4. LearnClock

!!! abstract "要点"

    `r` と `lam` は毎回の更新で動かし、重みは `LearnClock` が決めた更新でだけ動かす。ノードは毎回その更新の `mu` を送るので、予測が跳んでも追加の処理は要らない。

ノードは毎回の更新で `tick = self.clock.tick(dt, act["time"], self.timer.now())` を呼ぶ。返る `Tick` の意味。

| フィールド | 意味                                                   | 使い道                             |
| ---------- | ------------------------------------------------------ | ---------------------------------- |
| `dt`       | `max_dt` で打ち切った `dt`                             | `relax_` に渡す                    |
| `new`      | 受け取った入力の `time` が前回と違う（新しいサンプル） | 新しく始まるサンプルをリセットする |
| `learn`    | この更新で学ぶか                                       | `step_` を呼ぶか決める             |
| `elapsed`  | 前に学んでからの、打ち切った `dt` の和。この更新を含む | `step_` に渡す時間                 |

`learn` が True になるのは次のとき。`learn_interval` は `Layer`・`Output`・`Chain`・`LearnClock` に渡す秒数。

| `learn_interval` | 学ぶ更新                                                                                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 正の秒数         | `time + learn_interval`、`time + 2 * learn_interval`、… の各時刻（`time` はホストが入力を書いた時刻）以降の最初の更新。ただし `new` が True の更新では学ばず、次の更新で学ぶ |
| `None`           | 毎回の更新（入力が新しくても）。入力がフレームごとに変わるストリーム向け                                                                                                     |
| `math.inf`       | 学ばない。学ばない方法はこれだけ（optimizer は `tau=math.inf` を受け付けない）                                                                                               |

- `new` が True の更新で学ばないのは、その更新の `delta` がまだ前の入力のものだから
- だから同じ入力が続く間は `learn_interval` 秒ごとに学び、`elapsed` はおよそ `learn_interval`
- 学ばないまま入力が変わると、`elapsed` に溜めた時間は捨てる
- 自分のノードでは `pc.clock.LearnClock.of(tau, period=..., learn_interval=...)` で作り、属性に置く。それだけでノードの状態に入る

??? question "なぜノードが入力を受け取った時刻でなく、ホストの書き込み時刻から数えるの？"

    経路上のノードが深さによらずほぼ同時に学ぶため。受け取った時刻から数えると深いノードほど遅れて学び、下のノードが学んだ重みにまだ応じていない `delta` で学んでしまう。どのノードも同じ実行のタイマーを読むので、バリアは要らない。

## 5. TimeConstants

!!! abstract "要点"

    速さはすべて実行のタイマーの秒で決める。各ノードは実際に経った `dt` だけ式を進めるので、周期の違うノードも時間あたり同じ速さで変化する。

`pc.TimeConstants(h=, lam=, r=)` が `r` と `lam` の時定数を持つ。3 つとも必須。重みの時定数は optimizer が持つ（[6 節](#6-optimizer)）。

| 値        | 何の速さか                | 1 回の更新で動く割合                     | `math.inf` にすると                                              |
| --------- | ------------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| `tau.h`   | ずれ `r` が緩和する速さ   | `dt / tau.h`                             | 不可（有限で正）                                                 |
| `tau.lam` | `lam` が `r` を溜める速さ | `dt / tau.lam`                           | `lam` は 0 のまま。`r` が 0 に戻らず、代わりに credit を持つ     |
| `tau.r`   | ずれ `r` が減衰する速さ   | さらに `r` の `dt / tau.r` だけ 0 へ引く | 減衰しない。安定だが、`delta` が `-g` に追いつくのに複数回かかる |

時定数から既定が決まる値が 2 つある。`tau.period`（`tau.h / (1 + tau.h / tau.r)`。`tau.r = tau.h` なら `tau.h / 2`）はノードの更新の間隔で、`period=` を渡さないときの `Chain` の全ノードの周期。`tau.max_dt`（`min(tau.period, 2 * tau.lam)`）は 1 回の更新で数える `dt` の上限で、超えた分は緩和からも `elapsed` からも捨てる。

ふつうの値は組で決める。`tau.r = tau.h`、`tau.lam = tau.h / 2`、既定の周期 `tau.h / 2`。このとき 1 回の更新で `delta = r + lam` がちょうど `-g` になり、credit は 1 周期に 1 ノードずつ戻って行き過ぎない。刻みを長くすると行き過ぎ、その行き過ぎは通るノードごとに大きくなる。`tau.r` は平衡点を動かさない。`lam` が `r` を溜めるので、平衡点では `tau.r` によらず `r = 0` になる。`tau.lam` と `tau.r` を両方 `math.inf` にすると、ノードは固定予測の仮定を置いた predictive coding になり、誤差 `r` が BP の credit に落ち着く。ノードの周期は自分の `max_dt` より長くしてはいけない（`ValueError`）。

optimizer の `tau` は `tau.h` よりずっと大きくする。重みが活動の落ち着く速さよりゆっくり動けば、推論と学習を同時に回しても重みは落ち着いた後の `delta` で学ぶ。値の決め方は [時定数を決める](pc.md#%E6%99%82%E5%AE%9A%E6%95%B0%E3%82%92%E6%B1%BA%E3%82%81%E3%82%8B)、引数の一覧は [API リファレンス](../api/pc.md)。

??? question "なぜ dt を max_dt で打ち切るの？"

    歩みは陽的なので、既定よりずっと長い刻みは行き過ぎ、`r` と `lam` が発散する。負荷でノードが止まると次の `dt` が伸びるので、1 回に数える時間を打ち切り、超えた分は捨てる。

## 6. optimizer

!!! abstract "要点"

    重みも秒で動く。optimizer は時間に沿った流れで、時定数を秒で持ち、1 歩ごとにその歩みが受け持つ `elapsed` 秒だけ進む。だから学ぶ頻度を変えても、重みが 1 秒あたりに動く速さは変わらない。

### 設定を各ノードで束縛する

`pc.optim.SGD(tau=)` と `pc.optim.AdamW(tau=, tau_m=, tau_v=, eps=1e-8, weight_decay=1e-2)` は設定で、状態を持たない frozen な値。`Layer`・`Output`・`Chain` は必須の引数 `optimizer=` で受け取る。各ノードは設定を自分の勾配を取る重みに束縛し、**束縛した optimizer** を `self.optimizer` に置く。状態を持ち、重みを動かすのは束縛した optimizer。

```mermaid
flowchart LR
    S["pc.optim.AdamW(...)<br/>設定、状態なし"] -->|"bind(params)"| B0["Layer 0: 束縛した optimizer<br/>m, v, c_m, c_v"]
    S -->|"bind(params)"| B1["Layer 1: 束縛した optimizer<br/>m, v, c_m, c_v"]
    S -->|"bind(params)"| B2["Output: 束縛した optimizer<br/>m, v, c_m, c_v"]
```

```python
self.optimizer = optimizer.bind(params)  # in __init__, with only the weights that learn
# in on_update, when the node learns:
self.optimizer.step_([g / B for g in g_theta], tick.elapsed)
```

- 1 つの設定を全ノードに渡してよい（`Chain` はそうする）。状態はノードごとに、束縛した optimizer の中にある
- 束縛した optimizer はノードの状態のキー `"optimizer"` に入る（`LearnClock` が `"clock"` に入るのと同じ）
- ノードが $g_\theta$ を $B$ で割ってから `step_` に渡す。optimizer はバッチを知らない
- `elapsed == 0` の歩みは重みも状態も動かさない

### SGD と AdamW

平均の勾配を $g = g_\theta / B$ とする。

| optimizer        | 時間に沿った流れ                                                                                                                                    | `elapsed` $= \Delta t$ 秒の 1 歩                                                                                                      |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `pc.optim.SGD`   | $\tau_\theta \dot\theta = -g$                                                                                                                       | $\theta \leftarrow \theta - \frac{\Delta t}{\tau_\theta} g$。学習率 $\Delta t / \tau_\theta$ の SGD                                   |
| `pc.optim.AdamW` | $\tau_m \dot m = g - m$、$\tau_v \dot v = g^2 - v$、$\tau_\theta \dot\theta = -\left(\frac{\hat m}{\sqrt{\hat v} + \varepsilon} + w\,\theta\right)$ | $a_m = 1 - e^{-\Delta t/\tau_m}$ で $m \leftarrow m + a_m (g - m)$、$v$ も同様。続けて $\eta = \Delta t/\tau_\theta$ で AdamW の 1 歩 |

- `tau` は $\tau_\theta$、`tau_m` は $\tau_m$、`tau_v` は $\tau_v$ で、どれも秒。有限で > 0。学ばないならノードに `learn_interval=math.inf` を渡す
- `eps`（$\varepsilon$）は > 0。勾配が 0 のままだった重みが `0 / 0` にならないようにするため
- `weight_decay`（$w$）は無次元。$\theta$ を時定数 $\tau_\theta / w$ で減衰させる。`torch.optim.AdamW` の `weight_decay` が `lr` と組むのと同じ
- AdamW は勾配の大きさによらず、1 歩で各重みをおよそ $\eta$ ずつ動かす。Adam の `lr` に合わせるなら、2 回の学習の間の時間を $\Delta t$ とし（1 入力につき 1 回学ぶなら入力を見せる時間）、`tau` を $\Delta t$ / `lr` にする

### 質量 c: 時間で行う bias 補正

$m$ と $v$ は 0 から始まるので、はじめは小さすぎる。`torch.optim.AdamW` はステップ数 $t$ で補正し、$m$ を $1 - \beta_1^t$ で割る。`pc.optim.AdamW` はステップを数えない。代わりに moment ごとに**質量** $c$ を持つ。質量はその moment のうちこれまでの勾配が占める割合（残りは初めの 0）で、moment と同じ時定数で動く。

$$\tau_m \dot c_m = 1 - c_m,\quad c_m(0) = 0,\qquad \hat m = m / c_m$$

1 歩は moment と同じように $c_m \leftarrow c_m + a_m(1 - c_m)$ で質量を動かす。$t$ 秒ぶんの勾配の後は $c_m = 1 - e^{-t/\tau_m}$。$g$ が一定なら $m = c_m g$ なので、最初の 1 歩から $\hat m = g$。$v$ も自分の質量 $c_v$ を同じように持つ。

1 歩は、$g$ が $\Delta t$ の間一定だとして流れを厳密に解く。だから moment と質量は歩みの回数ではなく歩みの秒数の合計で決まる。$\tau_m = 0.1$ 秒のときは次のとおり。

| 歩み           | `pc.optim.AdamW` の $c_m$ | 0.05 秒の歩みに合わせて $\beta_1 = e^{-0.05/0.1}$ に固定し、ステップを数えると |
| -------------- | ------------------------- | ------------------------------------------------------------------------------ |
| 0.05 秒を 2 回 | $1 - e^{-1} = 0.632$      | $1 - \beta_1^2 = 0.632$                                                        |
| 0.1 秒を 1 回  | $1 - e^{-1} = 0.632$      | $1 - \beta_1^1 = 0.393$                                                        |

$\Delta t$ が一定なら、歩みは `torch.optim.AdamW(lr=Δt / tau, betas=(exp(-Δt / tau_m), exp(-Δt / tau_v)))` と同じになる。だから torch の $\beta$ からは $\tau_m = -\Delta t / \ln\beta_1$ と決まる。$\Delta t$ はここでも 2 回の学習の間の時間。32 ms ごとに学ぶなら、$\beta_1 = 0.9$ は約 0.30 秒、$\beta_2 = 0.999$ は約 32 秒。SGD と AdamW の選び方と値は [optimizer を選ぶ](pc.md#optimizer-%E3%82%92%E9%81%B8%E3%81%B6) へ。

??? question "なぜ torch.optim を使わないの？"

    1. ステップを数え、時間を数えない。$\beta$ と `lr` が 1 ステップあたりなので、1 秒あたりの速さが `learn_interval` や歩みごとに異なる `elapsed` で変わる
    2. ステップ数が float32 のテンソルで、$2^{24}$ 回で増えなくなる（24 ms ごとに学ぶと約 4.7 日）。mujou は学び続ける機械のためのもの
    3. 作るときに `torch._dynamo` を import し、free-threaded の Python（3.14t）でプロセス全体の GIL を有効に戻す

### 自分の optimizer

自作の optimizer は 2 つのクラスで書く。設定は `pc.optim.Optimizer` を継承して `bind(params)` を実装する。`bind` が返すものは `pc.optim.BoundOptimizer`（`mujou.Stateful` の一種）を継承し、`step_(grads, elapsed)`・`state_dict()`・`load_state_dict(state)` を実装する。どれかが欠けたクラスは、作るときに `TypeError` になる。`bind` には学ぶ重みだけを渡す。例は [API リファレンス](../api/pc.md) にある。

## 7. relaxer

!!! abstract "要点"

    ずれ `r` が自分に掛かる力に沿ってどう動くかも設定である。relaxer は `r` にとっての optimizer にあたる。状態を持たない設定をノードごとに bind し、`r` を `dt` 秒だけ動かす。

mujou の relaxer は `pc.relax.Euler()` の 1 つだけで、ほかに選べるのは自作の relaxer である。どちらも設定である。`Layer` と `Chain` が `relaxer=` で 1 つ取る。既定は `pc.relax.Euler()` である。`Chain` は同じ設定を各 `Layer` に渡す。設定は状態を持たないので、共有しても安全である。`Output` は `r` を持たないので取らない。各 `Layer` は設定を自分の `r` に bind し、**bind した relaxer** を `self.relaxer` に持つ。bind した relaxer が状態を持ち、`r` を動かす。

```python
self.relaxer = relaxer.bind(self.r)  # __init__ で
# on_update の functional.relax_ の中で、力 F を渡して:
self.relaxer.step_(r, force, dt, tau.h)
# 新しい入力のとき、reset が立ったサンプルについて:
self.relaxer.reset_(act["reset"])
```

力を $F = (1 + \tau_h / \tau_r)\,r + \lambda + g$ とすると、`pc.relax.Euler` の `dt` $= \Delta t$ 秒の 1 回の動きは $r \leftarrow r - \frac{\Delta t}{\tau_h} F$ である。

- relaxer が変えるのは `r` が不動点に着くまでの道で、不動点ではない。不動点では $F = 0$ で、relaxer の状態は静止している。relaxer は $\lambda$ を動かさない。$\lambda$ は、どの relaxer が `r` を動かしても、`r` が動いたあとにその `r` で進む（[3 節](#3-layer-%E3%81%AE-1-%E5%9B%9E%E3%81%AE%E6%9B%B4%E6%96%B0)）
- 新しい入力のとき、ノードは `reset` が立ったサンプルの $r$ と $\lambda$ を 0 にし、`reset_(reset)` を呼ぶ。力を覚える relaxer（速度を持つものなど）が覚えているのは前のサンプルの力なので、そのサンプルの分を捨てる。前に続くサンプルでは、$r$・$\lambda$・relaxer の状態を保つ。予測が跳んだとき（自分や下のノードの重みが跳んだとき）は $r$ も力も跳ばないので、`reset_` は呼ばない
- bind した relaxer はノードの状態のキー `"relaxer"` に入る。`pc.relax.Euler` は状態を持たないので空で、その `reset_` は何もしない
- relaxer と値の選び方は[relaxer を選ぶ](pc.md#relaxer-%E3%82%92%E9%81%B8%E3%81%B6)を見る

### 自分の relaxer

自作の relaxer は 2 つのクラスで書く。設定は `pc.relax.Relaxer` を継承して `bind(r)` を実装し、`r` と同じ shape・dtype・device の状態を作る。`bind` が返すものは `pc.relax.BoundRelaxer`（`mujou.Stateful` の一種）を継承し、`step_(r, force, dt, tau_h)`・`reset_(samples)`・`state_dict()`・`load_state_dict(state)` を実装する。どれかが欠けたクラスは、作るときに `TypeError` になる。不動点は力が 0 で、relaxer 自身の状態が静止しているところでなければならない。そうでないと `lam` は BP の credit にならない。例は [API リファレンス](../api/pc.md)にある。
