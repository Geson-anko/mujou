# 🔩 mujou.pc の仕組み

`mujou.pc` の部品が何をし、どうつながり、1 回の更新で何を計算するかを説明する。読み終えれば [PC-ALM](pc.md) の「自分のノードを書く」と `examples/pc_merge.py` に出てくる名前が分かる。使い方は [5. PC-ALM で学習する](../tutorial/5-learn-with-pc.md) で。

!!! abstract "このページで分かること"

    - `mujou.pc` の部品の役割と、利用者が直接使う部品・自分のノードを書くときに使う部品
    - ノードの間を流れるメッセージと Port の向き
    - `pc.Layer` が 1 回の `on_update(dt)` で行うこと
    - `pc.LearnClock` が返す `Tick`、`jumps`、`pc.TimeConstants` の意味

    前提は PyTorch・誤差逆伝播法（BP）と、[Mujou の仕組み](../concepts.md)のノード・Port・`dt`。predictive coding の知識は要らない。

## ノードが持つ量

PC-ALM では各ノードが自分の**活動** `h` を持ち、前のノードの活動から自分の活動を**予測**する。理解に要る量はこれだけ。

| 量      | 式                                   | 意味                                                                         |
| ------- | ------------------------------------ | ---------------------------------------------------------------------------- |
| `h_in`  |                                      | 前のノードから届いた活動                                                     |
| `mu`    | $\mu = f(h_{\mathrm{in}};\theta)$    | 予測。ノードの `module` の出力。$\theta$ は `module` の重み                  |
| `h`     |                                      | 自分の活動。次のノードへ送る                                                 |
| `lam`   | $\tau_\lambda\dot\lambda = h - \mu$  | 双対変数。予測からのずれ $r = h - \mu$ を時間で溜めた値                      |
| `delta` | $\delta = h - \mu + \lambda$         | ノードの誤差                                                                 |
| `g_in`  | $g_{\mathrm{in}} = -J_h^\top \delta$ | 前のノードへ送り返す勾配（$J_h$ は `module` の `h_in` についてのヤコビアン） |
| `g`     |                                      | 次のノードたちから届いた `g_in` の和                                         |

`h` は `delta` と `g` の和を打ち消す向きに動き、`lam` は `r` が残る間溜まり続ける。両方が止まると `r = 0`（`h` は forward の値）になり、`lam` は BP の credit $-\partial\ell/\partial h$ に一致する。そのとき重みの勾配も BP と一致する。理論の詳細は [PC-ALM の記事](https://pub.sakana.ai/pc-alm/)にある。

## 1. 部品

!!! abstract "要点"

    利用者が普段使うのは `Chain`・`show`・`TimeConstants`・損失・`Sample`。自分のノードを書くときに `LearnClock`・`functional`・`Activity` を使う。

| 部品                                        | 種類       | 役割                                                                                                                      | 主に使う場面    |
| ------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `pc.Input`                                  | ノード     | ホストが書いた `x` をそのまま活動 `h` として送る。`h` も重みも持たない                                                    | 手で組む        |
| `pc.Layer`                                  | ノード     | `module` を 1 つ持ち、`h` と `lam` を動かし、重みを学ぶ                                                                   | 手で組む        |
| `pc.Output`                                 | ノード     | 最後の `module` と損失を持ち、予測 `y_hat` を出し、重みを学ぶ。`h` と `lam` は持たない                                    | 手で組む        |
| `pc.Chain`                                  | Network    | `Input → Layer → … → Output` をまとめて作りつなぐ                                                                         | 普段            |
| `pc.show`                                   | 関数       | ホストがサンプルを書き、一定時間見せ、学ぶ前の予測を返す                                                                  | 普段            |
| `pc.Sample`・`pc.Activity`・`pc.Prediction` | メッセージ | 流れるデータの型（[2 節](#2-%E3%81%A4%E3%81%AA%E3%81%8C%E3%82%8A%E3%81%A8%E6%B5%81%E3%82%8C%E3%82%8B%E3%82%82%E3%81%AE)） | `Sample` は普段 |
| `pc.TimeConstants`                          | 設定       | 速さを秒で持つ（[5 節](#5-timeconstants)）                                                                                | 普段            |
| `pc.MSELoss`・`pc.CrossEntropyLoss`         | 損失       | サンプルごとの損失を返す。`Output` に渡す                                                                                 | 普段            |
| `pc.LearnClock`                             | 部品       | いつ学ぶかを決める（[4 節](#4-learnclock-%E3%81%A8-jumps)）                                                               | 自分のノード    |
| `pc.functional`                             | 関数群     | 状態を持たない式（`vjp`・`loss_vjp`・`relax_`・`learn_`）。`Layer` と `Output` もこれで書く                               | 自分のノード    |
| `pc.activity_spec`                          | 関数       | `Activity` の spec を作る                                                                                                 | 自分のノード    |

`Layer` と `Output` は `LearnClock` と `functional` を組み合わせたもの。同じ部品で入力が 2 つのノードのような `Layer` で書けないノードも作れる。

## 2. つながりと流れるもの

!!! abstract "要点"

    前へは活動を `Activity` で、後ろへは勾配 `g_in` を送る。目標 `y`・`mask`・時刻 `time` は活動と同じメッセージに載せて中継する。

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

| メッセージ      | フィールド                        | 流れ                         |
| --------------- | --------------------------------- | ---------------------------- |
| `pc.Sample`     | `x`、`y`、`mask`                  | ホスト → `Input.prev`        |
| `pc.Activity`   | `h`、`jumps`、`y`、`mask`、`time` | `next` → 次のノードの `prev` |
| `g_in`          | テンソル（`h_in` と同じ shape）   | `prev` → 前のノードの `next` |
| `pc.Prediction` | `y_hat`、`y`、`mask`、`time`      | `Output.next` → ホスト       |

- `mask`: サンプルごとに目標があるか。False のサンプルは損失に入らず、`Layer` はその行の `g_in` を 0 にする。`mask` がすべて False の入力でも `h` と `lam` は緩和し、重みだけ学ばない。評価はこれで行う
- `time`: ホストがその入力を書いた時刻（`Timestamp`）。書き込みの前は `Timestamp(ns=0)`
- `jumps`: 送り手の `h` が跳んだ回数（`int`）（[4 節](#4-learnclock-%E3%81%A8-jumps)）

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

    `Layer.on_update(dt)` は「読む → `tick` → `vjp` → `g_in` を送る → `relax_` → `Activity` を送る → `learn_`」の順に進む。

| 段  | 使う部品            | 計算                                                                                                                                                                            |
| --- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `LearnClock.tick`   | `prev` から `Activity` を読み、`Tick`（`dt`・`new`・`learn`・`elapsed`）を得る                                                                                                  |
| 2   | —                   | 前の更新で自分の重みを学んでいれば、その跳びを `h` に足す（段 4 の付け替えとは別。[jumps](#jumps) の表の 3 行目）                                                               |
| 3   | `functional.vjp`    | $\mu = f(h_{\mathrm{in}})$、$g_{\mathrm{in}} = -J_h^\top\delta$、学ぶなら $g_\theta = -J_\theta^\top\delta$ を 1 回の VJP で。`g_in` を `prev` から送る                         |
| 4   | `functional.relax_` | 跳びがあれば `h` を付け替え、$h \leftarrow h - \frac{dt}{\tau_h}(h - \mu + \lambda + g)$、続けて動かした `h` で $\lambda \leftarrow \lambda + \frac{dt}{\tau_\lambda}(h - \mu)$ |
| 5   | `Activity`          | 必要なら自分の `jumps` を進め、`h` と `jumps` に受け取った `y`・`mask`・`time` を付けて `next` から送る                                                                         |
| 6   | `functional.learn_` | $\theta \leftarrow \theta - \frac{\mathrm{elapsed}}{\tau_\theta}\cdot\frac{g_\theta}{B}$。`tick.learn` が True で、`mask` に True があるときだけ                                |

- 段 3 の `delta` は段 2 の後、段 4 の前の `h`・`lam` で作る
- 勾配はバッチで和を取り、サンプルの数 $B$ で割るのは `learn_` だけ。だから `lam` はサンプルごとの credit になる。1 回の学習は学習率 `elapsed / tau.theta` でバッチの平均勾配を下る SGD に当たる
- `vjp` の `delta` は値ではなく、`mu` から `delta` を作る関数（`mu` は `vjp` の中で計算するため）。ふだんは `lambda mu: h - mu + lam`。段 4 で付け替える更新では、付け替え前のずれ `h - mu_last` を使い `lambda mu: (h - mu_last) + lam` を渡す（[jumps](#jumps)）
- `vjp` は `h_in` を detach してから微分するので、計算グラフは前のノードに届かない

`Output` は `h` も `lam` も持たない。段 3 で `functional.loss_vjp` を使い、損失 $\ell = \sum_b \mathrm{mask}_b\,\ell_b$ の勾配 $\partial\ell/\partial h_{\mathrm{in}}$ を `g_in` として送り、`Prediction` を送る。学ぶ条件と `learn_` は `Layer` と同じ。付け替える `h` が無いので `jumps` は読まない。

??? question "g_in を前のノードへ送るのは BP と何が違うの？"

    `g_in` の計算は BP の 1 層分と同じ VJP（$-J_h^\top\delta$）で、平衡点では `lam` が BP の credit と同じ漸化式を満たす。違うのは `g_in` の扱い方。

    1. 送る側は自分の `delta` から `g_in` を作る。`delta = h - mu + lam` は自分の状態だけで決まる
    2. 受け取る側は届いた `g_in` を足して `g` にし、段 4 の `relax_` で自分の `h` を動かす力に使う。`g` をそのまま前のノードへ流さない
    3. `h` と `lam` が動くと自分の `delta` が変わり、次の更新で送る `g_in` も変わる。credit は 1 回の逆向きの掃引でなく、各ノードの緩和を通じて時間をかけて入力側へ伝わる

    forward と backward の段階も、全ノードの待ち合わせも無い。各ノードは手元にある最新の `g`（古いこともある）で更新する。`vjp` は `h_in` を detach するので、計算グラフはノードをまたがない。BP の勾配と一致するのは平衡点だけで、学ぶまで `learn_interval` だけ待つのはこのため。

## 4. LearnClock と jumps

!!! abstract "要点"

    `h` と `lam` は毎回の更新で動かし、重みは `LearnClock` が決めた更新でだけ動かす。`jumps` は「予測が跳んだので `h` も一緒に動かせ」という次のノードへの合図。

### LearnClock

ノードは毎回の更新で `tick = self.clock.tick(dt, act["time"].ns, self.timer.now().ns)` を呼ぶ。返る `Tick` の意味。

| フィールド | 意味                                                   | 使い道                  |
| ---------- | ------------------------------------------------------ | ----------------------- |
| `dt`       | `max_dt` で打ち切った `dt`                             | `relax_` に渡す         |
| `new`      | 受け取った入力の `time` が前回と違う（新しいサンプル） | `h` を付け替える        |
| `learn`    | この更新で学ぶか                                       | `learn_` を呼ぶか決める |
| `elapsed`  | 前に学んでからの、打ち切った `dt` の和。この更新を含む | `learn_` に渡す重み     |

`learn` が True になるのは次のとき。`learn_interval` は `Layer`・`Output`・`Chain`・`LearnClock` に渡す秒数。

| `learn_interval` | 学ぶ更新                                                                                                                                |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 正の秒数         | 書き込み時刻 `time` の `learn_interval` 後、2 倍後、… の各時刻を過ぎた最初の更新。ただし `new` が True の更新では学ばず、次の更新で学ぶ |
| `None`           | 毎回の更新（入力が新しくても）。入力がフレームごとに変わるストリーム向け                                                                |
| `math.inf`       | 学ばない。`tau.theta` が `math.inf` のとき `LearnClock.of` もこうする                                                                   |

- `new` が True の更新で学ばないのは、その更新の `delta` がまだ前の入力のものだから
- 学ばないまま入力が変わると、`elapsed` に溜めた時間は捨てる
- 自分のノードでは `pc.LearnClock.of(tau, period=..., learn_interval=...)` で作り、属性に置く。それだけでノードの状態に入る

??? question "なぜノードが入力を受け取った時刻でなく、ホストの書き込み時刻から数えるの？"

    経路上のノードが深さによらずほぼ同時に学ぶため。受け取った時刻から数えると深いノードほど遅れて学び、前のノードが学んで起きた `h` の跳びが入った `delta` で学んでしまう（発散した）。どのノードも同じ実行のタイマーを読むので、バリアは要らない。

### jumps

緩和は `mu` が少しずつ動くことを前提にしている。`mu` が一度に跳ぶと、`h` を置いたままでは $r = h - \mu$ が急に大きくなり、偽の誤差が `g_in` と重みの勾配に入る。そこで `mu` が跳んだ更新では `h` も同じだけ 1 回動かし（付け替え）、`r` と `lam` を保つ。

$$h \leftarrow \mu_{\text{new}} + (h - \mu_{\text{last}})$$

`mu_last` は前回の更新の予測。付け替えた `h` も跳ぶので、自分の `jumps` を 1 進めて次のノードに知らせる。

| `mu` が跳ぶ原因           | ノードの気づき方                | `h` の動かし方                                                             | 自分の `jumps`                           |
| ------------------------- | ------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------- |
| 新しいサンプル            | `tick.new` が True              | 付け替え                                                                   | +1                                       |
| 前のノードの `h` が跳んだ | 受け取った `jumps` が前回と違う | 付け替え                                                                   | +1                                       |
| 自分の重みを学んだ        | 自分で学んだ                    | 次の更新の最初に跳び $f(h_{\mathrm{in}};\theta_{\text{new}}) - \mu$ を足す | `learn_interval` が `None` でなければ +1 |

```mermaid
flowchart LR
    I["Input<br/>書き込みで jumps +1"] -->|"jumps が変わる"| L0["Layer<br/>付け替え、jumps +1"]
    L0 -->|"jumps が変わる"| L1["Layer<br/>付け替え、jumps +1"]
    L1 --> O["Output<br/>jumps を読まない"]
```

- 原因が同じ更新で重なっても、付け替えも `jumps` も 1 回だけ
- `Input` はホストの書き込みのたびに `jumps` を進める
- `learn_interval=None` では重みは毎回少しずつ動くので、跳びを自分の `h` に足すだけで `jumps` は進めない。毎回進めると次のノードが毎回付け替え、深い鎖が発散する
- 付け替えるので、新しいサンプルで `h` や `lam` をリセットしない。持ち越すのは `r` と `lam` だけ

## 5. TimeConstants

!!! abstract "要点"

    速さはすべて実行のタイマーの秒で決める。各ノードは実際に経った `dt` だけ式を進めるので、周期の違うノードも時間あたり同じ速さで変化する。

`pc.TimeConstants(h=, lam=, theta=)` が 3 つの時定数を持つ。

| 値          | 何の速さか                | 1 回の更新で動く割合         | `math.inf` にすると                                                                        |
| ----------- | ------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------ |
| `tau.h`     | 活動 `h` が緩和する速さ   | `dt / tau.h`                 | 不可（有限で正）                                                                           |
| `tau.lam`   | `lam` が `r` を溜める速さ | `dt / tau.lam`               | `lam` は 0 のまま。`r` が 0 に戻らず、平衡点で BP と一致しない（ただの predictive coding） |
| `tau.theta` | 重みが学ぶ速さ            | 学習率 `elapsed / tau.theta` | 学ばない                                                                                   |

時定数から既定が決まる値が 2 つある。`tau.period`（`tau.h / 4`）はノードの更新の間隔で、`Chain` の全ノードの周期。`tau.max_dt`（`min(tau.h / 2, 3 * tau.lam)`）は 1 回の更新で数える `dt` の上限で、超えた分は緩和からも `elapsed` からも捨てる。

`tau.theta` は `tau.h` よりずっと大きくする。重みが活動の落ち着く速さよりゆっくり動けば、推論と学習を同時に回しても重みは落ち着いた後の `delta` で学ぶ。値の決め方は [時定数を決める](pc.md#%E6%99%82%E5%AE%9A%E6%95%B0%E3%82%92%E6%B1%BA%E3%82%81%E3%82%8B)、引数の一覧は [API リファレンス](../api/pc.md)。

??? question "なぜ dt を max_dt で打ち切るの？"

    `g` は次のノードが最後に更新したときの `h` から作られ、ノード同士は 1 歩遅れで結合している。負荷で全ノードが止まった後に `dt` が一斉に伸びると、この結合でネットワークが発散する。ノードの内側で刻みを分割しても防げないので 1 回に数える時間を打ち切る。
