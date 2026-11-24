# 🕹️ シミュレーション

## 2 つの走らせ方 {#two-ways}

|        | [`mujou.run`][mujou.run]                            | [`mujou.run_manually`][mujou.run_manually]                           |
| ------ | --------------------------------------------------- | -------------------------------------------------------------------- |
| 用途   | 現実世界への配備（センサー、ロボット）              | 環境のシミュレーション、実時間より速い学習、テスト                   |
| 時間   | 実時間（既定は `global_timer`）。時計に合わせて待つ | 仮想時間。`step` でだけ進む `VirtualTimer`。待たずに計算能力いっぱい |
| period | 各ノードが平均 `period` ごとに更新                  | 使わない。それを更新する `step` ごとに 1 回                          |
| 結果   | 実行ごとに揺らぐ                                    | 毎回ビット単位で同じ                                                 |
| 戻り値 | [`Run`][mujou.Run]                                  | [`ManualRun`][mujou.ManualRun]（`step` を持つ `Run`）                |

??? question "`mujou.run` で時間の速さを大きくするのではだめ？"

    `time_scale=10` の `Timer` は 10 倍速く進むが、ノードの更新は相変わらずマシンの都合で決まり、`dt` も結果も揺らぐ。マシンがどれだけ速くても、その倍率より速くはならない。`ManualRun` は前のステップが終わるまで時間を進めない。

## シミュレーションのループ {#loop}

[ノードを書く](../tutorial/2-node.md#chain) の `LeakyChain` に正弦波を入れ、仮想時間で 2 秒分を 1 ms 刻みで進める。1 回のループは `run.write` で次のステップの入力を書いてから `run.step(H)` を呼び、読むのはステップの間（ここでは最後のステップの後）。

```python
--8<-- "examples/simulation.py:simulate"
```

## 2 種類のステップ {#steps}

更新の順が問題になるとき（a → b の順に更新するアルゴリズムと比べるときなど）以外は `step(dt)` を使う。どちらもタイマーを `dt` 進めてから更新し、規則の全体は [`ManualRun.step`][mujou.ManualRun.step]。

=== "`run.step(h)`（Jacobi）"

    ```mermaid
    sequenceDiagram
        participant H as ホスト
        participant R as ManualRun
        participant A as net.a
        participant B as net.b
        H->>R: run.step(h)
        R->>R: 仮想のタイマーを h 進める
        par ワーカーで並列に
            R->>A: on_update(h)
        and
            R->>B: on_update(h)、ステップ開始時の a の値を読む
        end
        Note over A,B: a と b が送った値がここで見える
        R-->>H: 戻る
    ```

=== "`run.step(h, net.a, net.b)`（Gauss-Seidel）"

    ```mermaid
    sequenceDiagram
        participant H as ホスト
        participant R as ManualRun
        participant A as net.a
        participant B as net.b
        H->>R: run.step(h, net.a, net.b)
        R->>R: 仮想のタイマーを h 進める
        R->>A: on_update(h)
        A-->>B: send（すぐ見える）
        R->>B: on_update(h)、a の新しい値を読む
        R-->>H: 戻る
    ```

### 順をそろえる {#orders}

鎖 `u → a → b` を 3 通りで進めて比べる。全ノードを渡しても `step(dt)` と同じになるとは限らない。

```python
--8<-- "examples/simulation.py:orders"
```

| 進め方                  | `b` が読む `a`                     | 結果             |
| ----------------------- | ---------------------------------- | ---------------- |
| `step(h)`               | ステップ開始時の `a`               | 基準             |
| `step(h, net.b, net.a)` | 古い `a`（`b` が先に更新するため） | `step(h)` と同じ |
| `step(h, net.a, net.b)` | このステップで送られた `a`         | 違う             |

### ステップをまたぐ dt {#dt-across-steps}

<figure class="mujou-diagram">
<svg viewBox="0 0 640 170" role="img" aria-label="3 回のステップ。net.a は毎回更新して dt は h。net.b は 3 回目だけ更新し、dt は実行の開始からの 3h">
<line class="mujou-axis" x1="110" y1="150" x2="620" y2="150"/>
<text class="mujou-note" x="130" y="166" text-anchor="middle">開始</text>
<text class="mujou-note" x="270" y="166" text-anchor="middle">h</text>
<text class="mujou-note" x="410" y="166" text-anchor="middle">2h</text>
<text class="mujou-note" x="550" y="166" text-anchor="middle">3h</text>
<text class="mujou-label" x="10" y="54">net.a</text>
<line class="mujou-lane" x1="110" y1="50" x2="620" y2="50"/>
<g class="mujou-tick-a"><line x1="270" y1="38" x2="270" y2="62"/><line x1="410" y1="38" x2="410" y2="62"/><line x1="550" y1="38" x2="550" y2="62"/></g>
<text class="mujou-note" x="200" y="30" text-anchor="middle">dt = h</text>
<text class="mujou-note" x="340" y="30" text-anchor="middle">dt = h</text>
<text class="mujou-note" x="480" y="30" text-anchor="middle">dt = h</text>
<text class="mujou-label" x="10" y="114">net.b</text>
<line class="mujou-lane" x1="110" y1="110" x2="620" y2="110"/>
<g class="mujou-tick-b"><line x1="550" y1="98" x2="550" y2="122"/></g>
<path class="mujou-brace" d="M 132 98 Q 132 90 140 90 L 540 90 Q 548 90 548 98"/>
<text class="mujou-note" x="340" y="84" text-anchor="middle">dt = 3h（外れた 2 回のステップの時間も足される）</text>
</svg>
<figcaption>1・2 回目は <code>step(h, net.a)</code>、3 回目は <code>step(h)</code>。縦の線が更新</figcaption>
</figure>

## 決定的に保つ {#determinism}

次の規則を守れば `ManualRun` は毎回ビット単位で同じ結果になり、その結果はワーカーの数にもスレッドの割り当てにもよらない（ワーカーが多いときに torch のスレッドでコアを取り合わないための設定は [プロセス全体の設定](threads-and-devices.md#process-settings)）。

| 規則                                                                                        | 理由                                                                                                 |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 乱数は `seed` を渡した [`self.generator`][mujou.Node.generator] から引く                    | 既定の乱数生成器は全ノードが共有し、引く順が毎回変わる（[seed と乱数](threads-and-devices.md#seed)） |
| 時間は `step` でだけ進め、`run.timer.advance` を呼ばない                                    | ノードが想定外の `dt` を見る                                                                         |
| `write` はステップの間に呼ぶ                                                                | ステップ中の書き込みは、そのステップで届くノードと次のステップで届くノードに分かれる                 |
| ノード同士で Python のオブジェクトを共有しない                                              | 1 つのリストに追記するノードなどは、追記の順が毎回変わる                                             |
| ステップを回すスレッドで `wait_for`・`subscribe`・`run.timer.sleep` を呼ばず、`read` を使う | 次のステップまで何も送られず時間も進まないので、タイムアウトまで（無ければいつまでも）待つ           |
| 複数の実行の時間をそろえるなら、同じ `VirtualTimer` を各実行に渡す                          | 1 本の時間軸に載り、1 つの実行の `step` がほかの実行の時間も進める                                   |

```python
timer = mujou.time.VirtualTimer()  # 複数の実行で 1 本の時間軸を共有する
with (
    mujou.run_manually(a, timer=timer) as run_a,
    mujou.run_manually(b, timer=timer) as run_b,
):
    run_a.step(0.001)  # run_b のタイマーも 1 ms 進む
```

## ステップの境界はシミュレーションだけのもの {#boundaries}

更新則はステップの境界に頼らない。`run.step(dt)` でしか動かないネットワークは、`mujou.run` で配備すると壊れる（[ノードが守る規則](../concepts.md#rules)）。

|                      | `ManualRun` の 2 つのステップの間                                      | `mujou.run`                      |
| -------------------- | ---------------------------------------------------------------------- | -------------------------------- |
| 全ノードの更新       | 終えている                                                             | 各ノードが自分の時刻にばらばらに |
| 時間                 | 止まっている。境界はどのノードの一部でもなく、その間に起きる           | 進み続ける                       |
| 全ノードがそろう瞬間 | ある                                                                   | 無い                             |
| ノードが読むもの     | 自分の状態と隣のノードの最新の値だけ。`ManualRun` にいることを知らない | 同じ                             |

## 保存・再開と例外 {#save-and-errors}

状態はステップの間に `run.state_dict()` で取得し、新しいインスタンスに `mujou.run_manually(..., state=tree)` で渡せば、その後のステップは止めずに続けた実行と同じ結果になる（[ノードをテストする](../tutorial/3-test.md#restore)）。
ノードが例外を投げると `step` が `NetworkError` を送出する（[例外と終了](errors.md)、pytest での書き方は [ノードをテストする](../tutorial/3-test.md#exceptions)）。
