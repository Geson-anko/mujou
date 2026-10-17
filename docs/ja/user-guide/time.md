# 時間の制御

このページでは、時刻を読む方法、時間の上で待つ方法、時間の速さの変え方、実行ごとのタイマーの使い方を書く。`dt` に比例して更新する考え方と、一時停止しても時間が飛ばない理由は[グローバル時間](../concepts.md#%E3%82%B0%E3%83%AD%E3%83%BC%E3%83%90%E3%83%AB%E6%99%82%E9%96%93)にある。

## どこで何を使うか

| したいこと         | ホスト（既定のグローバルタイマー）                                                  | ホスト（自分の `Timer` で走らせた実行） | ノードの hook の中            |
| ------------------ | ----------------------------------------------------------------------------------- | --------------------------------------- | ----------------------------- |
| 時刻を読む         | `mujou.time.now()`（`Timestamp`）、`mujou.time.time()`（秒）、`time_ns()`（ナノ秒） | `timer.now()`                           | `self.timer.now()`            |
| 時間の上で待つ     | `mujou.time.sleep(seconds)`                                                         | `timer.sleep(seconds)`                  | できない                      |
| 時間の速さを読む   | `mujou.time.get_time_scale()`                                                       | `timer.get_time_scale()`                | `self.timer.get_time_scale()` |
| 時間の速さを変える | `mujou.time.set_time_scale(scale)`                                                  | `timer.set_time_scale(scale)`           | できない                      |

- ホストは、Python 標準の `time.sleep` ではなく、その実行のタイマーの `sleep` で待つ（入力を一定の時間見せるときなど）。Python 標準の `time.sleep` は実時間で待つので、時間の速さを無視し、実行が一時停止している間も数え続ける（状態を取る間、実行は止まる）
- `scale` は、実時間の 1 秒にタイマーが何秒進むかである（5 なら、実時間の 1 秒がタイマーの 5 秒）。有限で 0 より大きい数で、それ以外は `ValueError` になる

## Timestamp

`run.write`・`run.read`・`run.subscribe`・`run.wait_for` が返す時刻と、`mujou.time.now()`・`timer.now()`・`self.timer.now()` の値は、どれも `mujou.time.Timestamp` である。

| 書き方            | 値                                 |
| ----------------- | ---------------------------------- |
| `float(ts)`       | 秒（`float`）                      |
| `ts.ns`           | ナノ秒（`int`）。正確な値          |
| `later - earlier` | 2 つの時刻の差。これも `Timestamp` |
| `a < b` など      | 時刻の前後の比較                   |

`Timestamp` は、どのタイマーの時刻かを持たない。比べてよいのは同じタイマーの時刻同士だけである。`mujou.time.now()` と比べてよいのは、グローバルタイマーで走らせた実行の時刻だけである。

`write` が返す時刻と `read` が返す時刻を比べると、ある出力が入力を切り替えた後に送られたかを判断できる。

```python
t_in = run.write({net.prev: x})
mujou.time.sleep(0.1)
y, t_out = run.read(net.next)
if t_out >= t_in:               # y は x を書いた後に送られた
    print(float(t_out - t_in))  # x を書いてから y が送られるまでの秒数
```

## 実行ごとのタイマー

`mujou.time.Timer()` を作って `mujou.run(..., timer=timer)` に渡すと、その実行は自分のタイマーの上で走る。新しいタイマーは時刻 0、倍率 1 で始まる。

```python
--8<-- "examples/timers.py:own-timer"
```

自分のタイマーを渡すのは、次のときである。

- その実行だけ時間を速めたり遅くしたりしたい（デバッグや、実時間より速い試行）
- 状態から復元したいが、同じグローバルタイマーでほかの実行が走っている（[状態の保存と復元](state.md)）
- 互いに止め合わない別々の実行で、複数のネットワークを走らせる（[ネットワークを束ねる](compose.md)）

このとき、ホストはその実行を `timer.sleep` で待ち、`run.read` の時刻は `timer.now()` と比べる。`mujou.time` の関数（`mujou.time.sleep`、`mujou.time.time()` など）が扱うのはグローバルタイマーで、この実行のタイマーではない。そのため、この実行には使わない。

## グローバルタイマーの速さを変える

`mujou.time.set_time_scale` は、グローバルタイマーで走るすべての実行の速さを変える。

```python
--8<-- "examples/timers.py:global-scale"
```

グローバル時刻は、プロセスが初めて使ったときに 0 から始まる。

## ノードは時間を読むだけ

ノードの `self.timer` は、実行のタイマーの読み取り専用の側（`mujou.time.ReadOnlyTimer`）で、`now()` と `get_time_scale()` だけを持つ。

hook の中から `mujou.time.sleep`・`mujou.time.set_time_scale`・`Timer.sleep`・`Timer.set_time_scale`・`VirtualTimer.advance` を呼ぶと、`RuntimeError` になる。1 つのノードが時間を止めたり速さを変えたりすると、同じ実行のほかのノードの `dt` まで変わるからである。

- ノードを一定の間隔で更新したいときは、hook の中で待たずに、`super().__init__(period=seconds)` で周期を渡す。周期は、2 回の更新の間の平均の時間（実行のタイマーの秒）である（[ノードを書く](writing-a-node.md)）
- `self.timer` はどの hook の中でも読める（`on_thread_start` と `on_start` を含む）。`__init__` の中で読むと `RuntimeError` になる

## 復元すると時刻が戻る

`mujou.run(..., state=tree)` で復元すると、その実行のタイマーの時刻が、木を取った時刻に戻る。グローバルタイマーなら、`mujou.time.time()` も巻き戻る。詳しくは[状態の保存と復元](state.md)にある。

## テストでは仮想のタイマーを使う

テストで `dt` を決まった値にしたいときは、`mujou.testing.VirtualTimer` を使う。手順は[テスト](testing.md)にある。
