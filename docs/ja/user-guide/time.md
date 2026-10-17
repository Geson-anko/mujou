# 時間の制御

このページでは、時刻を読む方法、時間の上で待つ方法、時間の速さの変え方、実行ごとのタイマーの使い方を書く。`dt` に比例して更新する考え方と、一時停止しても時間が飛ばない理由は[グローバル時間](../concepts.md#%E3%82%B0%E3%83%AD%E3%83%BC%E3%83%90%E3%83%AB%E6%99%82%E9%96%93)にある。

## どこで何を使うか

| したいこと             | ホスト（既定のグローバルタイマー）           | ホスト（自分の `Timer` で走らせた実行） | ノードの hook の中            |
| ---------------------- | -------------------------------------------- | --------------------------------------- | ----------------------------- |
| 時刻を読む             | `mujou.time.now()`（`Timestamp`）            | `timer.now()`                           | `self.timer.now()`            |
| 時間の上で待つ         | `mujou.time.sleep(seconds)`                  | `timer.sleep(seconds)`                  | できない                      |
| 時間の速さを読む       | `mujou.time.get_time_scale()`                | `timer.get_time_scale()`                | `self.timer.get_time_scale()` |
| 時間の速さを変える     | `mujou.time.set_time_scale(scale)`           | `timer.set_time_scale(scale)`           | できない                      |
| 時間を止める・再開する | `mujou.time.pause()` / `mujou.time.resume()` | `timer.pause()` / `timer.resume()`      | できない                      |
| 止まっているかを見る   | `mujou.time.is_paused()`                     | `timer.is_paused()`                     | 持たない                      |

- ホストは、Python 標準の `time.sleep` ではなく、その実行のタイマーの `sleep` で待つ（入力を一定の時間見せるときなど）。Python 標準の `time.sleep` は実時間で待つので、時間の速さを無視し、実行が一時停止している間も数え続ける（状態を取る間、実行は止まる）
- `scale` は、実時間の 1 秒にタイマーが何秒進むかである（5 なら、実時間の 1 秒がタイマーの 5 秒）。有限で 0 より大きい数で、それ以外は `ValueError` になる

## Timestamp

`run.write`・`run.read_with_time`・`run.subscribe`・`run.wait_for` が返す時刻と、`mujou.time.now()`・`timer.now()`・`self.timer.now()` の値は、どれも `mujou.time.Timestamp` である。

| 書き方            | 値                                                                       |
| ----------------- | ------------------------------------------------------------------------ |
| `ts.float()`      | 秒（`float`）。`float(ts)` も同じ                                        |
| `ts.ns`           | ナノ秒（`int`）。正確な値                                                |
| `later - earlier` | `earlier` から `later` までの秒（`float`）。`later` の方が前なら負になる |
| `a < b` など      | 時刻の前後の比較                                                         |

`Timestamp` は、どのタイマーの時刻かを持たない。比べてよいのは同じタイマーの時刻同士だけである。`mujou.time.now()` と比べてよいのは、グローバルタイマーで走らせた実行の時刻だけである。

`write` が返す時刻と `read_with_time` が返す時刻を比べると、ある出力が入力を切り替えた後に送られたかを判断できる。

```python
t_in = run.write({net.prev: x})
mujou.time.sleep(0.1)
y, t_out = run.read_with_time(net.next)
if t_out >= t_in:               # y は x を書いた後に送られた
    print(t_out - t_in)  # x を書いてから y が送られるまでの秒数
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

このとき、ホストはその実行を `timer.sleep` で待ち、`run.read_with_time` の時刻は `timer.now()` と比べる。実行のタイマーは `run.timer` でも取れる（渡さなかった実行ではグローバルタイマー）。`mujou.time` の関数（`mujou.time.sleep`、`mujou.time.now()` など）が扱うのはグローバルタイマーで、この実行のタイマーではない。そのため、この実行には使わない。

## グローバルタイマーの速さを変える

`mujou.time.set_time_scale` は、グローバルタイマーで走るすべての実行の速さを変える。

```python
--8<-- "examples/timers.py:global-scale"
```

グローバル時刻は、プロセスが初めて使ったときに 0 から始まる。

## 時間を止める

`timer.pause()` は、`timer.resume()` を呼ぶまでそのタイマーの時間を止める。グローバルタイマーなら `mujou.time.pause()` と `mujou.time.resume()` を使う。同じタイマーで走る複数の実行の状態を取るときなど、ホストが 1 つの時刻でまとめて何かをしたいときに使う（[状態の保存と復元](state.md#%E8%A4%87%E6%95%B0%E3%81%AE%E5%AE%9F%E8%A1%8C%E3%81%AE%E7%8A%B6%E6%85%8B%E3%82%92%E5%90%8C%E3%81%98%E6%99%82%E5%88%BB%E3%81%A7%E5%8F%96%E3%82%8B)）。

時間が止まっている間は、次のようになる。

- そのタイマーで走るすべての実行のノードが更新しない。時間が再び進むまで待ち、再開後の最初の `dt` に止まっていた時間は入らない
- そのタイマーの `sleep` は、時間が再び進むまで戻らない
- `VirtualTimer.advance` は `RuntimeError` になる
- `run.state_dict()` は取れる。止めた時刻の状態が返る

`pause` は回数を数えないスイッチである。2 回呼んでも、`resume` は 1 回でよい。状態の取得も、このスイッチとは別にタイマーを止める。そのため `resume` を呼んでも取得中の停止は解けず、取得が終わってもホストの停止は解けない。`is_paused()` は、どちらかで止まっていれば真になる。

## ノードは時間を読むだけ

ノードの `self.timer` は、実行のタイマーの読み取り専用の側（`mujou.time.ReadOnlyTimer`）で、`now()` と `get_time_scale()` だけを持つ。

hook の中から `mujou.time` や `Timer` の `sleep`・`set_time_scale`・`pause`・`resume`、`VirtualTimer.advance` を呼ぶと、`RuntimeError` になる。1 つのノードが時間を止めたり速さを変えたりすると、同じ実行のほかのノードの `dt` まで変わるからである。

- ノードを一定の間隔で更新したいときは、hook の中で待たずに、`super().__init__(period=seconds)` で周期を渡す。周期は、2 回の更新の間の平均の時間（実行のタイマーの秒）である（[ノードを書く](writing-a-node.md)）
- `self.timer` はどの hook の中でも読める（`on_thread_start` と `on_start` を含む）。`__init__` の中で読むと `RuntimeError` になる

## 復元すると時刻が戻る

`mujou.run(..., state=tree)` で復元すると、その実行のタイマーの時刻が、木を取った時刻に戻る。グローバルタイマーなら、`mujou.time.now()` も巻き戻る。詳しくは[状態の保存と復元](state.md)にある。

## テストでは仮想のタイマーを使う

テストで `dt` を決まった値にしたいときは、`mujou.testing.VirtualTimer` を使う。手順は[テスト](testing.md)にある。
