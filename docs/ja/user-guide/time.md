# 時間の制御

このページでは、時刻を読む方法、時間の上で待つ方法、時間の速さの変え方、実行ごとのタイマーの使い方を書く。`dt` に比例して更新する考え方と、一時停止しても時間が飛ばない理由は[グローバル時間](../concepts/time.md)にある。

## どこで何を使うか

| したいこと         | ホスト（既定のグローバルタイマー）                                                  | ホスト（自分の `Timer` で走らせた実行） | ノードの hook の中            |
| ------------------ | ----------------------------------------------------------------------------------- | --------------------------------------- | ----------------------------- |
| 時刻を読む         | `mujou.time.now()`（`Timestamp`）、`mujou.time.time()`（秒）、`time_ns()`（ナノ秒） | `timer.now()`                           | `self.timer.now()`            |
| 時間の上で待つ     | `mujou.time.sleep(秒)`                                                              | `timer.sleep(秒)`                       | できない                      |
| 時間の速さを読む   | `mujou.time.get_time_scale()`                                                       | `timer.get_time_scale()`                | `self.timer.get_time_scale()` |
| 時間の速さを変える | `mujou.time.set_time_scale(倍率)`                                                   | `timer.set_time_scale(倍率)`            | できない                      |

- ホストは、Python 標準の `time.sleep` ではなく、その実行のタイマーの `sleep` で待つ（入力を一定の時間見せるときなど）。Python 標準の `time.sleep` は実時間で待つので、時間の速さの変更と一時停止（状態を取る間、実行は止まる）に合わない
- 倍率は有限で 0 より大きい数である。それ以外は `ValueError` になる

## Timestamp

`run.read` が返す時刻、`run.write` が返す書き込みの時刻、`mujou.time.now()`・`timer.now()`・`self.timer.now()` の値は、どれも `mujou.time.Timestamp` である。

```python
--8<-- "examples/timers.py:33:36"
```

- `float(ts)` は秒、`int(ts)` は秒の整数部、`ts.ns` はナノ秒の正確な値である
- 2 つの差 `later - earlier` も `Timestamp` である。`earlier` の方が遅いと `ValueError` になる
- 自分で作るときは `Timestamp(ns=...)` とナノ秒をキーワードで渡す
- `Timestamp` は、どのタイマーの時刻かを持たない。比べてよいのは同じタイマーの時刻同士だけである。`mujou.time.now()` と比べてよいのは、グローバルタイマーで走らせた実行の時刻だけである

## 実行ごとのタイマー

`mujou.time.Timer()` を作って `mujou.run(..., timer=timer)` に渡すと、その実行は自分のタイマーの上で走る。新しいタイマーは時刻 0、倍率 1 で始まる。

```python
--8<-- "examples/timers.py:38:60"
```

自分のタイマーを渡すのは、次のときである。

- その実行だけ時間を速めたり遅くしたりしたい（デバッグや、実時間より速い試行）
- その実行の一時停止（状態の取得）で、グローバルタイマーで走るほかの実行を止めたくない
- 状態から復元したいが、同じグローバルタイマーでほかの実行が走っている（[状態の保存と復元](state.md)）

このとき、ホストはその実行を `timer.sleep` で待ち、`run.read` の時刻は `timer.now()` と比べる。`mujou.time` の関数（`mujou.time.sleep`、`mujou.time.time()` など）はグローバルタイマーで働き、この実行のタイマーには効かない。そのため、この実行には使わない。

## グローバルタイマーの速さを変える

`mujou.time.set_time_scale` は、グローバルタイマーで走るすべての実行の速さを変える。

```python
--8<-- "examples/timers.py:62:67"
```

グローバル時刻は、プロセスが初めて使ったときに 0 から始まる。

## ノードは時間を読むだけ

ノードの `self.timer` は、実行のタイマーの読み取り専用の側（`mujou.time.ReadOnlyTimer`）で、`now()` と `get_time_scale()` だけを持つ。

hook の中から `mujou.time.sleep`・`mujou.time.set_time_scale`・`Timer.sleep`・`Timer.set_time_scale`・`VirtualTimer.advance` を呼ぶと、`RuntimeError` になる。1 つのノードが時間を止めたり速さを変えたりすると、同じ実行のほかのノードの `dt` まで変わるからである。

- ノードを一定の間隔で更新したいときは、hook の中で待たずに、`super().__init__(period=秒)` で周期を渡す。周期は、2 回の更新の間の平均の時間（実行のタイマーの秒）である（[ノードを書く](writing-a-node.md)）
- `self.timer` は、実行がノードを起動してから読める。`__init__` の中で読むと `RuntimeError` になる

## 復元すると時刻が戻る

`mujou.run(..., state=木)` で復元すると、その実行のタイマーの時刻が、木を取った時刻に戻る。グローバルタイマーなら、`mujou.time.time()` も巻き戻る。詳しくは[状態の保存と復元](state.md)にある。

## テストでは仮想のタイマーを使う

テストで `dt` を決まった値にしたいときは、`mujou.testing.VirtualTimer` を使う。手順は[テスト](testing.md)にある。
