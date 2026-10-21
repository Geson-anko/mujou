# ⏱️ 時間

実行ごとに時間軸は 1 本で、それが実行のタイマーである。各ノードの `dt`、各メッセージの時刻、実行の一時停止はすべてその上にある。このページではタイマー、時間の単位、時間の速さ、一時停止、再開した実行のタイマーを説明する。

## タイマー

| 名前                        | 何か                                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `mujou.time.global_timer`   | 自分のタイマーを渡さないすべての実行のタイマー。プロセスで初めて使ったときに 0 から始まる             |
| `mujou.time.Timer()`        | 1 つの実行専用のタイマー（`mujou.run(net, timer=Timer())`）。`start`（0）から `time_scale`（1）で進む |
| `mujou.time.VirtualTimer()` | 進めたとき（`advance(seconds)`）だけ進む `Timer`。`mujou.testing.run` が使う                          |
| `mujou.time.ReadOnlyTimer`  | タイマーを読むだけのビュー。ノードの `self.timer` がこれで、`now()` と `time_scale()` だけを持つ      |
| `mujou.time.Timestamp`      | タイマーの上の時刻                                                                                    |

ホストは `run.timer` で実行の時間を操作する。`now()`・`sleep(seconds)`・`pause()`・`resume()`・`is_paused()`・`time_scale()`・`set_time_scale(scale)` がある。ノードは hook の中で `self.timer.now()` で読むだけ。

## 単位

| 単位               | 使うもの                                                                                        |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| 実行のタイマーの秒 | `dt`、`period`、`Timer.sleep`、`VirtualTimer.advance`、`pc.show(seconds=)`、`mujou.pc` の時定数 |
| 実時間の秒         | すべての `timeout`                                                                              |

timeout が実時間なのは、時間の速さや一時停止によらず、止まった実行を見切れるようにするため。

## Timestamp

```python
--8<-- "examples/timers.py:timestamp"
```

`write`・`read_with_time`・`subscribe`・`wait_for`・`recv_with_time`・`now()` は `Timestamp` を返す（まだ送られていない値なら `None`）。`<` などで前後を比べ、引き算でその間の秒数を得る。`Timestamp` はどのタイマーの時刻かを持たないので、1 つの実行の時刻とその `run.timer.now()` のように、同じタイマーの時刻同士だけを比べる。

```python
t_in = run.write({net.prev: x})
run.timer.sleep(0.1)
y, t_out = run.read_with_time(net.next)
if t_out is not None and t_out >= t_in:  # y は x を書いた後に送られた
    print(t_out - t_in)  # 書いてから y が送られるまでの秒数
```

## 実行専用のタイマー

```python
--8<-- "examples/timers.py:own-timer"
```

その実行だけ時間を速めたり遅くしたりしたいとき（デバッグや実時間より速い試行）や、ほかの実行と一緒に止まらないようにしたいときに、実行専用のタイマーを渡す。時間の速さは実時間 1 秒あたりにタイマーが進む秒数で、10 なら実時間の 1 秒がタイマーの 10 秒。有限で 0 より大きい数に限る。

グローバルタイマーも速さを変えられ、それで走るすべての実行が変わる。

```python
--8<-- "examples/timers.py:global-scale"
```

## 時間を止める

`timer.pause()` は `timer.resume()` までそのタイマーの時間を止める。`global_timer.pause()` はグローバルタイマーで走るすべての実行を止める。止まっている間は次のようになる。

- そのタイマーで走る全実行のノードは更新せず、`resume` 後の最初の `dt` に停止中の時間は入らない
- そのタイマーの `sleep` は戻らず、`VirtualTimer.advance` は `RuntimeError` になる
- `run.state_dict()` は取得でき、止めた時刻の状態が返る

状態の取得も、利用者の `pause` とは別に実行のタイマーを止める。`pause` は回数を数えないスイッチで、`resume` を呼んでも取得中の停止は解けない。`is_paused()` はどちらかで止まっていれば真になる。

1 つのタイマーで走る複数の実行の状態を同じ時刻で取得するときに使う。1 つずつ取得すると、後の状態ほど時刻が遅れる。

```python
--8<-- "examples/timers.py:pause"
```

!!! tip "`resume` は `finally` に書く"

    例外で時間が止まったままにならない。

一時停止は状態の取得と観測のためのもの。学習の手順の一部にしない。

## 再開した実行のタイマー

`mujou.run(net, state=tree)` で再開した実行は、木の時刻と時間の速さから始まる新しい `Timer` で走る。グローバルタイマーやほかの実行の時刻は戻らない。

```python
--8<-- "examples/timers.py:restore"
```

`state=` と一緒に `timer=` を渡すと `ValueError` になる。`global_timer.pause()` では再開した実行は止まらないので、その実行の `run.timer` を止める。そのため、同じ時刻で取得した木から再開した実行同士は、もう時間軸を共有しない。それぞれのタイマーは少しずつずれ、1 つを止めてもほかは止まらない。再開した後も 1 本の時間軸に載せたい実行は 1 つの実行（`mujou.run(Group(a=a, b=b))`）として走らせる。その 1 つの木は 1 つのタイマーに復元される。

## ノードは時間を読むだけ

ノードの `self.timer` は実行の `ReadOnlyTimer` で、`on_start` を含むどの hook の中でも読めるが、それ以外では読めない。`__init__` の中やホストから読むと `RuntimeError` になる。実行が終わった時刻で止まるので、`on_exception` と `on_stop` からは時間が止まって見える。

時間を動かせるのはホストだけ（[ノードが守る規則](../concepts.md#%E3%83%8E%E3%83%BC%E3%83%89%E3%81%8C%E5%AE%88%E3%82%8B%E8%A6%8F%E5%89%87)）。一定間隔で更新したいなら、ノードに `period` を渡す。
