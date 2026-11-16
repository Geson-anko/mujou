# 🧪 3. ノードをテストする

`mujou.run` で始めた実行では `dt` が実時間の経過で決まり、ノードの更新順序も毎回違うので、結果は毎回わずかに変わる。このページでは [2. ノードを書く](2-node.md)のノードを、毎回同じ結果になるように走らせる `mujou.run_manually` でテストする。

## 道具

| 名前                            | 役割                                                                                                                                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mujou.run_manually(root, ...)` | `mujou.run` の `timer`・`inputs`・`state`・`workers` を受け取って木を走らせ、`ManualRun` を返す。ノードは `step` が更新するまで更新しない。既定では新しい `mujou.time.VirtualTimer` の上で走る |
| `ManualRun.step(dt)`            | タイマーを `dt` 秒進めてから、全ノードを同時に 1 回ずつ更新する                                                                                                                                |
| `ManualRun.step(dt, *nodes)`    | タイマーを `dt` 秒進めてから、渡したノードだけを渡した順に 1 回ずつ更新する                                                                                                                    |
| `ManualRun.timer`               | 実行の `VirtualTimer`                                                                                                                                                                          |

`ManualRun` は `Run` なので、`write`・`read`・`state_dict` なども `mujou.run` と同じように使える。

## テスト

```python
--8<-- "examples/test_leaky.py:test"
```

```sh
uv run pytest examples/test_leaky.py
```

1. `mujou.run_manually(node, inputs=...)` はノードを時刻 0 の新しい `VirtualTimer` の上で走らせる。`inputs` は開始時からの入力の Port の値
2. 各 `run.step(h)` は時間を `h` 秒進めてから全ノードを 1 回ずつ更新する。ノードの `dt` はちょうど `h` になる
3. 参照は同じ係数で計算した厳密解。`torch.testing.assert_close` で比べる。参照が同じ演算を同じ順に行うなら `torch.equal` で比べられる（次の節）
4. ノードが出力を送った時刻はちょうど `steps` ミリ秒。手動の実行には実時間が入らない

## 更新の順序

```python
--8<-- "examples/test_leaky.py:chain"
```

- ノードを渡さない `step(dt)` は実行のワーカーで全ノードを同時に更新する。どのノードもステップ開始時の値を読み、ステップ中に送った値がほかのノードに届くのは次のステップ
- `step(dt, *nodes)` は渡したノードを 1 つずつ更新し、後のノードは前のノードがこのステップで送った値を読む。`step(h, net.a, net.b)` では `b` は `a` がこのステップで送った値を読み、`step(h, net.b, net.a)` なら前のステップの値を読む
- 全ノードを `step(dt, *nodes)` に渡しても `step(dt)` と同じにはならない。メッセージが出力の方向にだけ流れる chain に限り、出力側から入力側への順が `step(dt)` と一致する（`test_a_step_without_nodes_equals_the_order_from_the_output`）。メッセージが逆向きにも流れるなら、一般にはどの順とも一致しない
- ノードの `dt` は前回の更新からの時間で、最初は実行の開始から。同じ `step` に 2 回渡したノードは 1 回だけ更新する。2 回目は時刻が進んでいないので飛ばす
- ノードの周期は手動の実行では使われない

## 保存と復元

途中で状態を取得し、新しいインスタンスで続けた結果は、止めずに続けた結果とビット単位で一致する。

```python
--8<-- "examples/test_leaky.py:restore"
```

状態から復元した実行は、その状態の時刻から始まる自分の `VirtualTimer` で走る。

## 例外

ノードが例外を投げると、`step` はその場で `NetworkError` を送出し、その `__cause__` がノードの例外になる。以後は `with` ブロックの終わりの `shutdown` を含め、実行のどの呼び出しも同じ例外を送出する。そのため `pytest.raises` はブロック全体を囲むように同じ `with` 文に並べる。

```python
from mujou import NetworkError

with pytest.raises(NetworkError) as info, mujou.run_manually(net) as run:
    run.step(0.001)  # ここでノードが ValueError を投げる
assert isinstance(info.value.__cause__, ValueError)
```

## 決定的にするためのほかの条件

- 乱数を使うノードには `seed` を渡し、乱数は `self.generator` から引く（[スレッドとデバイス](../topics/threads-and-devices.md#seed-%E3%81%A8%E4%B9%B1%E6%95%B0)）
- 変更する Python のオブジェクトをノード同士で共有しない。`step(dt)` はノードを同時に更新するので、どちらが先に動いたかで結果が変わる。共有が避けられないなら、`step(dt, *nodes)` で順を固定して更新する
- 時間は `step` でだけ進める。`step` を呼ばなければ何も送られないので、`step` を呼ぶスレッドで `timeout` なしに `wait_for` や `subscribe` で待つと、いつまでも終わらない
- 複数の実行を 1 本の時間軸に載せるには、`mujou.time.VirtualTimer()` を作って同じものを各実行に渡す。1 つの実行の `step` でほかも進む

## 次へ

- 学習するノード: [4. ノードの中で学習する](4-learn-in-a-node.md)
- 手動の実行の仕組みとシミュレーションの方法: [シミュレーション](../topics/simulation.md)
