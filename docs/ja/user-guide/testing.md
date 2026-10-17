# テスト

ふつうに走らせたネットワークでは、`dt` は実時間の経過で決まり、ノードが更新する順序も毎回違う。そのため結果は毎回わずかに変わる。このページでは、`mujou.testing` を使って、ノードとネットワークを毎回同じ結果になるように走らせ、テストする手順を書く。mujou 自身のテストも同じ API を使う。

## 道具

`mujou.testing` は `import mujou.testing` で読み込む。`import mujou` だけでは読み込まれない。

| 名前                           | 役割                                                                                                                                               |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mujou.testing.run(net, ...)`  | `mujou.run` と同じ引数で走らせ、`ManualRun` を返す。ノードは `step` で許すまで更新しない。`timer` を渡さなければ、新しい `VirtualTimer` の上で走る |
| `mujou.testing.VirtualTimer()` | 仮想のタイマー。時刻 0 から始まり、`advance(seconds)` で進めた分だけ進む                                                                           |
| `ManualRun.timer`              | 実行のタイマー。`mujou.testing.run` が作った `VirtualTimer` か、渡した `timer`                                                                     |
| `ManualRun.step(*nodes)`       | 渡したノードを、渡した順に 1 回ずつ更新し、全部終わってから戻る                                                                                    |

`ManualRun` は `Run` の派生なので、`write`・`read`・`state_dict` なども `mujou.run` と同じように使える。

## 手順

次のテストでは、リーク積分器を 2 つ並べたネットワークを、ノードと同じ厳密な係数 `1 - exp(-h / tau)` で書いた離散アルゴリズムと比べる。

```python
--8<-- "examples/testing_a_node.py:test"
```

1. 比べる相手（参照）を、ノードと同じ演算で書く。ここでは `a` を更新してから、その新しい値で `b` を更新する
2. `mujou.testing.run(net, inputs=...)` で走らせる。時刻 0 の新しい `VirtualTimer` の上で走り、`run.timer` がそれを返す
3. 各ステップで、まず `run.timer.advance(h)` で時間を `h` 秒進め、次に `run.step(...)` でノードを更新する。各ノードの `dt` はちょうど `h` になる
4. `run.read` で出力を読み、参照と比べる

参照とノードが同じ演算を同じ順に行えば、結果はビット単位で一致するので、`torch.equal` で比べられる。演算の順序が違うときは `torch.testing.assert_close` で比べる。

## step の決まり

- `step()` を引数なしで呼ぶと、ネットワークの木の順に全ノードを 1 回ずつ更新する
- 順序で結果が変わる。上の例の `step(net.a, net.b)` では、`b` は `a` がこのステップで送った値を読む。`step(net.b, net.a)` なら、`b` は `a` の前のステップの値を読む
- 前回の更新からタイマーの時刻が進んでいないノードは、飛ばされる（`dt = 0` になるため）。`advance` を忘れると何も起きず、同じ `step` の中で同じノードを 2 回渡しても 2 回目は飛ばされる
- ノードの周期（`period`）は、手動の実行では使われない
- `step` は時刻を進めない。時刻を進めるのは `advance` だけである
- `step` を呼ばなければ、何も送られない。そのため、`step` を呼ぶスレッドで `timeout` なしに `wait_for` や `subscribe` で待つと、いつまでも終わらない

## タイマーを選ぶ

`mujou.testing.run` は、既定では実行ごとに新しい `VirtualTimer` を作る。既定がグローバルタイマーの `mujou.run` とは違う。次の場合は `timer` を渡す。

- 複数の実行を 1 本の時間軸で走らせるとき。たとえば、状態を取った実行のタイマーの上で復元するとき（下の「保存と復元のテスト」）。`VirtualTimer()` を作り、同じものを各実行に渡す
- 仮想でない `Timer` を渡すとき。時計はひとりでに進み、各 `dt` は `step` の間に経った実時間になる。そのテストは決定的にならない

## 決定的にするためのほかの条件

- 乱数を使うノードには `seed` を渡す（[ノードの中の PyTorch](torch.md)）。`seed` を渡さないと、生成器は毎回違う seed で作られる
- テストの中でも、ホストは `time.sleep` で待って結果を決めない。時間は `advance` で進める

## 保存と復元のテスト

途中で状態を取り、新しいインスタンスに復元して続けた結果が、止めずに続けた結果と一致するかを確かめられる。`VirtualTimer` の上なら、両者はビット単位で一致する。次のテストは、上のプログラムの続きである。後半の復元では前半と同じ `timer` を使う。前半の実行はもう終わっているので、同じタイマーで復元してよい。

```python
--8<-- "examples/testing_a_node.py:restore"
```

## 例外のテスト

ノードが例外を投げると、その例外は `step` から `NetworkError` として届く。`__cause__` がノードの例外である（[例外と終了](errors.md)）。その後は、`with` ブロックの終わりの `shutdown` を含めて、実行のどの呼び出しも同じ例外を送出する。そのため、`pytest.raises` はブロック全体を囲むように、同じ `with` 文に並べる。

```python
import pytest

import mujou.testing
from mujou import NetworkError

with pytest.raises(NetworkError) as info, mujou.testing.run(net) as run:
    run.timer.advance(0.001)
    run.step()  # ここでノードが ValueError を投げる
assert isinstance(info.value.__cause__, ValueError)
```
