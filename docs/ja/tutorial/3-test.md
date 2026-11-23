# 🧪 3. ノードをテストする

[2. ノードを書く](2-node.md) の `Leaky` を [`mujou.run_manually`][mujou.run_manually] でテストする。この実行は、[`step`][mujou.ManualRun.step] でだけ進む仮想時間の上で走る（[2 つの走らせ方](../topics/simulation.md#two-ways)）。

## テスト {#test}

```sh
uv run pytest examples/test_leaky.py
```

```python
--8<-- "examples/test_leaky.py:test"
```

出力は同じ係数で計算した厳密解と比べる。手動の実行には実時間が入らないので、送った時刻はちょうど `steps` ミリ秒（`t.ns == steps * 1_000_000`）になる。

期待値をノードと同じ演算・同じ順序で計算すれば、`assert_close` の代わりに `torch.equal` でビット単位に比べられる（次の節のテスト）。

## 更新の順序 {#order}

```python
--8<-- "examples/test_leaky.py:chain"
```

`step(h, net.a, net.b)` は渡した順に更新するので `b` は `a` がこのステップで送った値を読み、`step(h)` ではどのノードもステップの開始時の値を読む（[2 種類のステップ](../topics/simulation.md#steps)）。

## 保存と復元 {#restore}

```python
--8<-- "examples/test_leaky.py:restore"
```

途中の状態を取得し、新しいインスタンスで続けた結果は、止めずに続けた結果とビット単位で一致する。

## 例外 {#exceptions}

=== "避ける"

    ```python
    with mujou.run_manually(net) as run:
        with pytest.raises(NetworkError):
            run.step(0.001)  # ここでノードが ValueError を投げる
    # with ブロックの終わりの shutdown が同じ NetworkError を再び送出する
    ```

=== "こう書く"

    ```python
    from mujou import NetworkError

    with pytest.raises(NetworkError) as info, mujou.run_manually(net) as run:
        run.step(0.001)  # ここでノードが ValueError を投げる
    assert isinstance(info.value.__cause__, ValueError)
    ```

[`NetworkError`][mujou.NetworkError] は `step` だけでなく `with` を抜けるときの `shutdown` も送出するため、`pytest.raises` は `run_manually` と同じ `with` 文に並べてブロック全体を囲む。

## 次へ

- 学習するノード: [4. ノードの中で学習する](4-learn-in-a-node.md)
- 決定的にするための条件（乱数、共有するオブジェクト、複数の実行）: [シミュレーション](../topics/simulation.md#determinism)
