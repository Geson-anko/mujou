# 🧪 3. Test a Node

This page tests `Leaky` from [2. Write a Node](2-node.md) with [`mujou.run_manually`][mujou.run_manually]. Such a run goes on a virtual time that moves only with [`step`][mujou.ManualRun.step] ([two ways to run](../topics/simulation.md#two-ways)).

## The test {#test}

```sh
uv run pytest examples/test_leaky.py
```

```python
--8<-- "examples/test_leaky.py:test"
```

The output is compared with the exact solution computed with the same coefficient. No real time enters a manual run, so the time of the output is exactly `steps` milliseconds (`t.ns == steps * 1_000_000`).

When the reference does the same operations as the node in the same order, compare bit for bit with `torch.equal` instead of `assert_close` (the test in the next section).

## The order of the updates {#order}

```python
--8<-- "examples/test_leaky.py:chain"
```

`step(h, net.a, net.b)` updates the nodes in the given order, so `b` reads what `a` sent in this step; with `step(h)` every node reads the values from the start of the step ([two kinds of step](../topics/simulation.md#steps)).

## Save and restore {#restore}

```python
--8<-- "examples/test_leaky.py:restore"
```

A run that takes the state midway and goes on in a new instance matches a run that went on without stopping, to the last bit.

## Exceptions {#exceptions}

=== "Avoid"

    ```python
    with mujou.run_manually(net) as run:
        with pytest.raises(NetworkError):
            run.step(0.001)  # a node raises ValueError here
    # the shutdown at the end of the with block raises the same NetworkError again
    ```

=== "Write this"

    ```python
    from mujou import NetworkError

    with pytest.raises(NetworkError) as info, mujou.run_manually(net) as run:
        run.step(0.001)  # a node raises ValueError here
    assert isinstance(info.value.__cause__, ValueError)
    ```

Not only `step` but also the `shutdown` on leaving the `with` block raises [`NetworkError`][mujou.NetworkError], so put `pytest.raises` in the same `with` statement as `run_manually`, around the whole block.

## Next

- A node that learns: [4. Learn in a Node](4-learn-in-a-node.md)
- The conditions for determinism (random numbers, shared objects, several runs): [Simulation](../topics/simulation.md#determinism)
