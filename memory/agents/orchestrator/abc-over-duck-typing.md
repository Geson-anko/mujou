# abc-over-duck-typing（orchestrator のメモ）

PR #103（pc-optimizers）をマージした後、main から分岐して行う。

## ユーザーの決定（2026-10-23）

- ダックタイピングと Protocol を廃止し、ABC に置き換える
- 状態への自動登録: 公開の ABC `mujou.Stateful`（`state_dict` / `load_state_dict` が抽象）。`LearnClock` と束縛した optimizer は継承する。`nn.Module` は `Stateful.register(nn.Module)` で明示的に登録する。利用者の独自オブジェクトは継承が必要になる
- `pc.show` の引数: `prev` / `next` を抽象 property に持つ公開の ABC `pc.Shown`。`Chain` が継承し、手で組んだネットワークも継承すれば渡せる
- 内部の `current.Running`: 抽象を置かず、具体型 `Driver` で型付けする（循環 import は TYPE_CHECKING で避ける）
- `pc.Optimizer` / `pc.BoundOptimizer` は ABC にする

## 段階 1: 計画

ブランチ `refactor/2026-10-23/abc-over-duck-typing`（PR #103 の先頭から。#103 のマージは auto mode に止められたのでユーザーに依頼）。

### ユーザーの決定からの変更（要報告）

- `Running` を具体型 `Driver` にする案は書けない: `current.py`（層 2）は `driver.py`（層 9）より下で、`tests/test_layers.py` は `TYPE_CHECKING` の下の import を禁じる。`Running` は依存を逆にするための型なので、ABC にして `Driver` が継承する
- ポートやドライバの部品はインスタンス属性（`__dict__` が唯一のレジストリ）なので、抽象 property にできない（代入が property と衝突し、抽象のままでインスタンス化も断られる）。`pc.Shown` と `Running` は属性を型注釈で宣言した ABC（抽象メソッドなし、名目上の型）にする

### 公開 IF

- `mujou.Stateful(ABC)`（`node.py` に定義し `mujou.__all__` に足す）: 抽象 `state_dict(self) -> Any`、抽象 `load_state_dict(self, state: Any, /) -> None`。`Stateful.register(nn.Module)`
- `node.is_state(value)`: `isinstance(value, torch.Tensor | Stateful)`。`_Stateful` と `_has_state` を消す。`Node.__load` も `isinstance(value, Stateful)`
- `pc.LearnClock(Stateful)`
- `pc.Optimizer(ABC)`: 抽象 `bind`。`pc.BoundOptimizer(Stateful)`: 抽象 `step_`。`SGD` / `AdamW` は `Optimizer`、`_BoundSGD` / `_BoundAdamW` は `BoundOptimizer` を継承
- `pc.Shown(ABC)`: `prev: Port[Never, Sample]`、`next: Port[Prediction, Never]` の注釈。`Chain(Network, Shown)`、`show(run, chain: Shown, ...)`。`pc.__all__` に足す
- `current.Running(ABC)`: `node`、`wires`、`clock` の注釈。`Driver(Running)`

### 振る舞いの変化

- `state_dict` を持つが `Stateful` でないオブジェクトは、状態に入らなくなる（黙って落ちる）。検出のガードは hasattr のダックタイピングに戻るので置かない。docs と specs に書く
- torch の型で登録するのは `nn.Module` だけ。ほかは利用者が `mujou.Stateful.register(cls)` する

### テスト観点

- `Stateful` を継承した独自オブジェクトは状態に入る。`state_dict` を持つだけのオブジェクトは入らない
- `nn.Module` は `Stateful`（register）
- 抽象メソッドを欠いた `Stateful` / `BoundOptimizer` / `Optimizer` の派生はインスタンス化で `TypeError`
- `LearnClock`、束縛した optimizer は `Stateful`。`Chain` は `pc.Shown`
- `__all__` に `Stateful`（mujou）と `Shown`（pc）
- 既存の test helper と examples（`examples/mnist/watch.py`）のダックタイプのオブジェクトを継承に直す

## 段階 2〜5 の記録

- 段階 2: 新しいテスト 12 件が red（`Stateful` が無い、`Shown` が無い、Protocol に isinstance できない、抽象を欠いた派生が作れてしまう）
- 段階 3: `just run` green（914 passed）。計画外: `Running` の属性を抽象 property にしたため、`Driver` は `_node` / `_clock` / `_wires` を持ち property で返す。hot path（`update`）は property を通さず `_node` を読む。`examples/pc_merge.py` の `TwoTowers` が `pc.show` に渡されていたので `pc.Shown` を継承させた
- 段階 4（自己レビュー）: `Node` の docstring の折り返し崩れ、`Running` の docstring の言い回し（`execution` は `driver` より上の層なので例から外した）を直した。整理点はほかに無し
- 段階 5: docs（concepts の表、tutorial 4・6、topics/update-rules、topics/pc、topics/pc-concepts の en/ja）、API リファレンス（`mujou.Stateful`、`pc.Shown`）、specs（nodes、pc、scenarios の K24）、memory（`feedback_no_duck_typing.md` を新設、project_pc_optimizers・project_node_lifecycle）。docs の変更は 1〜2 文なので teacher-student と yomiyasu は回していない

## 対象（2026-10-23 の洗い出し）

- `src/mujou/node.py`: `_Stateful`（Protocol）、`_has_state`（hasattr）、`is_state`、`_state_of`、`Node.__load`
- `src/mujou/current.py`: `Running`（Protocol）
- `src/mujou/pc/chain.py`: `_Shown`（Protocol）
- `src/mujou/pc/optim.py`: `Optimizer` / `BoundOptimizer`（Protocol）
