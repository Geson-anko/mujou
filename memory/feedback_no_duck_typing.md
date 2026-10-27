---
name: feedback-no-duck-typing
description: mujou はダックタイピングと Protocol を使わず、ABC の継承で名目上の型にする（2026-10-23 のユーザーの決定）。状態は mujou.Stateful、optimizer は pc.optim.Optimizer / pc.optim.BoundOptimizer、pc.show の引数は pc.Shown、Driver は current.Running
metadata:
  type: feedback
---

2026-10-23、pc の optimizer を Protocol にした理由を尋ねたうえで、ユーザーが「ダックタイピングは廃止、ABC へすべて置換」と決めた。

- 状態への自動登録は `isinstance(value, torch.Tensor | mujou.Stateful)`。`hasattr(state_dict)` では判定しない。継承できない他のライブラリの型は `Stateful.register(cls)`（mujou は `nn.Module` を登録）
- `pc.optim.Optimizer`（抽象 `bind`）、`pc.optim.BoundOptimizer`（`Stateful` の派生、抽象 `step_`）
- インスタンス属性で持つもの（ポート、Driver の部品）は抽象 property にできない（`__dict__` が唯一のレジストリで、代入が property と衝突する）。`pc.Shown` は属性を型注釈で宣言しただけの ABC。`current.Running` は抽象の読み取り専用 property にし、`Driver` は private 属性の上に property を置く
- 依存を逆向きにする型（下の層が上の層の実装を見る）も ABC にする。`tests/test_layers.py` は `TYPE_CHECKING` の import を禁じるので、具体型で型付けする逃げ道は無い

**Why:** Protocol では、メソッドを書き忘れたクラスが最初に呼ばれる時（実行中の hook の例外で全体が止まる）まで気付かれない。ABC なら作った時点で `TypeError` になる。名前が同じだけのオブジェクトが黙って状態に入ることも無くなる。

**How to apply:** 新しい差し替え点や「〜を持つもの」を受け取る口は、ABC を公開して継承させる。`Protocol`、`runtime_checkable`、`hasattr` での判定は src に書かない。代わりに `state_dict` を持つだけで `Stateful` でないオブジェクトは状態から黙って落ちる（ガードは置かない）ので、docs に書く。

関連: [[project-pc-optimizers]], [[feedback-good-taste]]
