---
name: feedback-super-calls
description: Node/Network のサブクラスは __init__ と override した hook の最初で super() を呼ぶ。サンプル・ドキュメント・docstring・specs・テストの例も常にそう書く
metadata:
  type: feedback
---

`mujou.Node` / `mujou.Network` のサブクラスは、`__init__` の最初で `super().__init__()` を、override した hook（`on_start`・`on_update(dt)`・`on_stop` など）の最初で `super()` の同じ hook を呼ぶ。`examples/`・`docs/`・docstring の例・`specs/` のコード例・テストのノードも常にそう書く。

**Why:** ユーザーの指示（2026-10-17）。「override するメソッドは基本 super() を呼んだ方が良い。慣習的に」。`super().__init__()` は必須で、呼ばないノードは run の段 1 で `TypeError`（K2）。基底の hook は何もしないので hook の `super()` を呼ばなくても壊れないが、自作の基底クラスを継承したときの保険になる。

**How to apply:** 新しいサンプルやドキュメントのノードを書くときは super() 呼び出しを入れる。「super() は省略できる」とは書かない。関連: [[project-node-lifecycle]]
