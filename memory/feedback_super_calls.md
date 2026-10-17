---
name: feedback-super-calls
description: Node/Network のサブクラスは __init__ と override した hook の最初で super() を呼ぶ。サンプル・ドキュメント・docstring の例も常にそう書く
metadata:
  type: feedback
---

`mujou.Node` / `mujou.Network` のサブクラスは、`__init__` の最初で `super().__init__()` を、override した hook（`on_start`・`on_update(dt)`・`on_stop` など）の最初で `super()` の同じ hook を呼ぶ。`examples/`・`docs/`・docstring の例・`specs/` のコード例も常にそう書く。

**Why:** ユーザーの指示（2026-10-17）。「override するメソッドは基本 super() を呼んだ方が良い。慣習的に」。`super().__init__()` を省略しても動くように `mujou.Node` はクラス属性に既定値を置いているが、それを利用者向けの書き方として勧めない。

**How to apply:** 新しいサンプルやドキュメントのノードを書くときは super() 呼び出しを入れる。「`super().__init__()` は省略できる」とは書かない。テスト（`tests/`）の中のノードは省略してもよい（省略しても動くことの確認を兼ねる）。関連: [[project-node-lifecycle]]
