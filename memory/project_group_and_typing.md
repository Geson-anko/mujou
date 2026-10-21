---
name: project-group-and-typing
description: ノードやネットワークを並べるだけの入れ物は Group(**named)（位置引数は持たない、R4）。mujou.typing は廃止し、MessageSpec はトップレベルの mujou.MessageSpec
metadata:
  type: project
---

- **`Group(**named)`:** 名前を付けて並べるだけで、互いに繋がない `Network`。名前を `Compose` にしなかったのは、torchvision の `Compose` のような直列の合成に読めるから（2026-10-17、ユーザーの決定）。位置引数は持たない（R4、2026-10-21 に承認: [[project-restructure-2026-10-20]]）
- **`mujou.typing` は廃止した。** 相手の数の型（`Single` / `Multi`）は `Port` / `MultiPort` のクラスになり（[[project-network-ports]]）、残った `MessageSpec` は `mujou.MessageSpec` にした。サブモジュールは中身が 1 名だけになる

**Why:** 位置引数で並べると、同じノードのパスが組み方によって変わり、保存した木が合わなくなる。

**How to apply:** 並べて走らせるときは `mujou.run(Group(enc=..., dec=...))`。注釈に要る型を新しく公開するときは、トップレベルに置く。
