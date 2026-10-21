---
name: project-group-and-typing
description: ノードやネットワークを並べるだけの入れ物は Group(**named)（位置引数は持たない、R4）。mujou.typing は廃止し、MessageSpec はトップレベルの mujou.MessageSpec。汎用の注釈の型は公開モジュール mujou.types（2026-10-21）
metadata:
  type: project
---

- **`Group(**named)`:** 名前を付けて並べるだけで、互いに繋がない `Network`。名前を `Compose` にしなかったのは、torchvision の `Compose` のような直列の合成に読めるから（2026-10-17、ユーザーの決定）。位置引数は持たない（R4、2026-10-21 に承認: [[project-restructure-2026-10-20]]）
- **`mujou.typing` は廃止した。** 相手の数の型（`Single` / `Multi`）は `Port` / `MultiPort` のクラスになり（[[project-network-ports]]）、残った `MessageSpec` は `mujou.MessageSpec` にした。サブモジュールは中身が 1 名だけになる

**Why:** 位置引数で並べると、同じノードのパスが組み方によって変わり、保存した木が合わなくなる。

- **汎用の型は公開モジュール `mujou.types` に置く（2026-10-21、ユーザーの決定）。** `Size`（`int | tuple[int, ...]`）のように、pc に限らずプロジェクトで使える注釈の型を集め、利用者は `from mujou.types import Size` で使う。トップレベルの `mujou` には再公開しない。名前を `typing` にしないのは標準ライブラリと紛れるから。集めた型は `Size`・`Device`（`torch.device | str`）・`Message`（`torch.Tensor | Mapping[str, object]`、ホストが書くメッセージ）。`Node | Network` や `AnyPort` のようにクラスに結びつく型は、層の下の `mujou.types` から import できないので元のモジュールに残す。`StateDict`（`Mapping[str, Any]`）は場所ごとに意味が違うので集めない（いずれも 2026-10-21、ユーザーの選択）

**How to apply:** 並べて走らせるときは `mujou.run(Group(enc=..., dec=...))`。注釈に要る汎用の型は `mujou.types` に置く。`MessageSpec` のようにクラスとして中身を持つものはトップレベルに置く。
