---
name: project-group-and-typing
description: 2026-10-17 のユーザーの決定。Compose を Group に改名し、注釈に使う型は専用の名前空間 mujou.typing に置く（Single / Multi / MessageSpec）
metadata:
  type: project
---

2026-10-17 のユーザーの決定。

- **`Compose` → `Group`:** ノードやネットワークを並べるだけでつながないのに、`Compose` は torchvision の `Compose` のような直列の合成に読めるため。振る舞いとパスの付け方は変えない。内部の `RunCompose` も `RunGroup`、サンプルは `examples/group.py`、User Guide のページは `user-guide/group.md`（"Grouping Networks"）
- **型は `mujou.typing`:** 注釈に使う名前は値を作る名前と分けて `mujou.typing` に置く。今は `Single` / `Multi`（Port の相手の数、`Port[S, R, M, CS, CR]` の `M`。CS / CR は 2026-10-20 に carry で足した）と `MessageSpec[T]`。`mujou.time` と同じく `import mujou` で読み込み、`__all__` には入れない（標準の `typing` を隠さないため）

**How to apply:** 利用者が注釈に要る型を新しく公開するときは `mujou.typing` に足す。仕様は `specs/network.md` の「複数のネットワーク」と公開名の一覧。
