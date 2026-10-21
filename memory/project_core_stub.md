---
name: project-core-stub
description: mujou._core._native の型スタブ src/mujou/_core/_native.pyi は just stub で生成してコミットする生成物。説明は Rust の doc comment に書く。手で編集しない
metadata:
  type: project
---

2026-10-16 決定（ユーザーの指示で、手書きのスタブを自動生成にした）。

- `src/mujou/_core/_native.pyi` は `just stub` で生成し、生成物をコミットする。手で編集しない
- docstring は Rust の doc comment から出る。前提条件・例外・スレッドの約束は doc comment に書く。`mujou.time` は native の型をそのまま再エクスポートするので、`Timer` などの docstring の正本も Rust の doc comment
- 生成は pyo3 の `experimental-inspect` と `crates/mujou-stubgen`（`pyo3-introspection`）。pyo3 の足りない所は生成器が補う（`__new__` の `Self`、注釈の無い `*args`、`@override` など）。例外は Python で定義するので、生成器は補わない
- 古さはローカルでは pre-commit のフックが、CI では Test のワークフローの `just stub && git diff --exit-code` が検出する

**Why:** 手書きのスタブは bindings とずれる。生成なら doc comment が唯一の正になる。

**How to apply:** bindings を変えたら `just stub` を回してスタブも一緒にコミットする。仕組みの詳細と制約は `specs/native-core.md` の「型スタブ」。

関連: [[project-native-core-rust]]
