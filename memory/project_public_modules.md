---
name: project-public-modules
description: モジュールは公開する。private な mujou._core パッケージと mujou/pc/_*.py をやめ、mujou/ 直下に 1 責務 1 ファイルで並べた。private なのは Rust の拡張モジュール mujou._native だけ。公開 API は __all__ と docs（2026-10-21 のユーザーの決定）
metadata:
  type: project
---

2026-10-21、作り直し（#93）を merge した後、ユーザーが「pc の構造で private が多すぎる。`_native` は private でよいが、core や pc 直下のモジュールは公開してもよいのでは。Python らしくない」と指摘し、推奨案（直下に平らに並べる）で決めた（ブランチ refactor/2026-10-21/public-modules）。

- `src/mujou/_core/*.py` は `src/mujou/*.py` に移した。`_core/run.py` は、関数 `mujou.run` とモジュール `mujou.run` が衝突するので `mujou/execution.py` にした（specs の「実行」に合わせた名前）
- `src/mujou/pc/_layer.py` などは `_` を外した（`pc/layer.py`、`pc/chain.py`、…）
- 拡張モジュールは `mujou._core._native` から `mujou._native` にした（maturin の `module-name`、`#[pymodule(module = "mujou")]`、pyclass の `module`、`import_exception!(mujou.errors, ...)`）
- テストは `tests/mujou/` が `src/mujou/` をそのままミラーする（`tests/mujou/native/` は拡張モジュール）

**Why:** Python では private かどうかを名前の `_` で示し、パッケージごと隠さない（torch・numpy・標準ライブラリと同じ）。利用者が traceback や「定義へ移動」で辿った先が `_core` や `_layer.py` だと、触ってはいけない場所に見える。「`_core` の中の `_` の無い名前は private」という二重の規約も要らなくなる。

**How to apply:** 新しいモジュールに `_` を付けない。公開 API は各パッケージの `__all__` と docs に載る名前で、複数のモジュールが使う内部の名前は `_` を付けずに `__all__` から外す。1 つのモジュールの中だけで使う名前に `_` を付ける。層の順序は `tests/test_layers.py` が正本で、`mujou.pc` は `mujou`・`mujou.time`・`mujou.testing` だけを import する。
