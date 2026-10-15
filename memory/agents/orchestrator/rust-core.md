# Rust コアのセットアップ（PR 2）

2026-10-15、solo-dev-cycle。

## 計画

- Cargo workspace をルートに置く。crate は 2 つ
  - `crates/mujou-core`: 純 Rust のコア。PyO3 に依存しない。`cargo test` が libpython なしで回る
  - `crates/mujou-python`: PyO3 の bindings。cdylib `_core`、`#[pymodule(gil_used = false)]`。`test = false`（libpython のリンクを避ける）
- ビルドバックエンドを hatchling から maturin（>=1.15,<2）に替える。`python-source = "src"`、`module-name = "mujou._core"`
- 版数の唯一の源は Cargo workspace の version。`mujou.__version__` は `_core.__version__`（`CARGO_PKG_VERSION`）を再 export する
- `[tool.uv] cache-keys` に Rust のソースを入れ、`uv run` が変更を検知して再ビルドするようにする
- 型スタブ `src/mujou/_core.pyi` は手書き（公開する名前が少ないうち。pyo3-stub-gen は名前が増えたら検討）
- just: `test` に `cargo test --workspace` を足す。pre-commit に local hook の `cargo fmt` と `cargo clippy -D warnings`
- `.gitignore` に editable install が置く `src/mujou/*.so`（`target/` は PR 1 で追加済み）

## テスト観点

- free-threaded ビルドで `mujou._core` を import しても GIL が有効に戻らない（`gil_used = false` の検証）。サブプロセスで確かめる

## 制約

- gh の token に `workflow` scope が無いので `.github/workflows/` は変えない。既存 CI の `uv sync` が maturin でビルドする（GitHub のランナーには Rust がある）。test.yml の paths に `**/*.rs` などを足す提案は PR 本文に書く

## レビュー（code-reviewer、approve）と対応

- should-fix `pyo3/extension-module` feature は PyO3 0.30 で非推奨。maturin >= 1.9.4 が `PYO3_BUILD_EXTENSION_MODULE` を設定する → 削除
- should-fix CI の paths に Rust のファイルが無い → workflow scope が無いので PR 本文に提案（test.yml と type-check.yml）
- nit `PYTHON_GIL=0` があるとテストが素通りする → subprocess の env から除いた
- nit CLAUDE.md の pre-commit の列挙 → cargo fmt / clippy を追加
- 見送り: プレリリース版での版数のずれ（今は起きない）、`just clean` の拡張（範囲外）、`gil_used = false` は既定値だが意図の明示として残す
