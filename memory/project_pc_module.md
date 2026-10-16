---
name: project-pc-module
description: PC-ALM は Rust ではなく PyTorch で実装し、mujou.pc から公開する（トップレベルには出さない）。α=0 で PC に退化するのでモジュール名は pc
metadata:
  type: project
---

2026-10-15 決定。

- **PC-ALM は PyTorch で実装する。Rust では実装しない。** 計算はノードの内側の行列演算なので、PyTorch の演算と、必要なら `torch.compile` などで足りる（ユーザーの判断）。Rust のコア（`mujou._core`）は学習則を知らない基盤にとどめる
- **公開元は `mujou.pc`。** `mujou.Node` を継承したクラスとして提供する。`mujou.*` のトップレベルには再公開しない（基盤と更新手法の区別を名前に残すため）。別の更新手法も手法ごとのモジュールにする
- モジュール名を `pcalm` でなく `pc` にしたのは、双対のステップサイズ α=0 で PC-ALM が通常の PC に退化し、PC も同じモジュールに収まるから
- 「PC-ALM をコアの責務として提供する」（[[project-core-message-framework]]）のコアは mujou の中核の機能の意味で、`mujou._core` のことではない

**Why:** Rust で書く利点（並行処理の安全性）はノードの間の基盤で効くもので、ノードの内側の行列演算には要らない。

**How to apply:** `mujou.pc` の中のクラス名は実装時に決める（ユーザーに確認）。`torch.compile` が 3.14t で多数のスレッドから使えるかは、使うときに確かめる。テストは `tests/mujou/pc/` に置く。仕様は `specs/update-rules.md` の「実装と公開元」。

関連: [[project-pluggable-update-rule]], [[project-torch-module]], [[project-node-lifecycle]]
