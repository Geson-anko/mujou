---
name: project-pc-module
description: PC-ALM は Rust ではなく PyTorch で実装し、mujou.pc から公開する（トップレベルには出さない）。α=0 で PC に退化するのでモジュール名は pc
metadata:
  type: project
---

2026-10-15 決定。

- **PC-ALM は PyTorch で実装する。Rust では実装しない。** 計算はノードの内側の行列演算なので、PyTorch の演算と、必要なら `torch.compile` などで足りる（ユーザーの判断）。Rust のコア（`mujou._core._native`）は学習則を知らない基盤にとどめる
- **公開元は `mujou.pc`。** `mujou.Node` を継承したクラスとして提供する。`mujou.*` のトップレベルには再公開しない（基盤と更新手法の区別を名前に残すため）。別の更新手法も手法ごとのモジュールにする
- モジュール名を `pcalm` でなく `pc` にしたのは、双対のステップサイズ α=0 で PC-ALM が通常の PC に退化し、PC も同じモジュールに収まるから
- 「PC-ALM をコアの責務として提供する」（[[project-core-message-framework]]）のコアは mujou の中核の機能の意味で、`mujou._core` パッケージ（とその中の Rust の拡張モジュール `_native`）のことではない

**Why:** Rust で書く利点（並行処理の安全性）はノードの間の基盤で効くもので、ノードの内側の行列演算には要らない。

**How to apply:** `mujou.pc` の中のクラス名は実装時に決める（ユーザーに確認）。`torch.compile` が 3.14t で多数のスレッドから使えるかは、使うときに確かめる。テストは `tests/mujou/pc/` に置く。仕様は `specs/update-rules.md` の「実装と公開元」。

関連: [[project-pluggable-update-rule]], [[project-torch-module]], [[project-node-lifecycle]]

## 実装計画の決定（2026-10-19〜20、`specs/pc-module.md`）

- ノードは任意の微分可能な `nn.Module` `f` を包む。ノードの内側は VJP（BP）、ノードの間は PC-ALM。バッチの規約は「VJP はバッチの和、1/B は θ の勾配にだけ」（ユーザー承認）
- **状態は予測からのずれ u = h − μ と λ で持ち、入力が変わってもリセットしない**（ユーザー「素晴らしい」）。h を状態にすると i.i.d. のバッチで h_old − μ_new が θ を壊すが、u なら持ち越すのは credit 程度。代わりに記事の離散更新との一致はオラクルにできず、不動点での BP 一致を使う
- **ターゲットは入力に付随させ、コアの carry で活動と同じメッセージに載せて中継する**（2026-10-20、ユーザーの決定。当初は入力の時刻で揃える案だったが、伝播より速く書くと一致しなくなるので切り替えた）。別の Port に分けると読み手が別々の瞬間に読み組み違うので、1 つのメッセージに入れる。コアは `Message` のバッファを `Arc` で持ちゼロコピーで中継する
- θ の更新間隔は秒の `learn_interval`、入力の時刻が変わったときと学習したときに位相を戻す。重みは前回からの Σdt（ユーザー承認）
- 周期の既定は τ_h/4（ユーザーの提案）。λ は更新後の u で更新（公式実装 SakanaAI/pc-alm と同じ順）
- 公開名 `pc.Layer`・`pc.Output`・`pc.TimeConstants` はユーザー承認。`pc.Input` も承認（2026-10-20）。付随データは `pc.Carried` / `pc.carry_spec`
- 数値の状態（入力の時刻・位相・経過時間）のためにコアへ `Node.register_state(name)` を足す（ユーザーの提案。任意の値、復元は代入で差し替え）

**Why:** 同期シミュレーションの MNIST（3 層、B=64）で、u で持ち持ち越す方式が S=32 で BP と同等（relu 97.60 対 97.65）、勾配も cos 1.000 だった。
