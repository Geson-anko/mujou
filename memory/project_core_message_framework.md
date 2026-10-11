---
name: project-core-message-framework
description: mujou のコアはブロック間のメッセージプロトコルとその実行フレーム。ブロック内部では PyTorch などを使ってよい
metadata:
  type: project
---

2026-10-11 の議論で合意した方針。

- **mujou のコアは、ブロック単位でメッセージをやり取りするプロトコルと、その実行フレームの実装である。** ブロック（[[project-compute-granularity]]）同士が非同期にメッセージを交換し、推論と学習を同時に進める仕組みそのものがフレームワークの本体になる
- **ブロックの内部では PyTorch などの既存ライブラリを使ってよい。** ブロックの内側は同期的に計算するので、テンソル演算や autograd（ブロック内 BP を含む）を既存ライブラリに任せられる

**Why:** ブロックの内側は自由、ブロックの間は不変条件付きのメッセージ、という分担にしたため（[[project-pluggable-update-rule]]）。mujou 固有の価値は、ブロックの内側の計算ではなく、ブロック同士を非同期につなぐ部分にある。

**How to apply:**

- 設計と実装の優先度は、メッセージプロトコル（何を、どの形式で、いつ送るか）と実行フレーム（スケジューラ・通信・状態の所有）を先にする
- ブロック内部の実装（PyTorch による PC-ALM ブロックや BP ブロックなど）は、プロトコルの上に載る差し替え可能な部品として扱う
- プロトコルとフレームは Rust で実装し、メッセージの値の型は独自に持つ。特定のテンソルライブラリには依存しない。PyTorch テンソル連携は future work（[[project-native-core-rust]]）

関連: [[project-purpose]], [[project-compute-granularity]], [[project-pluggable-update-rule]], [[project-native-core-rust]]
