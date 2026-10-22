---
name: project-core-message-framework
description: mujou のコアの責務は、ノードを非同期・分散的に回すこと（プロトコル + 実行フレーム）と、非同期・分散的に学習するためのフレームワーク自体の提供（PC-ALM）。ノード内部では PyTorch などを使ってよい
metadata:
  type: project
---

2026-10-11 の議論で合意した方針。

- **mujou のコアは、ノード単位でメッセージをやり取りするプロトコルと、その実行フレームの実装である。** ノード（[[project-compute-granularity]]）同士が非同期にメッセージを交換し、推論と学習を同時に進める仕組みそのものがフレームワークの本体になる
- **2026-10-13 追加: 非同期・分散的に学習するためのフレームワーク自体の提供も、コアの責務とする。** ノードを回す基盤だけでなく、その上で学習するためのフレームワークもコアが提供する。学習フレームワークとして提供するのは PC-ALM（2026-10-13 決定）。ここでのコアは mujou の中核の機能の意味で、PC-ALM は Rust の `mujou._native` ではなく PyTorch で実装し、`mujou.pc` から公開する（2026-10-15、[[project-pc-module]]）
- **ノードの内部では PyTorch などの既存ライブラリを使ってよい。** ノードの内側は同期的に計算するので、テンソル演算や autograd（ノード内 BP を含む）を既存ライブラリに任せられる

**Why:** ノードの内側は自由、ノードの間は不変条件付きのメッセージ、という分担にしたため（[[project-pluggable-update-rule]]）。mujou 固有の価値は、ノードの内側の計算ではなく、ノード同士を非同期につなぐ部分にある。

**How to apply:**

- 設計と実装の優先度は、メッセージプロトコル（何を、どの形式で、いつ送るか）と実行フレーム（スケジューラ・通信・状態の所有）を先にする
- PC-ALM をコアで提供しても、ノード・スロット・スケジューラは特定の学習則を前提にしない。更新手法の差し替え（[[project-pluggable-update-rule]]）と、ノード間の局所性・非同期性の不変条件を保つ
- ノード内部の実装（PyTorch による PC-ALM ノードや BP ノードなど）は、プロトコルの上に載る差し替え可能な部品として扱う
- プロトコルとフレームは Rust で実装し、メッセージの値の型は独自に持つ（`mujou._native` の内部に閉じ、利用者には `torch.Tensor` と Python のスカラーだけを見せる）。特定のテンソルライブラリには依存しない。torch に依存しないのは private な拡張モジュール `mujou._native` だけで、それ以外は PyTorch を全面的に採用する（[[project-native-core-rust]], [[project-torch-module]]）

関連: [[project-purpose]], [[project-compute-granularity]], [[project-pluggable-update-rule]], [[project-native-core-rust]]
