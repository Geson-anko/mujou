---
name: project-gpu-policy
description: mujou のコアは GPU をサポートしない。コアの値はホストのメモリだけに置き、GPU を使うノードは境界で device → host → device とコピーする。DLPack は CPU 上の受け渡しの形式としてだけ使う。ノードのデバイスは CPU と CUDA だけ（MPS などはサポートしない）。ノードの内側で GPU を使うのは自由
metadata:
  type: project
---

2026-10-15 決定。

- **コアとメッセージは GPU をサポートしない。** コアの値はホストのメモリにだけ置く。GPU を使うノードは、ノードの境界で device → host → device とコピーする
- **DLPack は、ゼロコピーや GPU 上の値の共有には使わない。** CPU 上の `torch.Tensor` とコアの間の受け渡しの形式としてだけ使い、読み書きとも必ずコピーする（[[project-message-types]]）
- **ノードのデバイスは CPU と CUDA だけ（2026-10-17、ユーザーの決定）。** それ以外の種類（`mps`、`xpu` など）は、torch が使えるかどうかに関わらず `Node(device=...)` が断る（K4）
- ノードの内側で PyTorch を使うときの設定（ノードごとの stream、同期）は Driver が行う（[[project-torch-module]]）

**Why:** コアの値をデバイス上に置くと、スロットにバッファと CUDA event の組を持たせ、stream をまたぐ同期、バッファの再利用の安全性、タイムスタンプの意味をコアが背負う。ストリームから 1 サンプルずつ学ぶ非同期のノードは小さな行列×ベクトル積の連続になり、GPU の効率はもともと出にくい。それでも大きな行列で CPU を食い潰さないよう、ノードの内側での利用は妨げない。MPS は 2 スレッドから同時に使うとプロセスが落ち（torch 2.14.1、GIL ありでも 10/10 で再現）、ノードを複数のスレッドで並行に計算する実行モデルと合わない。

**How to apply:**

- コア・スロット・バッファプールに device・stream・event の概念を入れない
- GPU を使うノードは、ステップごとに自分の stream を待ってからホストへ書き出す（Driver が行う）。stream を分ければ、正しさや順序の面ではノード間で影響しない。資源の取り合いは Δt の伸びとして吸収する
- ノード・ランタイムに `torch.accelerator` の汎用経路や MPS 向けの分岐を入れない。サポートするデバイスを増やすときはユーザーに確認する

関連: [[project-native-core-rust]], [[project-compute-granularity]]
