---
name: project-gpu-policy
description: mujou は GPU をサポートしない。コアの値はホストのメモリだけに置き、GPU を使うノードは境界で device → host → device とコピーする。DLPack は CPU 上の受け渡しの形式としてだけ使う（ゼロコピー・GPU 共有には使わない）。ノードの内側で GPU を使うのはノードの自由
metadata:
  type: project
---

2026-10-15 決定。

- **mujou（コア・メッセージ）は GPU をサポートしない。** コアの値はホストのメモリにだけ置く。デバイス上の値は持たない
- **GPU を使うノードは、ノードの境界で device → host → device とコピーする。** ノードの内側で GPU を使うかどうかはノードの自由
- **DLPack は、ゼロコピーや GPU 上の値の共有には使わない。** 2026-10-15 に利用者の値を `torch.Tensor` に一本化したとき、CPU 上の `torch.Tensor` とコアの間の受け渡しの形式としてだけ使うことにした。読み書きとも必ずコピーする（[[project-message-types]]）
- ノードの内側で PyTorch を使うときの設定（autograd を切る、ノードごとにストリームを分ける）は `mujou.Node` が `on_start` で行う（[[project-torch-module]]）

**Why:** コアの値をデバイス上に置くと、スロットにバッファと CUDA event の組を持たせる必要がある。さらに、ストリームをまたぐ同期、バッファの再利用の安全性、タイムスタンプの意味、Mac（MPS）で試せないことなどを、すべてコアが背負うことになる。ストリームから 1 サンプルずつ学ぶ非同期のノードは小さな行列×ベクトル積の連続になり、GPU の効率はもともと出にくい。それでも大きな行列で CPU を食い潰さないよう GPU を使いたい場面はあるので、ノードの内側での利用は妨げない。

**How to apply:**

- コア・スロット・バッファプールに device・stream・event の概念を入れない
- GPU を使うノードは、ステップごとに自分のストリームを同期してからホストへ書き出す。ノードごとにストリームを分ければ、正しさや順序の面ではノード間で影響しない（SM・帯域・メモリといった資源の取り合いは、CPU のコアの取り合いと同じく $\Delta t$ の伸びとして吸収する）

## 2026-10-17 追記: CUDA 以外のアクセラレータ

ユーザーの判断で、ノードが MPS など CUDA 以外のアクセラレータでも計算できるようにした（`specs/torch.md` の「CUDA 以外のアクセラレータ」）。

- 受け付けるのは CPU と、torch の現在のアクセラレータ（`torch.accelerator.current_accelerator(check_available=True)`）と同じ種類のデバイス。添字の無いデバイスは `current_device_index()` で固定する（`mps` と `mps:0` は等しくない）
- ノードごとのストリームは CUDA だけ。ほかのアクセラレータはデバイスの現在のストリームで計算し、CUDA でストリームを同期する箇所で `torch.accelerator.synchronize(device)` を呼ぶ
- **制約（未決定の対処）**: PyTorch の MPS は 2 つのスレッドから同時に使うとプロセスが落ちる（torch 2.14.1 で確認）。MPS のノードは 1 つまで、実行中にホストで MPS を使わない、と利用者向けの文書に書いた。ロックで直列にするか、構築時に拒むかは未決定

**Why:** 開発機が Apple Silicon の Mac で、手元で GPU の経路を試したい。ストリームを分けるのは CUDA でしか意味と性能を確かめていないので、ほかのバックエンドには広げない。

関連: [[project-native-core-rust]], [[project-message-types]], [[project-compute-granularity]]
