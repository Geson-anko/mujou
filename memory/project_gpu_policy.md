---
name: project-gpu-policy
description: mujou は GPU をサポートしない。Tensor はホストのメモリだけに置き、GPU を使うブロックは境界で device → host → device とコピーする。DLPack は使わない。ブロックの内側で GPU を使うのはブロックの自由
metadata:
  type: project
---

2026-10-15 決定。

- **mujou（コア・メッセージ）は GPU をサポートしない。** Tensor はホストのメモリにだけ置く。デバイス上の Tensor は持たない
- **GPU を使うブロックは、ブロックの境界で device → host → device とコピーする。** ブロックの内側で GPU を使うかどうかはブロックの自由
- **DLPack は使わない。** 外部ライブラリとの変換は従来どおりコピーで行う（[[project-message-types]]）
- ブロックの内側で PyTorch を使うときの支援（autograd を切る、ブロックごとにストリームを分ける）は別途検討する

**Why:** Tensor をデバイス上に置くと、スロットにバッファと CUDA event の組を持たせる必要がある。さらに、ストリームをまたぐ同期、バッファの再利用の安全性、タイムスタンプの意味、Mac（MPS）で試せないことなどを、すべてコアが背負うことになる。ストリームから 1 サンプルずつ学ぶ非同期のブロックは小さな行列×ベクトル積の連続になり、GPU の効率はもともと出にくい。それでも大きな行列で CPU を食い潰さないよう GPU を使いたい場面はあるので、ブロックの内側での利用は妨げない。

**How to apply:**

- コア・スロット・バッファプールに device・stream・event の概念を入れない
- GPU を使うブロックは、ステップごとに自分のストリームを同期してからホストへ書き出す。ブロックごとにストリームを分ければ、正しさや順序の面ではブロック間で影響しない（SM・帯域・メモリといった資源の取り合いは、CPU のコアの取り合いと同じく $\Delta t$ の伸びとして吸収する）

関連: [[project-native-core-rust]], [[project-message-types]], [[project-compute-granularity]]
