---
name: project-pluggable-update-rule
description: 更新手法は差し替え可能にする。PC-ALM は現状の既定として暫定的に選んだものであり、フレームワークの本質ではない
metadata:
  type: project
---

2026-10-11 決定。

- PC-ALM は、**現時点の既定の更新手法として暫定的に**採用したものである。mujou の本質ではない
- フレームワークは、PC-ALM 以外の更新手法も与えられるようにする。更新手法を差し替えられることを重要な要件として扱う
- mujou の本質は「推論と学習が同時に行われ、知能に『無常』（常が無いこと）を実現する」ことにある（[[project-purpose]]）。特定の学習則は、それを実現するための手段のひとつに過ぎない

**Why:** 継続的に変化し続ける知能を実現する手段は、PC-ALM に限らない。学習則をフレームワークに焼き付けると、将来よりよい手法が出てきたときに差し替えられない。

**How to apply:**

- ノードという実行の基盤（[[project-compute-granularity]]）と、更新手法の実装を分離する。ノード・メッセージ・スケジューラは、特定の学習則を前提にしない
- PC-ALM 固有の状態（双対変数 $\lambda$ など）やメッセージの中身（$W^\top \varepsilon$ など）は、PC-ALM の更新手法の側に閉じ込める。基盤の型や API に直接書き込まない
- 不変条件（skill `locality-and-async`）はノードの間に課す。ノードの内側の更新手法は自由に選べる。BP のような非局所的な手法も、1 つのノードの内側に閉じていれば取り込める（[[project-compute-granularity]]）
- skill `pc-alm` は既定の更新手法についての文書として扱う

関連: [[project-purpose]], [[project-compute-granularity]], [[reference-pc-alm]]
