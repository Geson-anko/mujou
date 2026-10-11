---
name: project-purpose
description: mujou の目的。推論と学習を同時に行い、知能に「無常」（常が無いこと）を実現するフレームワーク。既定の更新手法は暫定的に PC-ALM
metadata:
  type: project
---

`mujou` は Asynchronous Distributed Deep Learning Library for Continuous Learning。

- 本質は、**推論と学習が同時に行われ、知能に「無常」（常が無いこと）を実現する**こと
- PyTorch のような深層学習フレームワーク。現状の既定の更新手法は PC-ALM（[[reference-pc-alm]]）だが、これは暫定的な選択であり、他の更新手法も与えられるようにする（[[project-pluggable-update-rule]]）
- 名前は「諸行無常」の「無常」から。学習後に重みを固定して推論するしかなかった従来の機械学習に対し、ストリームから学び続け、常に変化し続ける学習機械を作る
- 狙う用途: リアルタイム処理、ロボティクス、生涯学習
- 局所的な更新則（既定では PC-ALM）を使い、非同期のブロック（ニューロン・レイヤー・スタックのいずれの大きさにもできる）単位で学習と推論を実行できるシステムにする（[[project-compute-granularity]]）

**Why:** BP はグローバルな forward/backward の同期を要し、継続的・非同期な学習と相性が悪い。PC-ALM は局所更新で BP に近い性能を出せると報告されているため、既定に選んだ。

**How to apply:** 設計判断で迷ったら「局所性」「非同期性」「止まらずに学び続けること」を壊さない側を選ぶ。具体的な不変条件は skill `locality-and-async`。

関連: [[project-open-decisions]], [[project-pluggable-update-rule]]
