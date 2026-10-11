---
name: project-purpose
description: mujou の目的。BP を使わず PC-ALM で学習し、層・ニューロン単位で非同期に学び続けるライブラリ
metadata:
  type: project
---

`mujou` は Asynchronous Distributed Deep Learning Library for Continuous Learning。

- PyTorch のような深層学習フレームワークだが、更新手法は BP ではなく PC-ALM（[[reference-pc-alm]]）
- 名前は「諸行無常」の「無常」から。学習後に重みを固定して推論するしかなかった従来の機械学習に対し、ストリームから学び続け、常に変化し続ける学習機械を作る
- 狙う用途: リアルタイム処理、ロボティクス、生涯学習
- PC-ALM の層局所性を活かし、レイヤーやニューロン単位で非同期に学習と推論を実行できるシステムにする

**Why:** BP はグローバルな forward/backward の同期を要し、継続的・非同期な学習と相性が悪い。PC-ALM は局所更新で BP に近い性能を出せると報告されている。

**How to apply:** 設計判断で迷ったら「局所性」「非同期性」「止まらずに学び続けること」を壊さない側を選ぶ。具体的な不変条件は skill `locality-and-async`。

関連: [[project-open-decisions]]
