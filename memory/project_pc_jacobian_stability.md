---
name: project-pc-jacobian-stability
description: pc.Chain（周期 τ_h/4 の陽的 Euler）の緩和は次のノードのモジュールのヤコビアンの最大特異値 σ ≲ 2.6 でしか安定しない。ResNet のステージ（学習後 5〜12）・ブロック（3〜6）は超える。遅れると境界は 1.2〜1.8。GPU のノードは境界のコピーが重い
metadata:
  type: project
---

2026-10-24、`examples/cifar10/` を作る中で測った（記録は `memory/agents/orchestrator/cifar10-resnet-example.md`）。

- `h` の緩和の線形化は `-(I + J^T J)/τ_h`（J は次のノードのモジュールのヤコビアン）。Δt = τ_h/4 の陽的 Euler では `1 + σ² < 8`、σ < 約 2.65（λ との結合は無視した見積もり）
- dt が max_dt（2 周期）まで伸びる、または 1 周期古い値を読むと、境界は線形の鎖で 1.2〜1.8 に下がる（algorithm-reviewer が pc.Chain で測定）。次が Output なら損失の曲率（CE は 1/2 以下）が σ² に掛かる
- ResNet-20（GroupNorm）を BP で学習すると、ステージの σ は 5〜12（10 epoch まで）、ブロック単体でも 3〜6。ステージごとのノードでは lam が際限なく増え、BP の credit と無相関になった。平均プーリングの head を含む合成（stage3∘head）だけが小さい: 3 epoch で 1.3、10 epoch で 2.2
- MNIST の MLP では問題にならなかった。σ は初期化（恒等写像から始める残差）では 1 でも、学習で育つ
- GPU のノードは、境界ごとにメッセージを device→host→device とコピーする（[[project-gpu-policy]]）。ブロックごとのノード・B=128（8 MB/メッセージ）では更新の CPU 時間の 6 割がコピー、2 ノード・B=64 では 1/4〜1/3。ノードを細かくするほど遅い

**Why:** 更新手法が「どこでノードを切れるか」を数理的に制約する。ノードの粒度はクラスで決まる（[[project-compute-granularity]]）が、深い畳み込みのネットワークでは今の pc の定式化では細かく切れない。

**How to apply:**

- pc のノードの切り方・新しいネットワークの例を設計するときは、学習後の σ(J_next) を見積もり、遅れのないときの境界 2.6 と遅れたときの 1.2〜1.8 に照らす。超えるなら切れ目を移すか、周期と max_dt を短くする（未測定）
- 細かい粒度を実現したいときの候補（陰的・前処理つきの緩和、ノードごとの τ_h）は src/ の pc の変更になるので、ユーザーに確認する
- 関連: [[project-pc-module]], [[reference-pc-alm]]
