---
name: project-pc-g-in-after-relax
description: pc.Layer は credit g_in を、その更新で緩和した後の δ から送り、同じ δ' で学ぶ（2026-11-16 のユーザーの判断、open-decisions の案 b）。step(dt) で 1 ステップに 1 ホップ、自由実行で往復が約半分。不動点・vjp の回数・公開 IF は同じ
metadata:
  type: project
---

2026-11-16 のユーザーの判断。PR #118 で `ManualRun.step(dt)` が並列の Jacobi になり、更新前の δ から g_in を送る `pc.Layer` では credit が 2 ステップに 1 ホップしか戻らないと分かった（algorithm-reviewer の S2）。案は (a) learn_interval の目安をスケジュールごとに書き分ける、(b) g_in を緩和した後の δ から送る、の 2 つで、ユーザーは (b) を選んだ。

- 1 回の更新の順: tick → reset → 受け取った g で r、λ を緩和 → δ' = where(mask, r' + λ', 0) で vjp を 1 回 → g_in を送る → μ を送る → 学ぶ
- 重みの勾配 g_θ も δ' から取る（ユーザーの判断。別の δ にすると backward が 2 回になる）。公式実装とは途中の軌道がずれるが、不動点では同じ
- μ は δ に依らないので、前向きのメッセージは変わらない。不動点では δ' = δ なので、BP 一致と α = 0 の退化も変わらない
- credit の速さ（L 層 + Output、hops = L + 1）: `step(h)` は層 i に 2·hops − i ステップ目に届く（前は 1 ホップに 2 ステップ）、出力側からの逐次は hops + 1、自由実行では往復が約 hops·周期（前は約 2·hops·周期）
- learn_interval の目安 2 × hops × 周期は変えない（根拠は「1 ホップに最大 1 周期」に書き直した）。examples の S と learn_interval も変えない（縮めるのは MNIST と CIFAR の測り直しを伴う別のタスク）
- CUDA graph（`pc.Layer(cuda_graph=True)`）のビット一致は CI の cuda ジョブで確かめる

**Why:** credit を受け取ったその更新のうちに前へ渡せば、どのスケジュールでも 1 ホップあたり 1 ステップ縮む。式の目安を書き分けるより、離散化の側を直す方が単純。

**How to apply:** pc のノードを書くとき・参照実装を書くときは、緩和を vjp の前に置く。設計は `specs/pc.md`、理論は skill `pc-alm` の 7.2。
