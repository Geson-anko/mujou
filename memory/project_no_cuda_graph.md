---
name: project-no-cuda-graph
description: ノードの更新を CUDA graph にしない（2026-10-29 ユーザー決定）。1 周期のコストは CPU の起動律速のまま受け入れ、ノードを少なく切って周期を減らす
metadata:
  type: project
---

2026-10-29、ユーザーが「CUDA graph は使いません」と決めた。理由は聞いていない。

背景: GPU 1 枚では全ノードの 1 周期のコストがノードの数によらず約 1.5〜2 ms で、CPU からカーネルを起動するところで律速される（11 スレッドの競合で、1 ノード単独の 0.4 ms の約 5 倍）。ノードの更新を CUDA graph にすると 1 周期が 0.4〜0.7 ms（B = 16〜32）まで下がると測っていた。採らなかったので、CIFAR-10 の例はステージごとの 5 ノード・B = 32・12 周期で BP の約 5.5 倍に収めている。ブロックごとの 11 ノードは BP の 10 倍の予算（[[feedback-pc-speed-budget]]）を超える。

**How to apply:** 速度の改善案として CUDA graph（`torch.cuda.CUDAGraph`、`torch.compile(mode="reduce-overhead")` を含む）を提案しない。速さはノードの切り方と周期の数で稼ぐ。関連: [[project-pc-jacobian-stability]]、[[project-open-decisions]]
