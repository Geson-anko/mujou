---
name: project-no-cuda-graph
description: 2026-11-04 に覆した。pc.Layer の vjp を CUDA graph で記録・再生するオプション（cuda_graph=True、既定は切り）を入れた。もとは 2026-10-29 の「使わない」の決定
metadata:
  type: project
---

**2026-11-04、ユーザーが覆した**（mujou-exp の M16 で、カメラ 1 台・B=1 のストリームを学習込みで実時間の 2 倍（60 fps）で回すという要件のため）。`pc.Layer(cuda_graph=True)` で、ノードの vjp を CUDA graph に記録して再生する（issue #116）。既定は切りで、使うかは利用者が選ぶ。

もとの決定（2026-10-29）: ユーザーが「CUDA graph は使いません」と決めた。理由は聞いていない。背景: GPU 1 枚では全ノードの 1 周期のコストがノードの数によらず約 1.5〜2 ms で、CPU からカーネルを起動するところで律速される（11 スレッドの競合で、1 ノード単独の 0.4 ms の約 5 倍）。CUDA graph にすると 1 周期が 0.4〜0.7 ms（B = 16〜32）まで下がると測っていた。

**How to apply:** CUDA graph は opt-in の速度の手段として使ってよい（`pc.Layer` は `cuda_graph`）。結果がビット単位で同じであることをテストで確かめる。速さは引き続きノードの切り方と周期の数でも稼ぐ。関連: [[project-pc-jacobian-stability]]、[[feedback-pc-speed-budget]]
