---
name: project-torch-module
description: PyTorch を使うときの支援は mujou.torch モジュールに置き、基底クラス TorchBlock が勾配の記録の停止・ブロックごとの CUDA ストリーム・乱数生成器を on_start で設定する。対象は CPU と CUDA、MPS は考えない
metadata:
  type: project
---

2026-10-15 決定。

- PyTorch を使うときの便利機能は `mujou.torch` モジュールに実装し、そこに `TorchBlock`（`Block` を継承）を置く
- TorchBlock は `on_start`（ブロックを実行するスレッドの上で呼ばれる）で、スレッドごとの設定を行う: `torch.set_grad_enabled(False)`、CUDA のときは自分の `torch.cuda.Stream` を現在のストリームにする、ブロックごとの `torch.Generator`
- 境界の変換（Tensor ⇄ torch のテンソル）も TorchBlock が提供する。書き出すときは自分のストリームだけを同期する
- パラメータは `nn.Module` / `nn.Parameter` を使わず、`requires_grad=False` のテンソルで持つ。`inference_mode` は既定にしない（ブロック内 BP で入力を backward に使えなくなるため）
- 状態は `add_`・`addmm_` などで**その場で更新する**。毎ステップ新しいテンソルを確保すると、多数のスレッドでメモリ確保が律速になる（7950X、n=1024、16 スレッドで 1.1k → 36.9k ステップ/秒）
- torch のブロックの専用スレッドは物理コア数を超えないようにする（論理コア数まで増やすと遅くなった）
- 対象デバイスは CPU と CUDA。**Apple の MPS は考えない**
- コアは torch に依存しない

**Why:** GPU をコアでサポートしない（[[project-gpu-policy]]）代わりに、ブロックの内側で GPU を使うときにブロック間の独立性を保つ設定を、作者が毎回書かずに済むようにするため。PC-ALM は更新式を直接計算するので autograd は要らない。

**JAX は不採用（2026-10-15）。** 本家 PC-ALM が JAX で、導関数を `jax.grad` で得られる点は魅力だったが、3.14t で測ると多数スレッドからの呼び出しが伸びなかった。PC-ALM の 1 層の `on_update`（バッチ 1）を各スレッドが独立に回すと、JAX は合計スループットが 4〜8 スレッドで頭打ちになり、論理コア数と同じスレッド数で崩壊した（Ryzen 9 7950X、n=32 で 32 スレッドのとき 1 ステップ 10ms）。XLA の設定（非同期ディスパッチ・内部スレッド）を変えても直らなかった。GPU でもブロックごとにストリームを分けられず、合計が一定で頭打ちになった。初回の測定で JAX が勝った n≥1024 の単独ブロックも、torch を状態のその場更新に書き直すと torch が上回った（CPU、n=1024、1 スレッドで 5.6k 対 3.0k、GPU、n=4096 で 6.1k 対 1.4k）。numpy・torch は n≤256 でコア数付近まで比例して伸びた。測定の詳細は `docs/torch.md`。

**How to apply:** 仕様は `docs/torch.md`。メソッド名、`on_start` の override で設定が漏れない仕組み、内部スレッド数の決め方、optional な依存の方法は未決定（[[project-open-decisions]]）。実装はコアの `Block` ができてから。

関連: [[project-gpu-policy]], [[project-block-lifecycle]]
