---
name: project-torch-module
description: torch 非依存は private な mujou._core だけで、それ以外は PyTorch を全面的に採用する（mujou.torch に分けない）。利用者が継承する mujou.Node（コアの Node を継承）が勾配の記録の停止・ノードごとの CUDA ストリーム・乱数生成器を on_start で設定する。対象は CPU と CUDA、MPS は考えない
metadata:
  type: project
---

2026-10-15 決定。

- **モジュール構成: torch に依存しないのは private な `mujou._core`（Rust コアの Python インターフェース。値のバッファ・スロット・コアの `Node` など）だけ。利用者が使う名前は `mujou.*` のトップレベルに公開する。それ以外は PyTorch を全面的に採用し、torch は必須の依存にする。** `mujou.torch` のようにバックエンドで分けない（2026-10-15 にいったん `mujou.torch` に分ける案にしたが撤回）。`mujou.torch` / `mujou.jax` は必要なら mujou 1.0 などの正式版で検討する
- **利用者が継承するノードは `mujou.Node`**（`mujou._core.Node` を継承）。利用者が直接使うので `Torch` の接頭辞を付けない（2026-10-15、`TorchBlock` から改名）
- `mujou.Node` は `on_start`（ノードを実行するスレッドの上で呼ばれる）で、スレッドごとの設定を行う: `torch.set_grad_enabled(False)`、CUDA のときは自分の `torch.cuda.Stream` を現在のストリームにする、ノードごとの `torch.Generator`
- Port の `recv()` / `send()` は `torch.Tensor` でやり取りし、コアとの変換（DLPack、必ずコピー）は `mujou.Node` が受け持つ。利用者は変換を意識しない。書くときは値を `detach()` し、GPU 上の値は自分のストリームでホストへコピーして、そのストリームだけを同期してからコアへ渡す
- パラメータは `nn.Module` / `nn.Parameter` を使わず、`requires_grad=False` のテンソルで持つ。`inference_mode` は既定にしない（ノード内 BP で入力を backward に使えなくなるため）
- 状態は `add_`・`addmm_` などで**その場で更新する**。毎ステップ新しいテンソルを確保すると、多数のスレッドでメモリ確保が律速になる（7950X、n=1024、16 スレッドで 1.1k → 36.9k ステップ/秒）
- torch のノードの専用スレッドは物理コア数を超えないようにする（論理コア数まで増やすと遅くなった）
- 対象デバイスは CPU と CUDA。**Apple の MPS は考えない**

**Why:** GPU をコアでサポートしない（[[project-gpu-policy]]）代わりに、ノードの内側で GPU を使うときにノード間の独立性を保つ設定を、作者が毎回書かずに済むようにするため。PC-ALM は更新式を直接計算するので autograd は要らない。

**JAX は不採用（2026-10-15）。** 本家 PC-ALM が JAX で、導関数を `jax.grad` で得られる点は魅力だったが、3.14t で測ると多数スレッドからの呼び出しが伸びなかった。PC-ALM の 1 層の `on_update`（バッチ 1）を各スレッドが独立に回すと、JAX は合計スループットが 4〜8 スレッドで頭打ちになり、論理コア数と同じスレッド数で崩壊した（Ryzen 9 7950X、n=32 で 32 スレッドのとき 1 ステップ 10ms）。XLA の設定（非同期ディスパッチ・内部スレッド）を変えても直らなかった。GPU でもノードごとにストリームを分けられず、合計が一定で頭打ちになった。初回の測定で JAX が勝った n≥1024 の単独ノードも、torch を状態のその場更新に書き直すと torch が上回った（CPU、n=1024、1 スレッドで 5.6k 対 3.0k、GPU、n=4096 で 6.1k 対 1.4k）。numpy・torch は n≤256 でコア数付近まで比例して伸びた。測定の詳細は `docs/torch.md`。

**How to apply:** 仕様は `docs/torch.md`。`on_start` の override で設定が漏れない仕組み、内部スレッド数の決め方、トップレベルに公開する名前の一覧は未決定（[[project-open-decisions]]）。実装はコアの `Node` ができてから。

関連: [[project-gpu-policy]], [[project-node-lifecycle]]
