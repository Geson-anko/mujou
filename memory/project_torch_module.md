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
- 対象デバイスは CPU と CUDA。**Apple の MPS は考えない**
- コアは torch に依存しない

**Why:** GPU をコアでサポートしない（[[project-gpu-policy]]）代わりに、ブロックの内側で GPU を使うときにブロック間の独立性を保つ設定を、作者が毎回書かずに済むようにするため。PC-ALM は更新式を直接計算するので autograd は要らない。

**How to apply:** 仕様は `docs/torch.md`。メソッド名、`on_start` の override で設定が漏れない仕組み、内部スレッド数の決め方、optional な依存の方法は未決定（[[project-open-decisions]]）。実装はコアの `Block` ができてから。

関連: [[project-gpu-policy]], [[project-block-lifecycle]]
