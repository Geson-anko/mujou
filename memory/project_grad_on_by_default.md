---
name: project-grad-on-by-default
description: 2026-10-19 ユーザーの決定。フレームワークは勾配の記録を切り替えない（hook の前に毎回切る _before_hook を廃止）。hook は通常の PyTorch と同じく勾配を記録し、ノードをまたぐグラフは Port の detach コピーで構造的に作れない。勾配の要らない更新手法（PC）は自分で no_grad を使う
metadata:
  type: project
---

2026-10-19、ユーザーの決定。mujou は勾配の記録を切り替えない。以前は `state_dict` 以外の hook の前に毎回 `torch.set_grad_enabled(False)` していた（内部の hook `_before_hook`、2026-10-18 の orchestrator の裁定）が、Rust 側の仕組みごと廃止した。

- hook は通常の PyTorch と同じく勾配を記録する。ノード内の BP は `torch.enable_grad()` も目印も無しに書き、パラメータのその場の更新は `torch.no_grad()` の中で行う（`examples/torch_node.py`）
- ノードをまたぐ計算グラフは作れない: Port は `detach` したコピーを送る（受け手は `requires_grad=False`）
- 勾配を切るのは「PC の実装時だけに止めるべき」（ユーザー）。PC など勾配の要らない更新手法は、自分の hook の中で `torch.no_grad()` を使う
- hook の前に毎回オンへ戻すこともしない。勾配のモードはスレッドごとの設定で、`with` 無しで変えると同じワーカーの次のノードに残るが、それは利用者の責任（ユーザー）。autocast などと同じく「文書だけ」で扱う（[[project-mn-scheduler]]）
- `# mujou: local-autograd` の目印と PostToolUse の locality guard は、`src/mujou/` の中のコード向けに残す。利用者のノード（`examples/`、docs）では使わない

**Why:** ノードをまたぐグラフは detach コピーで構造的に防げているので、局所性のために既定で勾配を切る必要が無い。既定で切ると BP を書く利用者に mujou 固有の書き方を強いる。

**How to apply:** ノードの例・docs はふつうの PyTorch の書き方で書く。注意として、パラメータから計算した値を属性で次の更新へ持ち越すとグラフが伸び続ける（detach するか no_grad の下で計算する）ことを案内する。関連: [[project-torch-module]]。
