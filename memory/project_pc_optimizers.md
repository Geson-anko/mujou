---
name: project-pc-optimizers
description: mujou.pc の重みは時間で進める optimizer（pc.SGD / pc.AdamW）が動かす。τθ は TimeConstants から optimizer に移した。設定（frozen dataclass）をノードが bind して状態を持つ束縛を作る。torch.optim は使わない。h と λ の則の差し替えは段階 2
metadata:
  type: project
---

設計の正本は `specs/pc.md` の「重みの optimizer」。実装は `src/mujou/pc/optim.py`。

- **τθ を `TimeConstants` から optimizer に移した（2026-10-22、ユーザーの決定）。** `TimeConstants(*, h, lam)`。破壊的変更で互換は残さない。学ばないノードは `learn_interval=math.inf` の 1 通りで表す（`tau=inf` は K31 で断る）
- **設定 + bind（ユーザーの決定）。** `pc.SGD(tau=)` と `pc.AdamW(tau=, tau_m=, tau_v=, eps=1e-8, weight_decay=1e-2)` は状態を持たない frozen dataclass（`Optimizer` Protocol）。ノードが `bind(params)` で束縛（`BoundOptimizer` Protocol: `step_(grads, elapsed)`、`state_dict`、`load_state_dict`）を作り `self.optimizer` に置く。状態の木のキーは `"optimizer"`。名前 `BoundOptimizer` と AdamW の既定（torch と同じ）はユーザーが推奨案で確定
- **`optimizer` は `Layer`・`Output`・`Chain` の必須の引数。** 時間で表した既定の学習率は無い
- **1/B はノードが掛けてから渡す。** optimizer はバッチを知らない
- **AdamW は時間で進める。** `a = 1 - exp(-Δt/τ)`、`m ← m + a(g - m)`、bias 補正はステップのカウンタの代わりに質量 `c ← c + a(1 - c)`（Python の float）。`elapsed == 0` は何もしない。θ は torch と同じ順（decay を歩みの前の θ に先に掛け、Adam の項は更新後の m・v・c で取る）。`eps` は有限で > 0（`eps=0` は勾配がずっと 0 の要素を 0/0 で NaN にするので K31 で断る。torch とは違う）。`elapsed` は `Tick.elapsed`（時計が捨てた時間を含まない）で、捨てた時間はモーメントも動かさない。Δt 一定なら `torch.optim.AdamW(lr=Δt/τ, betas=(e^{-Δt/τm}, e^{-Δt/τv}))` と一致する。`weight_decay` は無次元で 1/τθ と組む（torch と同じ意味）
- **段階 2（h と λ の則の差し替え）は研究から。** 今は実装しない（`specs/open-decisions.md`）

**Why:** 3.14t では `torch.optim` のコンストラクタが `torch._dynamo` → `triton._C` を import し、プロセス全体の GIL が戻る。torch の β は 1 ステップあたりで、学習の間隔が変わると実効の時定数が変わる。`state["step"]` は float32 で 2^24 回で止まる。

**How to apply:** `src/mujou/` で `torch.optim` を import しない（テストで torch.optim と比べるときは subprocess で）。新しい optimizer は `Optimizer` / `BoundOptimizer` の Protocol を満たす設定 + 束縛の 2 つのクラスで書き、時定数は秒で持つ。AdamW の推奨の時定数は未測定なので、docs で値を勧めない。

関連: [[project-pc-module]], [[project-pluggable-update-rule]]
