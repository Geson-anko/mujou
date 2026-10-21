---
name: project-mn-scheduler
description: Python のノードも既定で実行ごとのワーカープール（M:N）に載せる。専用スレッドは Node(dedicated_thread=True) で opt-in、mujou.run(workers=None)。既定は available_parallelism（quota を数える）、GIL ありならその半分（確認待ち）、実効の数はプールに載るノードの数で頭打ちし 0 もありうる。ノードはワーカー間を移ってよい
metadata:
  type: project
---

2026-10-18、ユーザーの要求「大量のスレッドを立てると OS のコンテキストスイッチを取り合うので、M:N でなければならない」。

- **専用スレッドは opt-in で残す（ユーザーの決定）。** `Node(dedicated_thread=False)` が既定で全ノードがワーカーに載る。`dedicated_thread=True` はブロッキングする I/O のノードのため。スレッドごとの設定は自分のスレッドで呼ばれる `on_start` に書く
- **プールは実行ごと（ユーザーの決定）。** `mujou.run(..., workers=None)`。`mujou.testing.run` には無い
- **ワーカー数の既定:** bindings が決める。free-threaded では Rust の `available_parallelism`（cgroup の quota と affinity を数える）、GIL が有効なとき（`sys._is_gil_enabled()`）はその半分（少なくとも 1。半分にしたのは推奨案で、ユーザーの確認待ち。`specs/open-decisions.md`）。実効の数は `min(要求か既定, プールに載るノードの数)` で、全ノードが専用スレッドなら 0。明示した値が 1 未満なら K20。テストは `Runtime.workers` で見る
- **ノードはワーカーの間を移ってよい（pinning しない、ユーザーの決定）。** hook の間で保つのは `contextvars`（ノードごとの Context）と、Driver が設定する CUDA の device と stream だけ。`threading.local` は保たない
- hook の中で変えたほかのスレッドごとの設定（`torch.autocast` など）は、利用者が hook を抜ける前に戻す。フレームワークで防ぐかは「文書だけ」（2026-10-18、ユーザーの決定。設定の種類は無数にある）

**Why:** ノードごとに OS スレッドを立てると、ノードが多いときにコンテキストスイッチとスレッド数に縛られ、ノード間の不公平も大きい。プロセス共有のプールや pinning は、実行ごとの停止・終了の分離が複雑になり、負荷の偏りを直せない。GIL ありの既定を半分にしたのは、論理コア数のワーカーで約半分の速さになったため（`specs/research/benchmarks.md`）。

**How to apply:** ノードの hook に「スレッドが固定」という前提を持ち込まない。SMT のある機械での論理と物理の差は未確認（`specs/native-core.md` の「ワーカー数の既定」）。
