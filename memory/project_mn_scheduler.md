---
name: project-mn-scheduler
description: 2026-10-18 のユーザーの要求で、Python のノードも既定で実行ごとのワーカープール（M:N）に載せた。専用スレッドは Node(dedicated_thread=True) で opt-in、プールは実行ごと（mujou.run(workers=None)）、ノードはワーカー間を移ってよい
metadata:
  type: project
---

2026-10-18、ユーザーの要求「大量のスレッドを立てると OS のコンテキストスイッチを取り合うので、M:N でなければならない」。それまで Python のノードは全部専用スレッドだった（GIL を理由にプールへ載せない、[[project-native-core-rust]] の旧記述）。計画は `memory/agents/implementation-planner/mn-scheduler.md`、PR は 3 つ（① コアのワーカープール ② Python のノードを載せ既定を切り替え ③ ベンチマーク）。

ユーザーの決定（同日、推奨案どおり）:

- **Q1 専用スレッドは opt-in で残す。** `Node(dedicated_thread=False)` が既定で全ノードがワーカーに載る。`on_thread_start` は専用スレッドのノードだけに、そのスレッドで `on_start` の直前に呼ぶ。ワーカーに載るノードのクラスが override していたら `mujou.run` が何も起動しないうちに `TypeError`
- **Q2 プールは実行ごと。** `mujou.run(..., workers=None)` で指定する。既定は `min(ワーカーに載るノードの数, 論理 CPU 数)`。`mujou.testing.run` には足さない
- **Q3 ノードはワーカーの間を移ってよい（pinning しない）。** hook の間で保つのは `contextvars`（ノードごとの `contextvars.Context`）と、フレームワークが hook の前に戻す勾配モード・CUDA の device と stream だけ。`threading.local`・`threading.current_thread()` は保たない

実装（PR 2）: 内部の hook `_on_thread_start` を `_on_enter` に置き換え、スレッドがノードに切り替わるたびに設定し直す（コアの `NodeHooks::enter`）。Port の所有はスレッド ID でなく「スレッドが今走らせているノード」（`hooks.current_node()`）で判定する。レビューを受けて、勾配の記録は切り替えのときだけでなく、`state_dict` 以外の公開の hook の前に毎回切るようにした（内部の hook `_before_hook`。決定性のため、2026-10-18 の orchestrator の裁定）。CUDA の device と stream は切り替えのたび。hook の中で変えたほかのスレッドごとの設定（`torch.autocast` など）は、利用者が hook を抜ける前に戻す（戻さないと同じワーカーの次のノードに漏れる）。この漏れをフレームワークで防ぐか（autocast も切る、debug 時に検出する）は、2026-10-18 のユーザーの決定で「文書だけ」にした。設定の種類は無数にあり、網羅的に戻すのは無理なため

**Why:** ノードごとに OS スレッドを立てると、ノードが多いときにコンテキストスイッチとスレッド数に縛られる。ブロッキングする I/O のノードはワーカーを占有するので、逃げ道として専用スレッドを残した。プロセス共有のプールや pinning は、実行ごとの停止・終了の分離が複雑になる・負荷の偏りを直せないので採らなかった。

**How to apply:** ノードの hook に「スレッドが固定」という前提を持ち込まない。スレッドごとの設定は hook の中の `with` に閉じるか、`_on_enter` で毎回設定する。既定のワーカー数が物理コア数より遅いかどうかは PR 3 のベンチマークで確かめる（[[project-open-decisions]]）。
