# コアのランタイム（`mujou.run(network)` まで）

2026-10-15、agent-team-startup。ユーザーの指示: PR を stack、並列化を最大化、コメントは doc-teacher-student、日本語は yomiyasu。疑問は書き溜めて最後に提示し、推奨で進める。

計画書: `memory/agents/implementation-planner/core-runtime.md`

## PR の stack

1. `chore/2026-10-15/core-scaffold`（main から）: torch の追加（Linux は CPU 版 index）、並列作業の衝突を避ける骨組み、計画書
2. `feature/2026-10-15/core-slots`（WP-A）
3. `feature/2026-10-15/core-runtime`（WP-B）
4. `feature/2026-10-15/network-structure`（WP-C）
5. `feature/2026-10-15/run-network`（WP-D）

A/B/C は scaffold から別 worktree で並列に実装し、stack にするときは前の PR のブランチを merge する（rebase・force push はしない）。

## 裁定

- 計画書の確認事項 1〜12 はすべて推奨案で進める（ユーザーへの質問として最後に提示）

## ユーザーへの質問（書き溜め）

計画書「確認事項」の 1〜12 に加えて:

- Q-a: docs は `net.run()`、指示は `mujou.run(network)`。両方を用意し、`mujou.run` は未 build なら build してネットワークを返す形にした
- Q-b: PC-ALM（`mujou.pc`）は「`mujou.run` まで」の範囲外として今回入れていない
- Q-c: タスク実行クラス（ワーカープール）はネイティブノードが無いので future work にした。今は全ノードが専用スレッド
