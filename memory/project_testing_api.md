---
name: project-testing-api
description: テスト用の口（仮想時計と手動ステップ）は mujou.testing などの名前で公開し、mujou 自身のテスト（既定のノードを含む）と利用者のテストが同じ口を使う。2026-10-16 に実装
metadata:
  type: project
---

2026-10-16 のユーザーの回答。利用者向けのテストの口は公開する。mujou 自身のテスト（既定のノードのテストを含む）と利用者のテストは、同じ公開の口を使う。名前は `mujou.testing` など。中身は仮想時計と、ノードを手で 1 ステップずつ進める仕組み。（当時は実装が後の段階で、ドキュメントには User Guide と API リファレンスの Testing ページの枠だけがあった。同じ日に実装した。下の How to apply）

**Why:** 利用者も自分のノードを決定的にテストできる必要がある。mujou と利用者で別の口を持つと、片方だけが古くなる。

**How to apply:** 実装済み（2026-10-16）: `mujou.testing` の `run` / `ManualRun.step(*nodes)` / `VirtualTimer`。2026-10-17 から `run` の `timer` の既定は実行ごとの新しい `VirtualTimer` で、`ManualRun.timer` が返す。`_manual` と `_step` は消し、mujou のテストもすべてこの口に書き換えた（`memory/agents/plan-implementer/run-handle.md`）。同じ日の決定「利用者が使うべきでないものは公開の API に見せない」の対象は、テスト以外の内部の口になる。
