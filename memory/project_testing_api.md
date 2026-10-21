---
name: project-testing-api
description: テスト用の口は公開し、mujou 自身と利用者のテストが同じ口を使う。mujou.testing.run と ManualRun.step(dt, *nodes)。VirtualTimer は mujou.time にある
metadata:
  type: project
---

2026-10-16 のユーザーの回答。利用者向けのテストの口は公開し、mujou 自身のテスト（既定のノードのテストを含む）と利用者のテストは、同じ口を使う。

- `mujou.testing.run(root, *, timer=None, inputs=None, state=None) -> ManualRun`: 既定のタイマーは新しい `VirtualTimer`（`state` を渡したなら保存した時刻から）
- `ManualRun.step(dt, *nodes)`: タイマーを `dt` だけ進めてから、`nodes` を渡した順に 1 回ずつ更新する。省くと木の順（Gauss-Seidel）。周期は無視する（ユーザーの決定）。同じ step に 2 回渡したノードは、2 回目は時刻が進んでいないので飛ばす
- `VirtualTimer` は `mujou.time` に置く（時間の名前を 1 か所に集める）

**Why:** 利用者も自分のノードを決定的にテストできる必要がある。mujou と利用者で別の口を持つと、片方だけが古くなる。advance と step を別々に呼ぶ形は、advance を忘れると黙って何も起きなかったので、`step(dt)` に 1 つにした。

**How to apply:** テストは `mujou.testing.run` と `step(dt)` で書く。利用者が使うべきでない内部の口は公開の API に見せない。
