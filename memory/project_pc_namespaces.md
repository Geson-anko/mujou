---
name: project-pc-namespaces
description: mujou.pc の公開名は役割ごとのサブモジュール（optim / relax / losses / messages / clock）に分けた。pc 直下に残すのは Chain・ノード・TimeConstants だけ（2026-10-27 のユーザーの指示）
metadata:
  type: project
---

2026-10-27、ユーザーが「`mujou.pc` の公開の形がすべてフラットで名前空間がごちゃつく。optim、relax 等に分けて」と指示した（ブランチ refactor/2026-10-27/pc-namespaces）。

- `pc.optim`: `SGD` `AdamW` `Optimizer` `BoundOptimizer`／`pc.relax`: `Euler` `HeavyBall` `Relaxer` `BoundRelaxer`／`pc.losses`: `MSELoss` `CrossEntropyLoss`／`pc.messages`: `Sample` `Activity` `Prediction` `activity_spec`／`pc.clock`: `LearnClock` `Tick`
- `pc` 直下は `Chain` `Shown` `show` `Input` `Layer` `Output` `TimeConstants` と、上のサブモジュール（`functional` を含む）。`import mujou.pc as pc` だけで `pc.optim.SGD` と書ける
- 各サブモジュールは `__all__` を持つ。`pc.SGD` のような旧名は残していない（pre-release のため互換 alias なし）

**Why:** 名前が 20 以上平らに並ぶと、利用者が役割（更新手法・損失・メッセージ）を探しにくい。

**How to apply:** 新しい更新手法や損失は対応するサブモジュールに足す。pc 直下には足さない。
