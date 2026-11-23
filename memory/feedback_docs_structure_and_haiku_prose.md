---
name: feedback-docs-structure-and-haiku-prose
description: docs/ の改稿は Opus が構造・図表を設計し、日本語の本文は haiku（medium）に書かせ、Opus が事実と表現を検査する。言葉より図・表で整理する（2026-11-23）
metadata:
  type: feedback
---

2026-11-23、ユーザーから「やはりドキュメントが読みにくい。haiku の medium effort は直感的に読みやすい日本語を書く。ただし図表が足りず言葉だけの箇所が多い。言葉よりも視覚的・構造的に整理されたドキュメントに」と指示された。

分担:

- 構造（節の順、mermaid / inline SVG の図、表、タブ、畳んだ「なぜ？」）は Opus が設計して書き切る。本文の位置には伝える事実を全部書いたプレースホルダー `<!-- PROSE: ... -->` を置く
- プレースホルダーを haiku（effort medium）が自然な日本語の本文に置き換える。図・表・コードには触らせない
- haiku は直感的でも誤った表現（条件の脱落、「常に」と「既定で」の取り違え、主語の欠落）を書くことがあるので、Opus が元のページとソースに照らして検査し、正しい言い回しは残す
- 英語版は同じ構造にそろえる

**Why:** Opus が書く日本語は硬く翻訳調になりやすく、文字だけの説明は非同期・dt のような仕組みが伝わらない。

**How to apply:** docs/ の大きな改稿ではこの分担で進める（workflow で群ごとに pipeline: 設計 → haiku → 検査 → 英語 → ビルド）。図表の基準は [[feedback-docs-teaching-quality]]、事実の置き場所と長さは [[feedback-docs-one-fact-one-place]]、文体は [[feedback-japanese-style]]。
