---
name: feedback-docs-teaching-quality
description: docs/ で仕組みを解説するページは教材サイト級の質にする。文字だけで並べず、節ごとに要点・図・表・コードを置く（2026-10-21）
metadata:
  type: feedback
---

2026-10-21、ユーザーから「Mujou の仕組みのページが文字ばかりで読みにくい。各セクションで仕組みを解説するとき、教材サイトクラスのクオリティであるべき」と指摘された。

`docs/{en,ja}/concepts.md` を次の形に作り直した。これが基準。

- 冒頭に「このページで分かること」と前提、全体図、部品と節の対応表、PyTorch との対応表
- 各節の頭に `!!! abstract "要点"` で 1〜2 文
- 仕組みは図で見せる（mermaid の flowchart / sequenceDiagram / stateDiagram、mermaid で描けないものはテーマ色に従う inline SVG。CSS は `docs/stylesheets/extra.css` の `.mujou-diagram`）
- 比べるものは表にする（Port と MultiPort、mujou.run と testing.run など）
- 設計の理由は `??? question "〜のはなぜ？"` に畳む
- 規則は一覧表＋節ごとに「避ける / こう書く」のタブ
- 完全なプログラムは examples/ から snippet で取り込み、短い断片だけ inline で書く

**Why:** 文字だけの説明は、非同期・最新値・dt のような直感に反する仕組みが伝わらない。

**How to apply:** docs/ の topics やチュートリアルを書く・直すときも同じ部品を使う。図の中の事実（hook の順、メッセージの中身など）は docstring と specs/ で確かめてから描く。描画はヘッドレスブラウザが無い環境では確かめられないので、`just docs` で見てもらうよう伝える。サイトの構成は [[project-docs-site]]、日本語の文体は [[feedback-japanese-style]]。
