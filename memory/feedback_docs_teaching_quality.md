---
name: feedback-docs-teaching-quality
description: docs/ で仕組みを解説するページは教材サイト級の質にする。文字だけで並べず、図・表・コードで見せる。要点の箱は置かず、図表で示したことを本文に書かない（2026-10-21、2026-11-23 改訂）
metadata:
  type: feedback
---

2026-10-21、ユーザーから「Mujou の仕組みのページが文字ばかりで読みにくい。各セクションで仕組みを解説するとき、教材サイトクラスのクオリティであるべき」と指摘された。

`docs/{en,ja}/concepts.md` を図表中心に作り直した。2026-11-23 の簡素化（[[feedback-docs-one-fact-one-place]]）で次の形に改めた。

- 全体図と PyTorch との対応表。「このページで分かること」と、節の案内の表（TOC と同じ）は置かない
- `!!! abstract "要点"` の箱は置かない。節の頭の 1 文は、図だけでは文脈が分からないときに限る
- 図・表で示したことを本文で言い直さない（「次の図は…」「実線は…」も書かず、凡例は図のラベルに入れる）。本文は図・表・コード・docstring のどれにも無い事実だけ
- 仕組みは図で見せる（mermaid の flowchart / sequenceDiagram / stateDiagram、mermaid で描けないものはテーマ色に従う inline SVG。CSS は `docs/stylesheets/extra.css` の `.mujou-diagram`）
- 比べるものは表にする（Port と MultiPort、mujou.run と testing.run など）
- 設計の理由は `??? question "〜のはなぜ？"` に畳む（4 行まで。docstring にあるならリンクだけ）
- 規則は一覧表。「避ける / こう書く」のタブは自然に書いてしまう誤りがあるときだけ、1 節に 1 組まで
- 完全なプログラムは examples/ から snippet で取り込み、短い断片だけ inline で書く

**Why:** 文字だけの説明は、非同期・最新値・dt のような直感に反する仕組みが伝わらない。

**How to apply:** docs/ の topics やチュートリアルを書く・直すときも同じ部品を使う。図の中の事実（hook の順、メッセージの中身など）は docstring と specs/ で確かめてから描く。描画はヘッドレスブラウザが無い環境では確かめられないので、`just docs` で見てもらうよう伝える。サイトの構成は [[project-docs-site]]、日本語の文体は [[feedback-japanese-style]]。
