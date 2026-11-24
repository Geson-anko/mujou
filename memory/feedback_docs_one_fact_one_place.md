---
name: feedback-docs-one-fact-one-place
description: docs/ は 1 つの事実を 1 か所に置き、ほかはリンクする。API の事実は docstring が持ち、docs は autorefs でリンクする。ページごとに本文行の上限を置く（2026-11-23）
metadata:
  type: feedback
---

2026-11-23、ユーザーから docs の簡素化を求められた。長い文章は人にも AI にもトークンのコストになる。同じ事実をいくつものページで言い直していたので、本文を約半分に削った（PR #126）。

- 事実ごとに持ち主のページ・節を 1 つだけ決める。ほかのページは書き直さずにリンクする。言い換えてよいのはリンクの文の 1 句だけで、理由・条件・例外は書き写さない
- API の docstring が持ち主の事実（シグネチャ、引数、既定値、送出する例外、メソッドごとのふるまい、状態の木の形）は docs に書かない。名前を mkdocs-autorefs でリンクする（`[`Node`][mujou.Node]`、`[mujou.Run.wait_for][]`）。リンクは節の中で最初に出たときだけ。`api/*.md#...` への相対リンクは使わない
- docs だけが持つ事実は、仕組みの図、設計の理由、選び方・決め方、落とし穴、測った数値、可視化ページの UI。docs が API 側へ寄せたい事実で docstring に無いものは、先に docstring に足してからリンクする
- 図は 1 つの仕組みにサイト全体で 1 つ。example の名前付き区間もサイト全体で 1 回だけ取り込む
- 見出しに番号を付けず、リンクされうる見出しには en と ja で同じ ASCII の明示 id `{#id}` を付ける
- 本文行（空行と、コードの囲み・SVG の中を除いた行）に、ページごとの上限を置く（ja。en も同じ構成）

| ページ | 上限 | ページ | 上限 |
| --- | --- | --- | --- |
| index | 50 | topics/ports | 60 |
| install | 35 | topics/groups | 25 |
| examples | 35 | topics/host-io | 45 |
| concepts | 85 | topics/time | 50 |
| tutorial/1-run | 30 | topics/simulation | 55 |
| tutorial/2-node | 55 | topics/threads-and-devices | 70 |
| tutorial/3-test | 25 | topics/errors | 30 |
| tutorial/4-learn-in-a-node | 35 | topics/update-rules | 35 |
| tutorial/5-learn-with-pc | 40 | topics/visualize | 45 |
| tutorial/6-save-and-resume | 55 | topics/pc-concepts | 140 |
| | | topics/pc | 150 |
| | | topics/pc-benchmarks | 60 |

数え方:

```bash
awk '/^[[:space:]]*```/{f=!f;next} /<svg/{s=1} /<\/svg>/{s=0;next} !f&&!s&&NF' docs/ja/<page>.md | wc -l
```

**Why:** 同じ事実が何か所にもあると、読む量が増え、直すときに食い違う。API の事実は docstring と docs の 2 か所に書くと必ずずれる。

**How to apply:** docs/ のページを書く・直すとき、まず事実の持ち主がどこかを決める。上限を超えたら、持ち主でない事実が残っていないかを先に疑う。図表の基準は [[feedback-docs-teaching-quality]]、改稿の分担は [[feedback-docs-structure-and-haiku-prose]]。
