---
name: project-carry
description: コアの carry（付随データのゼロコピー中継）は廃止した。PC は目標 y・mask・time を Activity の普通のフィールドで運び、各ノードが次のメッセージに写す（R1、2026-10-21 にユーザーが承認）
metadata:
  type: project
---

コアは付随データ（carry）を持たない。Port の `send_carry=` / `recv_carry=`、`carried()`、Schema と Message の carry、状態の木の `"carries"` は無い。`mujou.pc` は目標 `y`、`mask`、書き込みの時刻 `time` を `pc.Activity` のフィールドとして運び、各ノードが受け取った値をそのまま次のメッセージに写す（`specs/pc.md` の「メッセージ」）。

R1 として 2026-10-21 にユーザーが承認した（[[project-restructure-2026-10-20]]）。

**Why:** carry の使い手は PC だけだった。carry を持った理由（活動と目標を 1 つのメッセージに入れて組み違えを防ぐ）は、同じメッセージのフィールドでも成り立つ。pc の全ノードは同じ update の中で recv してから send するので、中継元が後で更新されて復元がずれるという carry の理由も、フィールドなら隠れた状態が無いので起きない。廃止で Port の型引数・引数・分岐、Schema と Message の carry、状態の木の項目、carry のガードが消えた。失うのはゼロコピーで、1 ホップあたり y・mask・time をコピーする（MNIST の悪化は 1% 未満、`specs/research/benchmarks.md`）。

**How to apply:** 経路に沿って運びたいデータは、メッセージの普通のフィールドにする。コアに中継の仕組みを足さない。
