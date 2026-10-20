---
name: feedback-direction-names
description: 向きの名前は Prev / Next に合わせる。Up / Down、Bottom / Top は重力方向の直感と食い違うので使わない（2026-10-20 のユーザーの指示）
metadata:
  type: feedback
---

2026-10-20、`mujou.pc` のメッセージ名に `pc.Up` / `pc.Down` を提案したところ、ユーザーが「Prev, Next と名前を合わせるべき。Up と Down、Bottom と Top は重力方向の直感性と相対するのでこのプロジェクトでは避ける」と指示した。

**Why:** ネットワークの向きは Port の `PrevPort` / `NextPort` で表しており、上下の比喩は図の描き方次第で逆に読める。

**How to apply:** API・docstring・仕様書で入力側／出力側を指すときは prev / next（「前」「次」）を使う。例: Port は `PrevPort` / `NextPort`、付随データの中継は `prev` から `next` へ。数式の説明で「上の層」と書くのも避け、「次の層」「前の層」と書く。
