---
name: feedback-direction-names
description: 向きは属性名の prev / next（「前」「次」）で表し、クラスは Port / MultiPort にする（R2）。Up / Down、Bottom / Top、upper / lower は重力方向の直感と食い違うので使わない（2026-10-20 のユーザーの指示）
metadata:
  type: feedback
---

2026-10-20、`mujou.pc` のメッセージ名に `pc.Up` / `pc.Down` を提案したところ、ユーザーが「Prev, Next と名前を合わせるべき。Up と Down、Bottom と Top は重力方向の直感性と相対するのでこのプロジェクトでは避ける」と指示した。

向きはクラスでは表さず、属性名の慣習 `prev` / `next` で表す（`self.prev = Port(...)`、`a.next.connect(b.prev)`）。接続点のクラスは相手の数で分けた `Port` / `MultiPort`。`PrevPort` / `NextPort` を置き換えたのは R2 で、2026-10-21 にユーザーが承認した（[[project-restructure-2026-10-20]]）。

**Why:** 上下の比喩は図の描き方次第で逆に読める。`PrevPort` / `NextPort` の違いは相手の数の既定値だけだったので、相手の数をクラスで表し、向きの語彙は属性名に残した。

**How to apply:** API・docstring・仕様書で入力側／出力側を指すときは prev / next（「前」「次」）を使う。数式の説明で「上の層」と書くのも避け、「次の層」「前の層」と書く。
