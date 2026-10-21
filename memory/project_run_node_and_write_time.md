---
name: project-run-node-and-write-time
description: mujou.run(root) はルートを 1 つだけ受け取り、ノード単体もそのままルートにする（パス ""）。Run.write は書いた時刻の Timestamp を返す
metadata:
  type: project
---

- **`mujou.run(root, *, timer=, inputs=, state=, workers=)`:** ルートはノードかネットワーク 1 つ。並べるときは `Group(**named)` に入れる（[[project-group-and-typing]]）。root を 1 つにしたのは R4 の推奨案で、ユーザーの確認待ち（[[project-restructure-2026-10-20]]）。`mujou.testing.run` も同じ形
- **ノード単体もルートにできる。** ノードのパスは `""`、ポートの label は `"Leaky.prev"`（2026-10-17、ユーザーの決定「その方がシンプル」）。エラーとログでは根のノードをクラス名で呼ぶ（`tree.name`）
- **`Run.write(values) -> Timestamp`:** 書いた値に付いた時刻（その実行のタイマー）。1 回の write の全ポートに同じ時刻が付き、ノードの `recv_with_time` と状態の木の `"in"` の時刻と一致する。`read_with_time` の時刻は送った時刻で別物

**Why:** ノード 1 つを試すのに包むネットワークを書かせない。パス（状態の木のキー）が、実行のしかたで変わらないようにする。write の時刻をホストが受け取り、ノードが見た時刻と突き合わせられるようにする。

**How to apply:** 仕様は `specs/network.md` の「パスと label」「ルートと Group」。
