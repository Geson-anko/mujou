---
name: project-run-node-and-write-time
description: 2026-10-17 の API の手直し。Run.write が Timestamp を返す、mujou.run / Compose / mujou.testing.run がノードを受け付け、ノード 1 つは Compose(node) としてパス "0" で走る
metadata:
  type: project
---

2026-10-17、サンプルを書いた担当の「API の気づき」から orchestrator が推奨案として決めた（ユーザー確認待ち）。

- `Run.write(values) -> Timestamp`: 書いた値に付いた時刻（その実行のタイマー）。1 回の write の全 Port に同じ時刻。ノードの `recv_with_time` と状態の木の `/in` の時刻と一致する。`read` の時刻は送った時刻で別物。空の write も時刻を返す
- `mujou.run(*networks: Node | Network)` / `mujou.testing.run` / `Compose(*networks: Node | Network, **named: Node | Network)`: ノードを受け付け、ネットワークと混ぜてよい。ネットワーク 1 つだけは包まずそのまま、それ以外は `Compose(*networks)`
- ノード 1 つのパスは `"0"`（Port は `"0.prev"`）。`""` にすると Port のパスが `".prev"` になり木のキーが崩れる。`run(a, b) == run(Compose(a, b))` の規則がノード 1 つでもそのまま成り立つ
- ノードの 2 回目の run は、新しい Compose の build が構築済みの印に当たって `RuntimeError`（"already built"）。断られた run(node) は Compose の構築を取り消すので、どの形でも走らせ直せる

**Why:** ノード 1 つを試すのに包むネットワークを書かせない。write の時刻をホストが受け取り、ノードが見た時刻と突き合わせられるようにする。

**How to apply:** 引数名は `*networks` のまま（ノードも入る）。改名するなら `mujou.testing.run` と揃える。仕様は `specs/network.md`。
