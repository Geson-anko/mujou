---
name: project-run-node-and-write-time
description: 2026-10-17 の API の手直し。Run.write が Timestamp を返す、mujou.run(component, *components) / Compose(*components) がノードを受け付け、1 つだけならノードも包まずに根（パス ""）にする（ユーザーの決定）
metadata:
  type: project
---

2026-10-17、サンプルを書いた担当の「API の気づき」から orchestrator が推奨案として決め、パスと引数名はユーザーが決めた。

- `Run.write(values) -> Timestamp`: 書いた値に付いた時刻（その実行のタイマー）。1 回の write の全 Port に同じ時刻。ノードの `recv_with_time` と状態の木の `/in` の時刻と一致する。`read` の時刻は送った時刻で別物。空の write も時刻を返す
- **ユーザーの決定:** `mujou.run(component, *components, timer=..., inputs=..., state=..., pause_timeout=...)`、`mujou.testing.run` も同じ形、`Compose(*components, **named)`。引数なしは型検査でも実行時でもエラー
- **ユーザーの決定（理由は「その方がシンプル」）:** コンポーネントが 1 つだけならパスの先頭に位置を足さない。ネットワーク 1 つはそのまま、ノード 1 つも包まずに根にする（ノードのパス `""`、Port は `"prev"`。根が空のパスなら区切りの `.` を入れない）。2 つ以上は `Compose(component, *components)` と同じで `"0.l1"` など。`Compose` は 1 つでも位置を付ける
- ノード 1 つの状態の木: `{"nodes": {"": ...}, "slots": {"prev/in": ..., "next/out": ...}, "last_update_ns": {"": ...}, "timer": ...}`。pickle して復元すれば往復する
- 根のノードはエラーとログでクラス名だけで呼ぶ（`"Leaky has already run; ..."`、`"Leaky declares the port ..."`）。ランタイム（Rust）に渡す名前は、パスが空のノードだけクラス名にする（`"node 'Leaky' raised in on_update"`）
- 2 回目の run は `"<クラス名> has already run; create a new instance to run again"`（ネットワークとノードでそろえる。mujou.run に渡したものにはパスを付けず、利用者の木の中ならそのパスを付ける）。構築済みの印は木ごとの記録（mujou.run が自分で構築したか、走ったか）を共有する。断られた run は mujou.run が構築した木（RunCompose かノードの根）を取り消すので走らせ直せる

**Why:** ノード 1 つを試すのに包むネットワークを書かせない。write の時刻をホストが受け取り、ノードが見た時刻と突き合わせられるようにする。

**How to apply:** パスの規則は `network._child`（根が空なら `.` を入れない）。仕様は `specs/network.md`。
