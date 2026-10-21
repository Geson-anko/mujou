---
name: project-carry
description: 2026-10-20 に実装したコアの carry（付随データのゼロコピー中継）の API の決定と理由。Port の send_carry= / recv_carry=、carried()、状態の木の "carries"、ホストは付随データを読み書きしない（読むと値だけ返る）
metadata:
  type: project
---

2026-10-20 に実装した（要件は `specs/pc-module.md`「コアの変更: carry」、仕様は `specs/network.md`「付随データ」と `specs/messages.md`「付随データ」）。計画の確認事項は orchestrator が推奨案で決定した。

- **宣言は向きごとに分ける:** `recv_carry=spec`（相手 1 つの Port だけ）と `send_carry=`。`send_carry` に spec を渡すと作る側（`send(x, carry=...)`、コピー）、同じノードの Port を渡すと中継する側（その Port で最後に `recv` したメッセージの付随データを `Arc` の複製で添える）。`carry=` 1 つの案は、作る側（送る向きの spec）と両向きに持つ Port を書けないので採らなかった
- **中継元は同じノードの Port:** 局所性のため。Port は `__init__` で所有ノードを知らないので `graph.build_tree` で検査する（`ValueError`）
- **状態の木に入れる:** 付随データを運ぶスロットの項目は `(value, time_ns, sent, carried)`（運ばないスロットは 3 つ組のまま）、木に `"carries": {Port のパス: carried}`。入れないと、`recv` の後に中継元が更新された時点で止めた木から再開したとき、`recv` より先に `send` するノードの中継がずれる
- **ホストは付随データを読み書きしない:** `send_carry` を持つ Port の `read` / `read_with_time` / `subscribe` / `wait_for` は、付随データを捨てて値だけを返す（2026-10-20、ユーザーの決定。algorithm-reviewer の K4 の案 (a)）。実装した当初は `ValueError` で断り、型でも `Port[S, Any, Any, Never, Any]` で弾いていたが、それでは `pc.Layer.next` が読めず、隠れ層の `h` を観測できなかった。値の意味は付随データの有無で変わらないので、捨てても読み違えない。付随データを読む `Run.read_carried` は足さない。`recv_carry` を持つ Port への `write` / `inputs=` は `ValueError` のまま（書き込みは値だけを渡し、付随データを作る口が無い）
- **型引数は `Port[S, R, M, CS, CR]`:** 既定は `Never`。不変なので、任意の Port を受ける注釈は `Port[Any, Any, Any, Any, Any]` にする（`Port[Any, Any, Any]` は付随データを持つ Port を受けない）
- **コア:** `Message` は `Vec<Arc<Buffer>>` と `carry_from`。`Schema::new(values, carry)`、`Outlet::send_with(target, relay, fill)`（`E: From<SlotError>`）、`Slot::new_carry`（復元用）。中継したバッファは元の書き手のプールへ戻る

**Why:** 活動とターゲットを別スロットで運ぶと、読み手が別々の瞬間に読んで別のサンプルの組になりうる（PC の `Output`）。1 つのメッセージに入れれば構造的に組み違えない。

**How to apply:** 経路に沿って運ぶデータは carry に載せる。`initial_carry=`（付随データの初期値）は要るまで足さない（初期値はゼロ）。
