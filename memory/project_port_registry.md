---
name: project-port-registry
description: 2026-10-17 のユーザーの決定。接続は Port 同士（a.next.connect(b.prev)）だけにし、Component.connect と「PrevPort / NextPort は直接の属性に 1 つずつまで」の制約を廃止。Port と子のノード・ネットワークは nn.Module と同じく __setattr__ で代入時に登録し、build は vars() を探索しない
metadata:
  type: project
---

2026-10-17 のユーザーの決定。詳細は `specs/network.md` の「接続」と「属性の登録」。

- **接続は `Port.connect` だけ。** `Component.connect`（`a.connect(b)` で a の NextPort と b の PrevPort をつなぐ省略形）と、それを支える「PrevPort / NextPort は直接の属性として 1 つずつまで」の制約・build の検査を消した。`PrevPort` / `NextPort` のクラスと相手の数の既定（PrevPort は 1、NextPort は複数）は残す。`prev` / `next` は property にせず、慣習の属性名として docs で説明する。`Port.__call__` は足さない
- **Port と子は代入時に登録する（nn.Module の `_modules` と同じ形）。** `Component.__init__` が `_ports`（属性名 → Port、代入順）と `_children`（属性名 → Node / Network）を用意し、`__setattr__` / `__delattr__` が登録・解除する。build はこの登録だけをたどる
- `super().__init__()` の前に Port や子を代入すると `AttributeError`。ノードにノード・ネットワークを代入すると代入時に `ValueError`（以前は build 時）。構築済みの木に Port・子を代入・削除すると `RuntimeError`

**Why:** ノードの入出力は複数ありうるので、コンポーネント同士をつなぐ経路は曖昧になる。接続をポート同士だけにすれば、どの経路かがいつも書いた行から読める。ポートは代入時に登録すれば、属性の探索（`vars()`）が要らなくなる（nn.Module と同じ）。

**How to apply:**

- 例・docs・テストでは `a.next.connect(b.prev)` と書く。ネットワークは子の Port を `self.prev = self.l1.prev` のように公開する
- `Component.__setattr__` は `on_update` の `self.x = tensor` でも通るので、普通の値の分岐を増やさない。`isinstance` は union（`X | Y`、呼ぶたびに作られて数倍遅い）ではなく型のタプルに対して行う
- dataclass のネットワークは、生成された `__init__` が `super().__init__()` を呼ばないので、子を代入すると `AttributeError` になる（nn.Module と同じ）

関連: [[project-network-ports]]
