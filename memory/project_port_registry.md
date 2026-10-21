---
name: project-port-registry
description: インスタンスの __dict__ を子とポートの唯一のレジストリにし、run の段 1 で tree.walk が代入の順に読む。Component は __setattr__ もレジストリも持たない（R7、推奨案で進めユーザーの確認待ち）
metadata:
  type: project
---

ノードとネットワークは、子（Node / Network）とポートをインスタンスの属性に代入するだけで宣言する。`mujou.run` の段 1 で `tree.walk` が `vars()` を代入の順に読み、ノードではポートを、ネットワークでは子を拾う（ネットワークの Port はただの参照として読み飛ばす）。`Component` は `__setattr__` / `__delattr__` もレジストリも持たず、クラス定義の時点の検査（K1: クラス属性のポート・ノード・ネットワーク、`__slots__`）だけを持つ。詳細は `specs/network.md` の「木の読み方」。

R7 の推奨案で進めた（以前の決定は代入時に `__setattr__` で登録する形）。ユーザーの確認待ち（[[project-restructure-2026-10-20]]）。

**Why:** 代入時に登録すると、属性の順序と所属を `__dict__` とレジストリの 2 か所で持ち、代入と `del` のたびに照らし合わせることになる。1 か所にすると、番兵、解除、`copy.copy` 対策、property の除外、`super().__init__()` の前の代入を拒む規則、構築後の変更を拒む状態が消え、`on_update` の `self.y = ...` も `__setattr__` を通らない。dataclass の Network もそのまま動く。失うのは誤りを代入の行で知らせられることだけで、代わりに段 1 で `パス.属性名` を名指しし（K2、K3、K6）、その run は同じインスタンスで直せる。

**How to apply:** 接続は `a.next.connect(b.prev)` と書く。ポートを list や dict に入れない（K6）。走った後に代入しても何も起きない。
