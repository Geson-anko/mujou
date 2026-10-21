---
name: project-network-ports
description: ノード間の接続の決定。接続点は Port（受け取る相手は高々 1、送るだけなら何本でも）と MultiPort（同じ種類の相手を束ねる）。向きを持たず send と recv の spec の組。接続は相手 → Channel の dict。Network に代入した Port はただの参照。集約は更新手法の責任
metadata:
  type: project
---

詳細は `specs/network.md`。Port / MultiPort は R2 の推奨案で進めた（ユーザーの確認待ち、[[project-restructure-2026-10-20]]）。

- **構成要素は ノード・ネットワーク・Port / MultiPort・Channel。** ノードは非同期処理の単位で、ポートを宣言する。ネットワークは組み立て専用の入れ物で hook を持たず、実行時の単位はノードだけ
- **ポートは向きを持たない。** send の spec と recv の spec の組（キーワード専用、片方だけでもよい）。`connect` は互いに逆向きに一致するときだけ繋ぐ（K10）。上流・下流はフレームワークで決めない
- **相手の数はクラスで表す。** `Port.recv()` は 1 つの値を返すので、受け取る Port の相手は高々 1 つ（K11。両端を見る）。送るだけの Port は何本でも繋げ、`send` は全員に配る（fan-out）。`MultiPort` は同じ種類の相手を束ね、`recv()` は `{Channel: 値}` を connect の順で返し、`send_to` で 1 本に送る（outbox には書かない）。役割の違う入力は Port を分ける
- **接続は各ポートの「相手 → Channel」の dict。** 同じ相手にもう一度繋ぐと同じ Channel を返す。自己接続も特別扱いしない。循環してよい。ノード同士をつなぐ `a.connect(b)` は持たない
- **Network に代入した Port はただの参照**で、内側のポートを `self.prev = self.l1.prev` の形で公開する。木の走査は読み飛ばす
- **ポートは実行時の状態を持たない。** 実行中の読み書きは Driver の `wires` が持つ `Wire` が担い、所有者の検査は lookup 1 回（K12）
- **`initial`** は送るポートが最初に send するまで相手に見える値で、公開の読み取り専用 property
- **ホストは未接続で受け取る `Port` に書き、送るポートの outbox を読む**（[[project-host-io]]）

**Why:** 以前は `PrevPort` / `NextPort` / `Port(multi=)` と相手の数を表す軸が 3 つあり、違いは既定値だけだった。受け取る Port を 1 つの相手に限るのは `recv()` の戻り値の形の帰結で、条件付きの例外ではない。接続を相手をキーにした dict で持つと、二重接続と自己接続で channel の番号が壊れる場合が表せなくなり、状態の木の `"to"` も相手のパスで一意に引ける。

**How to apply:** 複数の相手からのメッセージの集約（PC-ALM の $\sum_k W_k^\top \varepsilon_k$ など）をフレームワークに組み込まない。受け取るノードの更新手法が行う（[[project-pluggable-update-rule]]）。入れ子は組み立てだけの構造で、非同期の境界を変えない（[[project-compute-granularity]]）。

関連: [[project-message-types]], [[project-port-registry]]
