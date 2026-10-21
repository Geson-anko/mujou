# host-read-carried: 計画外の判断ログ

2026-10-20、ユーザーの決定（案 (a)）でホストが付随データを持つ Port を読むと値だけを返すようにした作業の記録。

- `Port` の docstring（付随データの段落）に「ホストは `send_carry` を持つ Port の値だけを読み、`recv_carry` を持つ Port には書けない」と 1 文足した。付随データの説明がまとまっている場所なので、ホストからの見え方もそこに書いた
- `pc.Layer.next` を読むテストは `TestScalar` に 1 行足した。平衡点で `run.read(layer.next)["h"]` が順伝播の値 $w_1 x$ に一致することをオラクルにした（制約 $h = f(h_{\mathrm{in}})$ が平衡点で満たされる）
- `tests/mujou/_core/test_node_typing.py` の付随データのテストは、`run.read(relay.next)` の型が `torch.Tensor` になることを `assert_type` で確かめる形に変えた（pyright の ignore を外した）
- `examples/carry.py` の `Report`（ID を普通のメッセージで送る）はそのまま残した。付随データそのものをホストに見せるには今も普通のメッセージで送る必要があるため
