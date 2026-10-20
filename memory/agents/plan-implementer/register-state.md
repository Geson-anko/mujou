# register-state（plan-implementer の判断ログ）

2026-10-20、`Node.register_state(name)` の実装。

- 登録表 `_mujou_state_names` の値を bool（`register_state` で登録したか）にした。`__setattr__` は普通の値の代入で `names.get(name) is False` を 1 回引くだけ（以前の `_forget_state_name` と同じ回数）
- `register_state` は property の検査を `vars(self)` の検査より先に行う。`device` などは `_device` に値を置くので、逆順だと AttributeError になりテストの期待（ValueError）と合わない
- 自動登録の名前を `register_state` すると値が True になり、以後は Tensor でない値を代入しても外れない（順序はそのまま）
- 既定の `state_dict` は登録した普通の値をそのまま返す（コピーはフレームワークの `copy.deepcopy` が取る）。`load_state_dict` は `copy.deepcopy` を `setattr` する
- docs の state.md では「数値は 0 次元テンソルにして自動で集める」の段落を `register_state` の説明に置き換えた。writing-a-node.md の「数値は状態に集まらない」（自動では集まらない、の意）は計画外なので触れていない
