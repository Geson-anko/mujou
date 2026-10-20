# Node.register_state レビュー

対象: `0a7a2b9 feat(node): register any value as state`（`git diff chore/2026-10-20/examples-as-guides...HEAD`）

## verdict: approve

must-fix なし。仕様（specs/pc-module.md「コアの変更: Node.register_state」、specs/nodes.md）の要件（任意の値の登録、非 Tensor の代入で外れない、del で外れる、load は deepcopy の代入、property は ValueError、未代入は AttributeError、`__setattr__` の速さ、浅いコピーで登録表を共有しない）はすべて満たしている。

## must-fix

なし

## should-fix

1. `src/mujou/_core/node.py` load_state_dict / `docs/*/user-guide/state.md` / `specs/nodes.md`: on_start より後（on_update の中など）で初めて `register_state` した値は、新しいインスタンスに復元できない。復元先がまだ登録していないので、`ValueError: ... not tensors and not registered ['k']` になる（再現済み）。Tensor には「on_start の後で初めて代入したものも復元できる」という救済があるのに、登録した値には無い。この非対称を docs と spec に書いていない。最低でも「`__init__` か `on_start` で登録する」を docs（en/ja）と nodes.md に書く。確信度: 高（挙動）／中（直し方）。深刻度: 中
2. `src/mujou/_core/node.py` load_state_dict のエラーメッセージ `"the state does not match the node's tensors and modules: "`: 登録した普通の値が欠けたときも同じ文言になる（`missing ['log']`）。今は tensors と modules に限らないので文言が実態とずれる。docstring の 1 文目「Copy state ... into the registered attributes in place」も同様（次の文で補足はある）。確信度: 高。深刻度: 低
3. `tests/mujou/_core/test_node.py` TestRegisterState: docstring と spec が約束する 2 点にテストが無い。(a) `register_state` した Tensor が load でその場で復元される（`is` が保たれる）こと（nodes.md「Tensor の名前を登録してもよく、その場の更新で復元される点は変わらない」）。(b) 自動で集めた名前を `register_state` すると、その後 Tensor でない値を代入しても外れないこと（`test_registering_twice_keeps_the_order` は順序しか見ていない）。確信度: 高。深刻度: 低

## nit

1. `node.py` load_state_dict: 登録名が load の時点で Tensor を持ち、木の値が Tensor でない（保存時は `None` だった）とき、`TypeError: the state's 'last' is a NoneType, not a tensor` になる（再現済み）。spec どおり（その時点で Tensor なら copy_）なので仕様違反ではない。ただ「何を代入しても状態に残る」という登録の性質と組み合わせると起こりうるので、docstring に一言あってもよい。確信度: 中
2. `node.py` register_state: `register_state("_mujou_state_names")` が通り、登録表そのものが状態に入る（再現済み。load で登録表を差し替える）。実害はまず無い。弾くなら `_STATE_NAMES` を拒否する。Port や子の Node を登録すると pickle できない木になる点も同様。確信度: 低
3. `docs/*/user-guide/state.md` Formats の箇条「If you override `state_dict` to put objects of your own classes in the tree...」は、`register_state` でも独自クラスが木に入る。「override するか register_state で」としてもよい
4. 判断依頼 1（0 次元テンソルの段落の削除）: 妥当。`register_state` の方が単純で、冒頭の「Tensor と nn.Module は自動で集まる」から 0 次元テンソルが集まることは読み取れる。併記するなら、GPU 上で計算した値（損失の移動平均など）は `.item()` の同期を避けるため 0 次元テンソルのまま置けば自動で集まる、という 1 文に限る。必須ではない
5. 判断依頼 2（writing-a-node.md 125 行目）: 段落の主題は「設定値」で、設定値は状態に入れるべきではないので、今の文は正しい。足すなら「Numbers are not collected into the state automatically」「数値は自動では状態に集まらない」と 1 語足す程度で、直前の段落が state.md へリンクしている。必須ではない

## 規約（判断依頼 3）

- カプセル化: 公開は `register_state` だけ。補助関数（`_state_names` など）と登録表の属性は `_` prefix。問題なし
- docstring とコメントは英語、specs は日本語、docs は en/ja の両方を同じ内容で更新している
- テスト: `TestRegisterState` に集約し、`tests/mujou/_core/test_node.py`（`src/mujou/_core/node.py` のミラー）に置いている。モックや private の直接テストは無い（`_x` の参照は既存）。乱数に依存しない。復元テスト（`TestRestore`）では、止めずに続けた実行との一致をオラクルにしており、`elapsed` も `> 0` 付きで比べているので、復元しなければ失敗する
- 成果物汚染（`</content>` など）: なし

## 検証結果

- just run: pass（586 passed, 20 skipped、pyright 0 errors、mkdocs build --strict 成功）。作業ツリーは変化なし
- 境界条件の再現スクリプト: /tmp/claude-1001/-data-geson-mujou/f26146d6-7c30-4c9b-99a6-ce3b2e433458/scratchpad/edge.py
