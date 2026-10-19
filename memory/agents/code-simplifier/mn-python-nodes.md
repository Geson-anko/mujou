# code-simplifier: M:N PR 2（Python のノードをワーカープールに載せる）の仕上げ

対象: `git diff d1ca90f..HEAD`（ブランチ `feature/2026-10-18/mn-python-nodes`）。2026-10-19。

## 残りの指摘への対応（orchestrator の裁定どおり）

- code S4: `tests/mujou/_core/test_native_runtime.py` の `TestFrameworkHooks` に `_before_hook` を記録する `BeforeHook` を足し、bindings の層で契約を固定した
  - `test_before_hook_precedes_every_public_hook_but_state_dict`: 専用スレッド・復元つきで start → step → capture → shutdown の hook の列を丸ごと比べる。`on_thread_start`・`on_start`・`load_state_dict`・`on_update`・`on_pause`・`on_resume`・`on_stop` の前に呼び、`state_dict`（`_capture_state`）と `_after_update` の前には呼ばず、`_on_enter` → `_before_hook` の順
  - `test_an_error_in_before_hook_is_reported_as_the_hook_and_skips_it`: `on_thread_start`〜`on_resume` の 6 つで、失敗をその hook の名前で報告し、その hook を呼ばない
- code N9 / algo R1: `call_as`（`crates/mujou-python/src/runtime.rs`）で、`_before_hook` の失敗を `on_exception` / `on_stop` では `self.error` でログに出してから本体を呼ぶようにした（`enter` の失敗と同じ契約）。Required の hook では今どおりその hook の失敗。テスト `test_an_error_in_before_hook_still_runs_the_clean_up_hook`。specs/native-core.md と `Node::_before_hook` の doc comment にも書いた
- code N8 / algo R2: `_before_hook` は `Context.run` の外で呼ぶ、という例外を specs/native-core.md・specs/torch.md・`Node` と `Node::_before_hook` の doc comment・`_native.pyi`（`just stub`）に書いた。`contextvars` に依る設定を `_before_hook` に足すと効かない、という注意も
- code N10: docs（en/ja の torch.md）の「hook の前に毎回」2 か所に `state_dict` を除くことを足した
- code N11: docs（en/ja の torch.md）の CUDA の同期を、ワーカーが待つのは `on_update` の後と復元の後だけで、状態の取得の同期はワーカーでない mujou のスレッド（終了後はホストのスレッド）で行う、と 2 項目に分けた。利用者向け docs に「制御スレッド」という語は無いので、その語は使っていない
- code N12: docs/ja/user-guide/torch.md の体言止め 2 か所を文にした
- code N13: `test_a_node_whose_setup_failed_cleans_up_as_itself` に、private の override で失敗を起こす理由のコメントを足した
- algo R4: skill locality-and-async の 2 か所を「CUDA のノードに切り替わるたび」に（skill edit-dot-claude の手順で /tmp 経由）

## 記録だけ残すもの（algo R3、直さない）

`mujou.Node._on_enter` は `enter_node(self)` の後で `torch.cuda.set_device` と `set_stream` を行う。このどちらかが失敗すると、続く BestEffort の `on_exception` / `on_stop` は、Port の所有と勾配モードは正しいが、CUDA の device / stream はワーカーが前に走らせたノードのもののまま走りうる。CUDA の設定の失敗は稀で、終了処理の中だけのことなので、orchestrator の裁定で直さない。直すなら、`_on_enter` の失敗時に CUDA の現在のストリームを既定に戻す、などが候補。

## 簡素化

- `PyNode::method(py, hook)` を抽出した。`call_as` の中の closure と `state_dict` の `match` が、同じ「起動時に引いたメソッド、無ければ名前で引く」を 2 回書いていた。1 か所にまとめ、`state_dict` は `self.method(py, "state_dict").and_then(|m| m.call0())` の 1 行になった。振る舞いは同じ（`getattr` + `call0` と `call_method0` は等価）
- ほかの PR 2 の差分（`node.py` の `checked_dedicated`・`_on_enter`・`_before_hook`、`hooks.py`、`run.py` の `workers` の検査、`session.py`）は整理点なし。どれも 1 つの関心事を最小の形で書いていて、`hooks.py` の `inside` と `node` は `running_hook`（制御スレッドはノードを持たずに hook の中に入る）があるので 1 つにできない
- 大きな書き換えはしていない

## 検証

- `just run`: green（詳細は報告）
