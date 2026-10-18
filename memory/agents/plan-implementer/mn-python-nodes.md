# M:N スケジューリング PR 2（Python のノードをワーカーに載せる）の判断ログ

計画書: `memory/agents/implementation-planner/mn-scheduler.md` の PR 2 と「ユーザーの決定（2026-10-18）」。仕様テストは spec-test-author（`memory/agents/spec-test-author/mn-python-nodes.md`、b7f30b9）。

## 実装したもの

- bindings（`crates/mujou-python/src/runtime.rs`）: `Runtime.start(workers=, dedicated=)`（`dedicated` 省略で全ノードがワーカー、`workers=0` は `ValueError`）。`PyNode` に `context`（ノードごとの `contextvars.Context`）と `dedicated`。`enter` で `_on_enter` を呼び、エラーは次の公開 hook の名前で報告。`start` は `dedicated` のときだけ `on_thread_start`。`public()` から `_on_thread_start` を消した
- `node.py`: `_on_thread_start` → `_on_enter`。`dedicated_thread` 引数・property・`checked_dedicated`。`thread_ident` / `end_thread` / `_thread_ident` を削除
- `hooks.py`: `enter_node(node)` / `current_node()`。`binding.py` は `current_node() is self._owner` で所有を判定
- `session.py` / `run.py`: `workers` と `dedicated` を native へ。`mujou.run(workers=None)` の検査
- docs（en/ja）、examples のコメント、memory、skill `locality-and-async`。specs はサブエージェントに委譲して確認した

## 計画にない判断（理由つき）

1. **ノードの `contextvars.Context` は空の `Context()` にした**（ホストの文脈の `copy_context()` ではない）。今まで Rust が立てたスレッドの文脈も空だったので、振る舞いを変えない。ホストの `decimal` の文脈などを持ち込まない
2. **`state_dict`（`_capture_state`）は bindings で文脈の外から直接呼ぶ**。計画どおり。`call()` は必ず `Context.run` を通すので、`state_dict` だけ別の経路にした
3. **`_on_enter` も `Methods` で 1 回だけ引く**。切り替えのたびに `getattr` しないため。`start` が終わるまでは名前で引く（ほかの hook と同じ）
4. **ノード専用の CUDA ストリームは最初の `_on_enter` で作る**（`self._stream is None` のとき）。ノードは 1 回しか走らないので、実行ごとに作り直す必要は無い
5. **`super().__init__` を呼ばないサブクラス向けの generator の生成は `_on_enter` に残した**（今まで `_on_thread_start` にあった）。`is None` の判定 1 回で済む
6. **`checked_dedicated` が予約名の検査と `on_thread_start` の override の検査を兼ねる**。計画は別の関数も挙げていたが、どちらも「そのノードが専用スレッドか」を決めるときの検査なので 1 つにした。override はクラスで判定する（`inspect.getattr_static(type(node), ...)`）。インスタンス属性での置き換えは見ない（計画も「クラスが override していたら」）
7. **`workers` の検査は `start_run` の `in_hook` の検査の直後、構築の前**。何も起動せず、木も作らない。`mujou.testing.run` は `workers=None` を渡す（`start_run` の引数は必須にした）
8. **`current_node` の束縛は外さない**。ワーカーも専用スレッドも実行の終わりに抜け、スレッドの Python の状態ごと消える。ホストと制御スレッドは `enter_node` を呼ばないので、`current_node()` は常に `None`
9. **`docs/*/index.md` の「自分のスレッドで更新」は「自分の周期で更新」に変えた**。ja の index はユーザーが直した文体の基準なので、語を 1 つ替えるだけにした
10. GIL ありの 3.14（Homebrew）でも `tests/mujou` と `tests/test_examples.py` が通ることを手元で確かめた（551 passed）

## 確認事項

なし（ユーザーの決定の範囲で実装した）。
