# carry の実装: 判断ログと計画からの逸脱

> 2026-10-20 plan-implementer。計画は `memory/agents/implementation-planner/carry.md`。確認事項 1〜3 は orchestrator が推奨案で決定済み。

## 計画からの逸脱・計画外の判断

1. **`Schema::fields()` は全フィールド:** 値 + 付随データの全フィールド（スキーマ順）を返し、値だけは `Schema::values()`、付随データは `Schema::carry()` にした。プール・`fill`・復元が全フィールドを扱うので、`fields()` を全体にすると既存の呼び出し（`restore` など）の意味が変わらない
2. **付随データの無いスキーマへの中継は、空の中継でも `CarryMismatch`:** 計画どおり「スキーマに付随データが無い」をエラーにした。空の `Carry`（付随データの無いメッセージの `carry`）の中継も断る。文言は `"slot 'w/ch0': expected to carry nothing, got nothing"`
3. **`Port[Any, Any, Any]` を `Port[Any, Any, Any, Any, Any]` に置き換え:** 型引数 CS / CR が不変で既定が `Never` のため、3 引数の注釈は付随データを持つ Port を受けない。`src/` の全箇所、`port.py` の self-typed overload（`recv` / `recv_with_time` / `has_received`）、tests の注釈（conftest を含む）を 5 引数にした。利用者が `Port[Any, Any, Any]` と書いている場合も同じ制約がかかる（docs の writing-a-node と `mujou.typing` に追記）
4. **ホストの読み出しを型でも弾く:** `Run.read` / `read_with_time` / `subscribe` / `wait_for` の引数を `Port[S, Any, Any, Never, Any]` にし、`send_carry` を持つ Port を pyright が断るようにした（実行時の `ValueError` は計画どおり）。`write` は `P: Port[Any, Any, Any, Any, Any]` のまま（実行時だけ）
5. **中継の結び付けは 2 段:** Session は全 Port の `Binding` を作った後に `Binding.relay_from(source)` で中継元の Binding を結ぶ。中継元の Port が後に bind されることがあるため
6. **`Session._slots` の項目を `(slot, spec, carry spec)` に:** 型 `SlotSpecs` を `state.py` に置き、`make_tree` / `read_tree` が付随データの spec を受け取る。`read_tree` は `(restores, carry_restores, restored)` の 3 つ組を返す
7. **テストの置き場所:** Port の実行時の振る舞い（`carried`・中継・`carry` の渡し忘れ/渡し過ぎ・ゼロコピー・CUDA）は既存の TestRecv / TestSend に倣って `tests/mujou/_core/test_port.py`（`TestCarried`）に置いた。状態の木のテストは、ノード（`Tagged` / `PassOn` / `CheckTag`）と既存の TestState がある `tests/mujou/_core/test_run.py`（`TestCarry`）に置いた。`test_state.py` は `StateDict` だけのテストなので変えていない
8. **復元のオラクルを変異で確かめた:** `Session._restore` の `binding.carry = ...` を外すと `test_a_restored_run_goes_on_as_one_never_stopped` が落ちることを確認（最初のスケジュールでは落ちなかったので、b が再開直後に受け取らずに送る順に直した）
9. **docs の例を `examples/carry.py` に置いた:** docs の規約（サンプルは examples/ から snippets で取り込む）に合わせた。ホストが書いた時刻を ID として運び、`wait_for` で自分のサンプルの結果を待つ。手で実行して確認した
10. **ホストの書き込みの文言:** `"cannot write the port 'X': it receives carried data, which the host cannot write"`、読み出しは `"cannot read the port 'X': it sends carried data, which the host cannot read"`
11. **free-threaded と GIL:** bindings の `Outlet.send(relay=)` は `PyRef<PyCarry>` から `Arc` を attach 中に複製し、`fill` は従来どおり detach して行う（`PyRef` は detach を越えられない。`Arc` の複製は atomic で、`Carry` は不変なので GIL の有無に依らない）。コメントに根拠を残した。GIL ありのビルドは CI で確かめる

## 確認事項

- なし（計画の確認事項 1〜3 は決定済みのとおり実装した）

## 別タスクの提案

- `docs/en/api/` に `Carry` のような内部型は出ない（`_native` は非公開）。`Port.carried` / `send_carry_spec` / `recv_carry_spec` / `carry_source` は `mujou.Port` の docstring から生成される
- 送るたびの `Arc<Buffer>` の箱の確保はベンチマークで見ていない。必要なら `benchmarks/` で測る
