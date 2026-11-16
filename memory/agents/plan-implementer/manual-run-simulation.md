# manual-run-simulation 実装ログ

## 計画どおり
- Rust: `slot::Staging`（open/commit）、`Slot.staged`、`Outlet::new(.., staging)`、`RunConfig.manual: Option<Arc<Staging>>`、`Lifecycle` の複数許可（`granted: Vec<bool>` と `open_steps`）、`Controller::step(groups)`
- bindings: `PyStaging`、`PyOutlet(staging=)`、`start(staging=)`。`just stub` で `_native.pyi` 再生成
- Python: `run_manually`、`ManualRun` docstring、`_start`/`_Kind`/`_RUN`/`_MANUAL_RUN`、`wire(tree, clock, staging)`、`testing.py` 削除

## 計画の細部との違い・判断
- `Shared` に `staging: Option<Arc<Staging>>` を持たせ、`Lifecycle` の `manual: bool` は `staging.is_some()` から作る（Lifecycle は Staging を知らない）
- `Staging` の open と commit は制御スレッドの `step` の中だけ。open は control lock の下、commit は lock を放してから
- 並列ステップで 1 ノードが失敗したとき、「すでに走っている更新は最後まで走る」が、まだ始めていないノード（ワーカー数が少なく待ち行列にいるもの）は停止の要求を見て更新せずに Finish へ行く。計画の「ほかのノードの更新は最後まで走り」は走り始めたものに限る。Rust のテストは TwoWorkers で 2 ノードの待ち合わせ（Meeting）を使って確かめている
- Rust の結合テスト用に `Probe.meeting`（`Meeting`）を `tests/runtime/main.rs` に追加
- K32 を scenarios.md に新設。テストの参照先は `test_execution.py` の `TestManualRunStep` と書いた（spec-test-author の命名に合わせて直す必要があるかもしれない）

## 検証できなかったもの
- `just type`（pyright）はこの環境に node が無く nodeenv の取得に失敗するため未実行
