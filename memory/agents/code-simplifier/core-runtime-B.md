# core-runtime WP-B の簡素化

対象: `crates/mujou-core/src/runtime.rs`、`crates/mujou-python/src/runtime.rs`。公開 IF（`_core.pyi`、`mujou_core::runtime` / `time` の pub API）は変えていない。`time.rs`・`_core.pyi`・`docs/execution-model.md` は変更不要と判断した。

## コア（`mujou-core/src/runtime.rs`）

- `Shared::mark_finished(guard)` を抽出した。`Controller::finish` と `abandon` に同じ「凍結を解く → `live` を取り出して control lock の外で落とす → `finished` を立てる」が重複していた。lock の順序（LIVE の lock は control lock を手放してから）は変えていない
- `Shared::freeze_for_startup(timer_state)` を抽出した。`Runtime::start` から「LIVE の確認 → Startup の凍結 → `load_state` → 数える」を移しただけで、lock の順序（LIVE → control）と処理の順序は同じ。`start` は検証・構築・凍結・`last_update_ns`・spawn の流れが読めるようになった
- `Controller::end()`（`finish` してから終わり方のエラーを返す）を足し、`startup`・`ensure_running`・`step`・`capture`（2 か所）の `finish(); return Err(s.finished_error())` を置き換えた。pass-through の `Shared::finished_error` を消した
- `ensure_running` を `if self.done || failed { return Err(self.end()) }` にした（`finish` は終了後は何もしないので同値）
- `run_node` の `if started { .. } else { drop(c) }` を平坦化した（`started` が偽なら待ちも更新も無い）
- `capture` の `loop { if .. break }` を `while` にし、`state_dict` の収集を `collect::<Result<Vec<_>, ()>>()` にした（最初の失敗で止まるのは同じ）
- `remaining(deadline)` を抽出した（`capture` と `join` の重複）。`join` の末尾は `stopped(..).map(|()| true)` にした
- `Report` / `ExceptionHook` に `Default` を derive し、初期化の列挙を消した。1 回しか使わない `Panicked::from_payload` を消した
- LIVE の確認を `contains_key` にした（数が 0 になると要素を消すので同値。`LIVE` の doc にその不変条件を書いた）

## bindings（`mujou-python/src/runtime.rs`）

- `Pending { nodes, names }` を `Vec<Py<PyAny>>` にした。`names` は `self.names` と同じ値の重複だった
- `wait` の `let mut reply = reply;` を引数の `mut` にした

attach/detach の構造、atomic の順序、hook の順序には触れていない。

## 確認

- `cargo test --workspace` を 5 回、`tests/mujou/test_core_runtime.py` を 3.14t で 5 回・`PYTHON_GIL=1` で 3 回、すべて通過
- `just run` 通過
