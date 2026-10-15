# core-runtime WP-A（スロット・DLPack・bindings）の判断ログ

計画書: `memory/agents/implementation-planner/core-runtime.md` の WP-A。

## IF 変更通知

Python から見える `_core` の名前・引数・戻り値は計画書どおり。次の 3 点は計画書に無い振る舞いの追加で、既存の約束は変えていない。

- `InputGroup.write` に別の `InputGroup` の `InputSlot` を渡すと `ValueError`。同じ `InputSlot` を 1 回の write に 2 度渡しても `ValueError`（レビュー対応で追加）
- `Outlet.send` などの `values` に Sequence でなく Tensor 1 つを渡すと `TypeError`（Tensor も iterable なので、黙って行に分解されるのを防ぐ）
- `BufferView.__dlpack__` は `copy=True` と CPU 以外の `dl_device` を `BufferError` で断る（DLPack の規約どおり）。読む側は `torch.from_dlpack(view).clone()` でコピーする

注意（C/D 向け）: `DLDataType` と `SupportsDLPack` は stub にだけある型で、実行時の `mujou._core` には無い。`src/` から使うときは `if TYPE_CHECKING:` の下で import する。

## 計画からの逸脱（Rust の内部 API。bindings だけが使う）

- `InputGroup::write` は `Vec<(&InputSlot, Box<dyn FnOnce>)>` でなく `write(&[&InputSlot], fill: FnMut(usize, &mut [&mut [u8]]))`。クロージャを箱に入れずに済む
- `Slot::restore` / `InputSlot::restore` は `Arc<Message>` でなく `restore(time_ns, fill)`。`Message` はプールのバッファからしか作れないので、スロットがスキーマとプールを持ち、自分で作る。そのため `Slot` は `schema()` を持つ
- `wait_newer` 用の通知は外向きスロットだけでなく全スロットに持たせた。Python の `Slot` 型は外向きと Channel で共通なので、どちらでも `wait_newer` が動くようにした。待ち手がいなければ書き手のコストは SeqCst fence と atomic の読み出しが 1 回ずつで、lock は取らない（Rust のテストで確認）
- `SlotError` の各 variant が `slot` を持ち、`copy_from` はスロット名を知らないので空のまま返し、呼び手が `at_slot(name)` で埋める
- `DType` は `dlpack::DLDataType` の再 export

## 実装上の判断

- DLPack の capsule は消費しない（名前を `used_*` に変えない）。コピーの間 capsule を保持し、破棄時に producer の destructor が deleter を呼ぶ。計画の「deleter を呼んで名前を変える」と結果は同じで、所有権の移動が無い分だけ単純
- `__dlpack__(max_version=(1, 0))` が `TypeError` なら引数なしで呼び直す（計画どおり）。版の major が 1 以外の versioned capsule は `TypeError`
- バッファは 64 バイト境界に揃えて確保する（torch の CPU アロケータと同じ）。DLPack で貸すときに torch が作った Tensor と同じ揃え方になる
- 貸し出す DLTensor の strides は null にせず row-major の値を入れる（DLPack 1.2 が producer に求めている）。READ_ONLY は立てない（計画どおり）
- プールは縮めない。空きリストの長さは、同時に生きていたバッファ数の最大で頭打ちになる
- `Slot.wait_newer` は GIL を外し、50 ms ごとに `check_signals` する（`sleep` と同じ）。間隔の定数は `lib.rs` を触らないために `slot.rs` に別に置いた。WP-D 以降で共通化してよい
- エラーメッセージの dtype は `float32`・`bfloat16`・`bool` のように torch と同じ名前で出す。コアは dtype を持たないが、表示だけは利用者が読める名前にした
- stub の import は区切りの中に `# noqa: E402` 付きで置いた（B と import 行で衝突しないため）。結合後に先頭へまとめてよい

## 確かめたこと

- torch 2.14 は kDLBool（`(6, 8, 1)`）と bf16（`(4, 16, 1)`）を `__dlpack__` / `from_dlpack` の両方向で受け付ける（10 種のビット単位の往復テストが通る）

## レビュー対応（code-reviewer / algorithm-reviewer、どちらも approve）

- send・write・restore の memcpy を `py.detach` の中で行うようにした。capsule は呼び出し元のフレームで `Bound` のまま持ち（detach の中では触らない）、ポインタは `Send` の薄い wrapper（`TensorPtr`）で渡す。コピー中に別スレッドが元の Tensor を `resize_` や解放するのは利用者側のデータ競合で、stub の `Outlet` の docstring に書いた
- `Schema` は各フィールドのバイト数を `checked_mul` で求め、`isize::MAX` を超えたら `InvalidSchema`（`ValueError`）にする。以前は `Outlet` の作成で panic していた
- `__dlpack__` が無い値のエラーにスロット名とフィールド名を入れ、producer 自身の例外（torch の `requires_grad` の `BufferError` など）には `add_note` でスロット名を添える。数の不一致は「expected 2 tensors (fields 'x', 'y'), got 1」、Tensor のメッセージでは「expected 1 tensor」と出す
- DLPack で貸す tensor の shape・strides・`manager_ctx` は、`Box::into_raw` の後に box の中から取り直して設定する（Stacked Borrows でも生ポインタが無効にならない）
- 前提の明文化: `Outlet` の書き手は 1 スレッド（所有ノードのスレッド）で、呼び手が保証する。`Slot` / `InputSlot` の `restore` は全ノードが止まっている間だけ呼ぶ。`wait_newer` はホストの `subscribe` 専用で Port には出さない。`kDLCUDAHost` は stream を同期せずに読むので、送り手が同期してから渡す
- `InputGroup.write` の docstring は、algorithm-reviewer の指摘に合わせて「どの瞬間にも新旧は混ざらないが、2 つのスロットを順に読むとその間に write が終わることはある」に弱めた。`docs/network.md` 160 行の文面も同じく強すぎるが、docs は WP-A の範囲外なので WP-D の docs 更新で直す
