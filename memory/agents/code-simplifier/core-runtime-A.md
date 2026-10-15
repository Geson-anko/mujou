# core-runtime WP-A の簡素化とドキュメント同期

対象: `feature/2026-10-16/core-slots`（`d967e20` の上）。`_core.pyi` の WP-A の区切りにある名前・引数・戻り値・例外は変えていない。unsafe、atomic の順序（fence・Acquire/Release）、GIL を外す範囲も変えていない。

## 簡素化

- `crates/mujou-python/src/slot.rs`: 「GIL を持ったまま値を借りて、GIL を外してコピーする」処理が `PyOutlet::send_with`・`PySlot::restore`・`PyInputSlot::restore` の 3 か所に同じ形で書かれていたので、`write_values(values, schema, slot, write)` 1 つにまとめた。`write` は `&dyn Fn(bytes)` の `fill` を受け取り、core の `send_with` / `restore` に渡す。detach の中で動くのは以前と同じくコピーとコアの書き込みだけ。`InputGroup.write` は複数スロットを扱うので別のまま
- `borrow_values`: capsule と pointer を `_capsules` に push するループを `unzip` に置き換えた
- 「slot 'a', field 'x'」の書式が core（`write_location`）と bindings（`location`）に重複していたので、core の `slot::location` を pub にして bindings はそれを使う。core 側は `Vec` に push して join する形を 4 通りの `match` にした
- `Ungil` でなく `Send` を境界にした: pyo3 0.30 の非 nightly では `Ungil` は `Send` からの blanket impl なので、`impl Ungil` を境界にすると closure の中で `Send` が導けない

## ドキュメント同期

- `_core.pyi`: `Schema` の ValueError に「1 つの確保に収まらない大きさ」（`checked_nbytes`）を追加。`BufferView` に `copy=True` と CPU 以外の `dl_device` で `BufferError` を追記。`Slot.wait_newer` に timeout の `ValueError`、`Slot.restore` / `InputSlot.restore` に「`Outlet.send` と同じ例外」を追記
- bindings の `PyInputGroup::write` の doc comment を stub と同じく「どの瞬間にも」に弱めた

## 見送ったもの

- `__dlpack__` で capsule の作成に失敗したときの deleter 呼び出しと `destroy_versioned` / `destroy_legacy` の重複: 型が違う 2 つの構造体をまとめるには trait が要り、2 か所なので見送った
- `FieldSpec::numel` は `nbytes` からしか使われていないが、core の pub API なので残した
