# simplify-a: crates/mujou-core の簡素化（2026-10-19）

担当 A（Rust コア）。worktree `mujou-simp-a`、ブランチ `refactor/2026-10-19/simplify-a`。基点は `85e2ee7`。

## 結果

- 8 ファイルで +304 / -473 行（差し引き -169 行）
  - `src/`（`slot.rs` の単体テストを含む）: +93 / -121
  - `tests/runtime.rs`: +195 / -326
  - `tests/time.rs`: +16 / -26
- テストの本数: lib 34 → 33（重なっていた 1 本を統合）。runtime 49、time 37 は変わらない
- `uv sync` → `just run` は green（cargo clippy -D warnings、595 passed、pyright 0、mkdocs strict）
- bindings（`crates/mujou-python`）から使う公開 API は変えていない。`Runtime::failure` は未使用に見えたが、bindings が `core::Runtime::failure` をパスで使っていたので残した

## コミットと理由

1. 実時間の締め切りまでの残り時間を `time::remaining`（`pub(crate)`）1 つにまとめた。`Runtime::join`、`capture`、`Slot::wait_newer`、`Timer::wait_until_ns_or` の 4 か所で同じ計算をしていた
2. ランタイムのスレッドを起こす処理を `spawn` helper 1 つにした。`Builder` の連鎖と `RUNTIME_OF_THREAD.set` が 3 か所（専用スレッド、ワーカー、制御スレッド）に分かれていた。印を付ける場所が 1 つになり、付け忘れが起こらない。`on_thread` の内側で印を付ける順序は前と同じ
3. `fail(i, HookError)` の中で `NodeFailure` を組み立てるようにした。`call` と `state_dict` が同じ組み立てを重複して書いていた。`check_start` の長さの検査 4 つを 1 つのループにした。メッセージの文面と検査の順序は前と同じ
4. プールの `Task.last_ns` を削除し、`Shared::last_update_ns[i]` を読むようにした。2 つは同じロックの下で常に同じ値を書かれていた（二重の状態）
5. 呼び出し元が 1 つだけの `FieldSpec::numel` を `nbytes` に畳み込んだ
6. runtime の結合テストで、`Recorder`＋`Setup`＋`recorders`＋`events_of` と `stop_request::Gate` を、pool のテストの `Probe` に統合した。`Probe` をトップレベルへ移し、`fail_on_update` と `fail_enter` を `Failure { Error, Panic }` で表すことにした。重複していた定数 `LONG`/`QUIET` もトップレベルの 1 か所にまとめた。オラクルは減らしていない。順序を見るテストでは `enter` の記録を除いてから比べる
7. time のテストで 2 つの mod にあった `virtual_timer` を 1 つにまとめた（時刻 0 から始まるもの）
8. slot の `restore_sets_whether_sent` を、Slot と InputSlot それぞれの restore テストに統合した（sent の true/false をどちらも確かめる）
9. pool.rs のコメントにあった古い記述「period 0」を「period 1 ns」に直した。0 の周期は弾かれる

## 整理しなかったもの

- `runtime/pool.rs`・`control.rs`・`node.rs` のロックの取り方・起こし方・順序には手を付けていない。上の 2 と 4 は、同じ意味であることを説明できるものに限った
- テストの `Ticker`（period の各テスト）: `Probe` に統合するとノードごとの受信チャネルが共有の `(id, dt)` になり、40 ほどの assert を書き換えることになる。period のテストはノードごとのチャネルのほうが読みやすいので残した
- テストの `Starter`・`Caller`: 振る舞いが固有で（start で時刻を動かそうとする／自分のランタイムを待つ）、`Probe` に足すとフィールドが増えるだけなので残した。空の hook のボイラープレートをマクロで消すのは「賢い」書き換えになるので見送った
- `dlpack.rs` の `delete_versioned`/`delete_legacy`: trait を使えば generic な deleter 1 つにできるが、unsafe な FFI 境界が抽象的になるので見送った
- `Outlet::send_with` と `InputGroup::write` が `Message` を直接組み立てている箇所: 2 か所だけなので抽出しなかった
- `Slot::restore` と `InputSlot::restore` の doc の重複: どちらも公開 API の doc なので残した
