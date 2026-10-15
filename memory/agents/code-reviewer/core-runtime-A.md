# core-runtime WP-A（スロット・バッファプール・DLPack・bindings）レビュー

対象: worktree `.claude/worktrees/core-slots`、ブランチ `feature/2026-10-16/core-slots`（`aa38fe9`）。`chore/2026-10-15/core-scaffold..HEAD` の差分。

## verdict: approve

must-fix はない。unsafe と DLPack の扱い、ArcSwap と版番号の切り替え、プールの返却、`wait_newer` の通知に、メモリ安全やデータ競合の問題は見つからなかった。should-fix の 3 件は、D に入る前か D と並行して直すことを勧める。

## must-fix

なし

## should-fix

1. **コピーの間 GIL を握ったまま** — `crates/mujou-python/src/slot.rs:377-389`（`PyOutlet::send_with`）、`:443-461`（`PyInputGroup::write`）、`:361-369` と `:480-488`（`restore`）
   - 問題: `__dlpack__` の呼び出しを先に済ませたあと、`copy_from` による memcpy を `py.detach` なしで、attach したまま行っている。GIL ありのビルドでは、大きな Tensor を送るたびに、コピーの間ほかのノードのスレッドが Python を実行できない。free-threaded ビルドでも、attach したまま長く走るスレッドが stop-the-world（GC）を遅らせる
   - 根拠: `docs/native-core.md:26`「コアの処理中は GIL を解放する」。計画書 WP-A 手順 4 で「fill の中で Python を呼ばないよう先に `__dlpack__` を済ませる」としたのも、fill を detach して走らせるための準備と読める。`Borrowed` が `Bound` を持つので今の形のままでは detach できない。capsule は `Py<PyAny>` で持ち、ポインタは Send にした薄い wrapper で渡す必要がある
   - 注意: detach すると、コピー中に別スレッドがその Tensor を `resize_` するとポインタが dangling になる窓が、GIL ありのビルドでも開く（free-threaded では今もある）。利用者側のデータ競合なので許容範囲だが、docstring に書いておくとよい
   - 確信度: 高（GIL を握っていることは確実）。深刻度: 中（性能の問題で正しさは壊れない。docs を厳密な仕様と読むなら must-fix に上げてよい）

2. **スキーマが byte 数のオーバーフローを検査していない** — `crates/mujou-core/src/slot.rs:38-49`（`numel` / `nbytes`）、`:62-93`（`Schema::new`）
   - 問題: `numel` は i64 の `product`、`nbytes` は usize の掛け算で、どちらも検査していない。release ビルドでは wrap する
   - 再現（worktree の venv で確認）:
     - `Schema([("", (2**32, 2**32), (2,32,1))])` と `Outlet` が通り、長さ 0 のバッファに shape (2^32, 2^32) を付けて貸し出す。torch は storage の大きさのオーバーフローで `from_dlpack` を拒むので、torch が読み手なら実害はない。ただ、範囲外を指す DLTensor を作っていることに変わりはない
     - `Schema([("", (2**61,), (2,32,1))])` は通り、`Outlet(...)` が Rust の panic「capacity overflow」になって `PanicException` が出る（`ValueError` にならない）。確保できなくもない程度に大きいサイズでは、`handle_alloc_error` でプロセスが abort しうる
   - 根拠: 計画書 WP-A の `Schema::new` は「不正なスキーマ」を `SlotError::InvalidSchema` → `ValueError` にするとしている。`TensorSpec` の shape は利用者が書くので、D を経由して届く
   - 確信度: 高。深刻度: 中（`Schema::new` で `checked_mul` を使って `isize::MAX` 以下か確かめれば済む）

3. **非連続テストの "expanded" が stride 0 を通っていない** — `tests/mujou/test_core_slot.py:122`
   - 問題: `x.unsqueeze(0).expand(2, -1, -1)[1]` は stride `(6, 1)`、offset 0 の連続な Tensor になる（`is_contiguous() == True` を確認した）。broadcast（stride 0）の経路は Rust のテストでも試していない
   - 根拠: `dlpack.rs:180-220` の `copy_strided` は `row_stride == 0` の要素ごとのコピーと、外側の次元の stride 0 を扱うが、どのテストもそこを通らない。計画書のテスト観点「転置・スライスした非連続の Tensor」の範囲内で、torch は expand した Tensor を stride 0 のまま DLPack で出す
   - 確信度: 高。深刻度: 低から中（例えば `torch.arange(6.).expand(4, 6)` に置き換えれば足りる）

## nit

4. `crates/mujou-python/src/slot.rs:50-55, 74` — `__dlpack__` が無いときと capsule が不正なときの `TypeError` に、スロット名もフィールド名も入っていない（shape/dtype の不一致には入っている）。D でホストの `write` の誤りを利用者が追うときに手掛かりが少ない。torch の `__dlpack__` 自身の例外（`requires_grad` で `RuntimeError` など）にも文脈が付かない。確信度: 高
5. `crates/mujou-python/src/slot.rs:140-146` — Tensor のメッセージ（フィールド名 `""`）で数が合わないと `expected field count 1 (fields ''), got 2` と出る。`''` は読み手には意味が取れない。確信度: 高
6. `crates/mujou-core/src/slot.rs:619-645`（bindings は `:443`）— 同じ `InputSlot` を 1 回の `write` に 2 度渡すと、黙って後の値が勝つ。D は Mapping を渡すので起きないが、`_core` の契約として doc に書くか拒否するとよい。確信度: 高
7. `crates/mujou-core/src/dlpack.rs:285-297` — `describe` で `Box<[i64]>` から生ポインタを取ったあとに、その Box を `Lent` へ move している。Stacked Borrows（Miri の既定は fields も retag する）では、move 時の Box の retag で生ポインタが無効になると判定されうる。実機で問題が出ることはない。`Box::into_raw` のあとに `(*lent)._shape.as_mut_ptr()` から入れ直せば、`manager_ctx` と同じ形で避けられる。確信度: 低
8. `crates/mujou-core/src/dlpack.rs:256` — doc の「with its shape and strides kept in `ctx`」は引数と合っていない（`ctx` という引数は無い）。確信度: 高
9. `crates/mujou-core/src/dlpack.rs:94` — `kDLCUDAHost` を受けるが、producer に stream を渡さない（`stream=None`）ので、`non_blocking=True` の D2H コピーが終わる前の pinned Tensor を読みうる。D は自分のストリームを同期してから送る計画なので今は起きない。前提を doc に一行書いておくとよい。確信度: 中
10. `tests/mujou/test_core_slot.py:305` — コメント「The writer can only run while this thread waits if the wait drops the GIL」が GIL の解放を確かめられるのは GIL ありのビルドだけで、開発環境の 3.14t では何も確かめていない（CI の GIL ありのジョブでは効く）。その旨を書くか、計画どおり GIL ありのビルドで回る前提をコメントに残すとよい。確信度: 中
11. `BufferView.__dlpack__` の `copy=True` と CPU 以外の `dl_device` を断る処理（`crates/mujou-python/src/slot.rs:224-233`）にテストが無い。IF 変更通知で足した振る舞いなので、1 本あるとよい。確信度: 高
12. `Outlet` は `!Clone` だが、`PyOutlet` は frozen かつ Sync で、複数の Python スレッドから同時に `send` できる。2 本が並ぶと、`Target::All` の外向きスロットと Channel が別の Message になり、外向きの時刻が戻ることもある（メモリ安全は保たれる）。書き手を 1 つに保つのは D のスレッド照合の責任なので、stub の「the only writer」に「呼び手が保証する」と書いておくとよい。確信度: 高
13. 判断ログ `memory/agents/plan-implementer/core-runtime-A.md` の「待ち手がいなければ書き手のコストは atomic の読み出し 1 回」は、実際には SeqCst fence と atomic の読み出しが 1 回ずつ（`slot.rs:372-373`）。確信度: 高

## 確かめて問題が無かった主な点（裁定の参考）

- 版番号: `InputGroup::write` が前の書き込みの V を lock の下で読み、`resolve(V)` で繰り上げ、Release で V+1 を置く。`InputSlot::load` は pair を先に、V を Acquire で後に読む。このため、ある pair P_j を見た読み手の V は必ず j-1 以上になる（V=j-1 の store → lock → P_j の store → ArcSwap の load と happens-before がつながり、write-read coherence から言える）。A → B の順に読めば b ≥ a、つまり新旧が混ざって見えることはない。Rust の 10^5 回のテストもこの形を確かめている
- `Waiters`: 書き手の「store → SeqCst fence → count」と待ち手の「count+1 → SeqCst fence → lock → 値」で Dekker 型の対になっている。count>0 のときに lock を取り直すので、通知は失われない。`last` は呼び手が Arc で持っているので、`ptr_eq` に ABA は起きない
- DLPack の構造体のレイアウトは `dlpack.h` 1.x と一致している。versioned と legacy の選び方（`max_version` の major ≥ 1）、`PyCapsule_New` が失敗したときの deleter の後始末、`destroy_*` が `PyCapsule_IsValid` で消費済みを見分ける点、受け取った capsule を消費せず保持して producer の destructor に任せる点、どれも正しい
- `copy_from`: device と dtype を検査してから shape、そのあと null を検査している。strides が null か row-major なら一括でコピーし、それ以外は行ごとに集める（負の stride と byte_offset も扱える）。`dst.len() == nbytes` で上限が決まっているので、書き込みが溢れることはない
- プール: `Buffer` が `Arc<BufferPool>` を持ち、Drop で返却する。fill が失敗したときも返却される。`Message` が読み手側に残っても、プールは生きている
- WP-D が使う IF（`Outlet.outbox/channel/send/send_to`、`InputGroup.input/write`、`InputSlot.read/restore`、`Slot.read/wait_newer/restore`、`Message.buffers/time_ns`）はそろっていて、D が困る欠けは見当たらない

## 検証結果

- just run: pass（pre-commit 全項目 Passed、cargo test は core 25 件・time ほか 23 件 ok、pytest は `test_core_slot.py` を含め全件 pass、pyright 0 errors）。実行後の worktree に差分は出ていない
- `</content>` などの混入: 無し（`crates/`・`src/`・`tests/`・判断ログを grep で確認）
- 追加の確認: worktree の venv で Schema のオーバーフローを再現した（should-fix 2）。"expanded" ケースの stride が `(6, 1)` であることを確かめた（should-fix 3）
