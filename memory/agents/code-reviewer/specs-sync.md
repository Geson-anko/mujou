# specs/ と memory/ を実装に合わせる（docs/2026-10-18/specs-sync）レビュー

## verdict: approve

must-fix は無い。書き足した事実はどれも実装と合っていた。M:N・ワーカープール・実行クラス・ネイティブノード・generator の記述には触れておらず、「ユーザーの決定」と書いた箇所もすべて specs か memory に記録があった。ただし同じ食い違いが別の箇所に残っているので、should-fix の 1〜3 はこの PR で直すことを勧める。

## must-fix

なし

## should-fix

1. specs/execution-model.md:60 / 「どのタイマーも時刻を秒（f64）とナノ秒（u64）の両方で持つ」が残っている。messages.md:92 は「ナノ秒（u64）で持ち、秒はそこから計算する」に直したので、2 つの仕様が食い違う / crates/mujou-core/src/time.rs:9「The time is held as nanoseconds (`u64`); seconds (`f64`) are derived from it.」 / 確信度: 高
2. memory/project_message_types.md:21 / 「時刻は秒（f64）とナノ秒（u64）の両方で持つ」が残っている。同じファイルの :29「プールの初期サイズと縮小は実装時に決める」と :16「スキーマは構築時に確定する決定と揃えるため」も古い。同じファイルの :19 は直したのに、これらは直していない / slot.rs:308-326（Message は time_ns だけを持つ）、slot.rs:238-246,287-293（空で始め、縮小しない）、messages.md の修正 / 確信度: 高
3. memory/project_native_core_rust.md:10 / 「コアの処理中は GIL を解放する」が残っていて、native-core.md:28 の修正（`Slot.read` は GIL を持ったまま）と食い違う / crates/mujou-python/src/slot.rs:242-244 / 確信度: 高
4. specs/open-decisions.md:9 / 元の文には「クラス名・メソッド名 … は推奨案で実装（ユーザー確認待ち）」とあったが、書き換えで「クラス名・メソッド名」が文から消えた。決定済みにはしていないが、確認待ちの項目も減っている。共通ルール 3（記録の無い項目は状態を変えない）に照らすと、ユーザーの決定が記録されていない名前（`Network`・`Graph`・`connect`・`send_to` など）は確認待ちのまま残すべきである / git diff の旧 :9、memory/project_open_decisions.md:10 は「名前とホスト入出力は決定済み」 / 確信度: 中（memory 側が「名前は決定済み」と書いているので、意図した整合である可能性がある。orchestrator の裁定待ち）
5. specs/execution-model.md:239（PauseTimeoutError の箇条書きの末尾）/ 「止まらなかったノードは `on_pause()` を呼ばれないまま続ける」は、`on_update` の途中にいたノードにしか当てはまらない。`on_pause` が長く、その最中に時間切れになったノードは、戻った後に `paused_at` を立て、`run_epoch > epoch` なので `on_resume` も呼ばれる / crates/mujou-core/src/runtime/node.rs:186-203、control.rs:112-122、mod.rs:184-197 / 確信度: 中
6. specs/native-core.md:18 / 「詳細は [実行モデル](execution-model.md#実装)」のリンク先は「グローバルタイマー」の下の `### 実装`（保持とスイッチ）である。制御スレッドの詳細は `## 実装`（`#実装-1`、:229）にある。段落の前半が制御スレッドの話なので、両方へリンクするか、`#実装-1` も足す / specs/execution-model.md:68,223 / 確信度: 中

## nit

1. specs/overview.md:13,15 / 2 段落の中で「未実装」を 3 回書いている（「PC-ALM、未実装」「どちらもまだ実装していない」「である（未実装）」）。1〜2 回で足りる / 確信度: 高
2. specs/overview.md:28 / 括弧の中に括弧が入っている（「（状態の取得などのための停止だけは例外。その実行のタイマー（既定はグローバルタイマー）も止まる。…）」）/ 確信度: 高
3. specs/python-environment.md:7-9 / 新しい「## 依存」の節が、冒頭の 3.14t の段落と、その理由の節「free-threaded 版を使う理由」の間に入り、2 つを切り離している。末尾に置く方が流れがよい / 確信度: 中
4. specs/network.md:107 / 「`mujou.run` が実行するのは、宣言とは別のオブジェクトである構築した構造（`Graph`）だからである。」は名詞の前の修飾が重く、「〜だからである」で理由を後から足している（feedback_japanese_style）。元の文にあった形だが、書き換えたのを機に直せる / 確信度: 中
5. specs/open-decisions.md:9 / 「下の「2026-10-17 の推奨案」の節に書いた `Run` の API の整理も決定済み」が指す節の見出しは「2026-10-17 の推奨案（ユーザー確認待ち）」である。決定済みの項目を「確認待ち」の節の中で指していて、読者が迷う / 確信度: 中
6. specs/open-decisions.md:39 / 公開名の一覧に `StateDict` を足したが、この段落は「推奨案で実装（ユーザー確認待ち）」の段落である。`StateDict` そのものは :68 でユーザーの決定になっている。一覧を事実として示すだけなら問題は無い / 確信度: 低
7. specs/open-decisions.md:68 / 「木のテンソルは各ノードのデバイスに置いたまま」とあるが、スロットの値は CPU の新しい Tensor である（execution-model.md:231）。正確には「ノードの状態のテンソル」/ 確信度: 中
8. specs/update-rules.md:37、specs/pc-alm-continuous-time.md:71 / 「記録は `memory/reference_pc_alm.md` などにある」。記録が見つかったのは reference_pc_alm.md:9 だけで、「など」は要らない（項目リストの指示どおりの文言ではある）/ 確信度: 中
9. memory/project_host_io.md:8 / 「`timeout=)`（`timeout` は位置引数でも渡せる。…）（条件を満たす最初の値まで待つ。…）」と括弧が 2 つ続いている / 確信度: 高
10. memory/project_torch_module.md:14 / 書き換えで「更新式を直接計算するノードは `requires_grad=False` のテンソルで持つ」という指針が消えた。specs/torch.md:135 には残っている / 確信度: 低
11. memory/project_gpu_policy.md:13 / 「`on_start` の前に内部の hook `_on_thread_start` で」は事実として正しい。ただしスレッドの開始の hook と CUDA のストリームの話なので、M:N の PR と重ならないか、念のため orchestrator が確かめる / 確信度: 低
12. specs/native-core.md:28 / 「`Slot.read` だけは GIL を持ったまま」とあるが、`InputSlot.read` も同じく GIL を持ったまま lock を取る（slot.rs:447-449、core slot.rs:718-724）。「読み出し（`read`）は」とする方が正確 / 確信度: 中

## 確かめた事実（主なもの、すべて一致）

- PC-ALM・`mujou.pc` は未実装（src/mujou/ に pc が無く、tests/ にも PC-ALM のテストが無い）
- 書き込み・restore・wait_newer は `py.detach` で GIL を外す。`Slot.read` は GIL を持ったまま `Arc::clone(&lock(..))`。`kDLCUDAHost` を受け付ける（dlpack.rs:17,99）
- rust-toolchain.toml 1.96.0、`rust-version = "1.85"`、test.yml の MSRV ジョブ（`cargo +$msrv check --workspace --all-targets`）、`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`、pre-commit.yml の `SKIP: mujou-stub`、test.yml の `just stub && git diff --exit-code`、mujou-python の `test = false`
- 依存は numpy と torch
- capture: 制御スレッドは要求を 1 つずつ処理し、request_stop も制御スレッドのキューを通る。時間切れのときは止まったノードに resume。build の例外はネットワークを再開してホストへ上がる。ノードの state_dict の例外はシステムを終了する（control.rs:97-143）
- 停止中の例外ではノードに on_resume を呼ばない（node.rs:197-203）
- hook は start の最後に 1 回だけ探す。state_dict は `_capture_state` が毎回探す（bindings runtime.rs:169-207,313-317）
- hook の中からの `mujou.run` と Run のメソッドは RuntimeError（run.py:475,532-542）。`Run()` と `Graph()` は TypeError
- ルートのノードはクラス名（session.py:190-196）。timer が None になるのは `_settle` の後（session.py:251-283）
- 周期の ns への変換は `as u64` で飽和し、目標は saturating_add で張り付く
- TensorSpec・DictSpec の制約、Port の `initial` の TypeError、connect の例外、`Port[S, R, M]` の既定、キーワード専用の引数（spec.py, port.py）
- プールは空で始め、空きが無いときだけ確保し、drop で戻す（core slot.rs）
- Message は time_ns だけを持つ
- 起動の順序（スロット → 周期の検査 → state → inputs → timer → 起動、session.py:47-177）
- Node の `__setattr__`（node.py:215-224）
- `mujou.time` は torch を import しない。run.py は state を import する。state.py:78 で `add_safe_globals`。`TensorSpec.rand` は既定の生成器を使う。受け渡しに pinned memory を使わない（spec.py:371, binding.py:33）
- `__all__` に StateDict と `__version__` がある
- mod.rs のコメント: Rust の state_dict は control.rs:132 からしか呼ばれない（正常に終了した後の取得は Python のホストのスレッドで行う）
- 決定の記録: 2026-10-16 の Run ハンドルと inputs/state（project_runtime_api_revision.md:11）、on_start の例外を run が送出する形の承認（同 :24）、StateDict.cpu/to（execution-model.md:153）、pause_timeout の移動（execution-model.md:239、open-decisions.md:66）。has_received は決定の記録が無く、確認待ちの一覧に入れたのは正しい

## 検証結果

- just run: 実行していない（`format` が pre-commit を全ファイルにかけてファイルを書き換えうるため、「ファイルを修正しない」指示を優先した）。変更は文書と Rust のコメントだけである
- codespell（変更した全ファイル）: pass
- cargo fmt --check -p mujou-core: pass
- リンクとアンカー: 変更した specs と memory の相対リンクをスクリプトで確かめた。新しく壊れたリンクは無い（`#実装-1` は重複した見出しの正しいアンカー）
- `</content>` の混入: なし
- M:N・ワーカー・実行クラス・ネイティブノード・generator の生成時期についての差分の行: なし（generator は予約名として名前が出るだけ）
