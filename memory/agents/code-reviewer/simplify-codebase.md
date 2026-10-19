# コードベース全体の簡素化 レビュー

対象: worktree `mujou-simplify`、ブランチ `refactor/2026-10-19/simplify-codebase`、`git diff 85e2ee7..HEAD`（simplify-a〜d の merge、40 ファイル +700 / −873）。

## verdict: approve

must-fix はない。振る舞いの変化は報告どおり period の予約名の `TypeError` の文言 1 つだけと確認した。オラクルの弱化が 1 件ある（should-fix 1）。マージ前に直すのが望ましいが、ブロックはしない。

## must-fix

なし

## should-fix

1. `crates/mujou-core/tests/time.rs:481-485`（`timestamp::now_is_the_time_of_the_timer`）／オラクルが弱くなった／共通の `virtual_timer()`（同ファイル 16-26 行）は `load_state` で時刻をちょうど 0 にする。以前は `Timer::new()` + `use_virtual_clock(true)` で、切り替えまでに実時計が進んだ 0 でない時刻だった。今は `ns == 0` なので、`Timer::now()` が常に `Timestamp::from_ns(0)` を返す実装でもこのテストは通る。`advance_ns` で 0 でない時刻に進めてから比べれば元の強さに戻る／確信度: 高（深刻度: 低）

## nit

1. `crates/mujou-python/src/runtime.rs:231-240`（`PyNode::method`）と 183-193（`CACHED_HOOKS`）／`Vec<Py<PyAny>>` の位置が `CACHED_HOOKS` の並びと暗黙に対応していて、フックを呼ぶたびに `position` で線形に探す。名前が 1 か所に減った利点のほうが大きく、性能差も無視できる。`(self.methods.as_ref()).zip(..).map(..)` の連鎖は、以前の `match` より一読しにくい／確信度: 中
2. `tests/mujou/_core/test_run.py:168-194`（`run_together`）／docstring には「exactly one run started を確かめる」とあるが、実際に確かめるのは「`RuntimeError` がちょうど 1 つ」と「結果が `len(given)` 個そろったこと」。2 スレッドでしか使わないので同値だが、3 つ以上渡すと docstring と食い違う／確信度: 高
3. `crates/mujou-core/tests/runtime.rs:14`（`LONG`）／トップレベルに移した `QUIET` には doc コメントがあるのに、`LONG` には無い／確信度: 高
4. `.claude/hooks/_common.sh` を source する 3 本（`compaction-recovery.sh:10` など）／`dirname` は外部コマンドで、以前の `cat` を置き換えた形になる。依存の種類は前と同じ（coreutils）だが、`${BASH_SOURCE[0]%/*}` なら bash だけで済む。PATH が空でも exit 0 になることは確かめた（stderr に `dirname: command not found` が出る）／確信度: 中
5. `tests/test_claude_hooks.py`／shell hook の fail-open（JSON が壊れている／`_common.sh` が無い／python3 が無い）を確かめるテストが無い。もともと無かったので後退ではない。ただ、今回 `_common.sh` が無いと何も出力せずに終わる、という新しい失敗の形が増えた。手で試すと 3 本とも exit 0 だった／確信度: 中
6. `memory/agents/code-simplifier/simplify-c.md`（tests の段落）／「workers 側は以前と同じく `dedicated` を渡さない」とあるが、実際は `dedicated=None` を明示して渡している（`tests/mujou/_core/test_native_runtime.py:153-157`）。stub の既定値は `None` なので振る舞いは同じで、メモの書き方だけの問題／確信度: 高

## 確認した点（根拠）

- Rust コア
  - `spawn` helper（`runtime/mod.rs:696-710`）: `RUNTIME_OF_THREAD.set` は、以前と同じく `N::on_thread` の内側で body の前に走る。制御スレッドの `Controller` はスレッドの中で組み立てるようにしたが、spawn に失敗したときに落ちるもの（threads の handle、`messages`、`started_tx`）は以前と同じ
  - `check_start`: 検査の順序（names → dedicated → last update → periods の長さ → 0 の period）とエラー文は以前と同じ
  - pool の `Task.last_ns` の削除: `last_update_ns[i]` を書くのは start 前の初期化と `record`（control lock の下）だけ。読むのは `Phase::Loop` に入ってから（WaitGo→Loop の遷移で、以前は同じ値を `last_ns` に写していた）なので、常に同じ値になる
  - `time::remaining`（`pub(crate)`）を 4 か所で共有したのは同値の置き換え。`Runtime::failure`・`seconds_to_ns` など、bindings が使う pub API は残っている
- bindings
  - `public()` を消し、`update` で `call_as("_after_update", "on_update", ..)` にした。`_after_update` を経由するのは `update` だけなので同値。Python 側の `test_an_error_after_an_update_stops_the_network`（`match="raised in on_update$"`）が押さえている
  - `states.into_iter().flatten()` は、直前で長さを照合しているので同値。stubgen の並べ替えも、出力が同じ（stub の差分は Node docstring の空行 1 行だけ）
- Python
  - `Run.__exit__`: 通る例外の集合は以前と同じ
  - `_register`: `children_of` を遅らせて呼ぶようにした。読み取りだけで副作用が無いので同値
  - `__setattr__` / `__delattr__` の `_forget_state_name`、`_receiving`、`start_manual` の削除（残る参照は conftest の同名 fixture だけで別物）
- テストのオラクル
  - core の `restore_sets_whether_sent` は、Slot と InputSlot それぞれの restore テストに sent の true/false 両方が移っている
  - Gate のテストは `Probe` の `gate` とログの `Stop` で同じ性質を見ている
  - Recorder→Probe の移行では、`enter` の記録を `events()` と order の抽出で除いている
  - Python は削除 0。parametrize した 1 組は、workers 側にも同じ assert（順序と `loaded`）が残っている
- 境界をまたぐ参照: benchmarks が使う `mujou._core.hooks.enter_node`・`Node._on_enter`・`Node._before_hook` は残っている。`pyright` も 0 errors
- CI: 旧版と新版の docs / test / type-check の workflow を PyYAML で読み、`on`・`jobs`・`permissions` が完全に一致することを確かめた。旧版も push と pull_request の `paths` は同一だった。GitHub Actions は 2025-09 から anchor / alias を受け付ける（merge key `<<` は不可で、今回は使っていない）。この PR は 3 つの workflow ファイル自身を変更しているので、それぞれの `paths` に掛かり、PR の CI で anchor の解釈が実地に試される。`pre-commit/action@v3.0.1` は内部で `actions/cache@v4` を使って `~/.cache/pre-commit` を保存するので、cache の削除は重複の除去として正しい
- 成果物汚染: 変更した全ファイルを `</content>` / `</parameter>` で grep し、該当なし。全ファイル末尾に改行がある

## 検証結果

- just run: pass（cargo test: lib 33 / runtime 49 / time 37 passed、pytest 595 passed / 20 skipped、pyright 0 errors、mkdocs build --strict 成功）。実行後の作業ツリーは clean（format による変更なし）
