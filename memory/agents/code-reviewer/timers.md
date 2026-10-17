# timers（実行ごとのタイマー・Timestamp・スタブの自動生成）レビュー

対象: `feature/2026-10-16/timers`（`git diff feature/2026-10-16/cuda-tests..HEAD`、4 commit）

## verdict: request-changes

コアの変更（`Arc<Timer>` 化、LIVE 表、`Timestamp`、bindings）は正しく、後方互換も保たれている。must-fix は CI で stub の hook が動かない 1 件だけ。

## must-fix

### 1. pre-commit の CI ジョブで `mujou-stub` hook が動かない

- 対象: `.pre-commit-config.yaml:97-106`（`mujou-stub`、`language: system`、`entry: just stub`）、`.github/workflows/pre-commit.yml`
- 問題: hook は `just`・`uv`・`cargo` と、ビルド済みの拡張（`uv run` の中での `uv sync`。torch を含む）を必要とする。pre-commit の CI ジョブは `actions/setup-python` と `pre-commit/action` だけで、`just` も `uv` も入れていない。`pre-commit run --all-files` が `crates/` に当たるとこの hook が走り、`Executable 'just' not found` で失敗する。その結果 Format & Lint ジョブは PR ごとに赤くなり、計画の柱である「古いスタブの検出」も CI では働かない。`test_core_stub.py` が見るのは名前・docstring・基底クラスだけで、シグネチャの古さは検出しない
- 根拠: 判断ログ「古さの検出: `.pre-commit-config.yaml` の末尾に local hook を追記」。workflow に `setup-uv` / `setup-just` の手順がない（PR #14 の `tooling-py314` も python-version を変えるだけ）
- 対応案: pre-commit.yml に `astral-sh/setup-uv` と `extractions/setup-just` を足す。または CI ではこの hook を `SKIP=mujou-stub` で外し、test ジョブに `just stub && git diff --exit-code src/mujou/_core.pyi` を足す
- 確信度: 中〜高（workflow に手順がないのは事実。ubuntu-latest のイメージに `just` が無いことは高、`uv` が無いことは中）。深刻度: 高（CI が落ちる）

## should-fix

### 2. ランタイムとスロットが同じタイマーを使うことを、どこも保証していない

- 対象: `crates/mujou-python/src/runtime.rs`（`Runtime::new` の `timer`）、`crates/mujou-python/src/slot.rs`（`Outlet::new` / `InputGroup::new` の `timer`）
- 問題: 3 つが別々に `timer=` を受け取り、一致を確かめない。ずれると、Δt はタイマー A で測るのにメッセージの時刻はタイマー B で付く。さらに capture が凍結するのは A だけなので、B で時刻を付けるホストの書き込みは止まらず、状態の木のスロットの時刻とタイマーの状態が食い違う。今は `_session.py` がどれにも渡さない（すべてグローバル）ので、現時点では不具合にならない
- 根拠: 観点「スロットとランタイムが渡されたタイマーを一貫して使うか」。個々の部品の中では一貫している（`runtime.rs` の時刻の読み取りはすべて `self.timer`、slot は `self.timer`）。部品をまたぐ一貫性は利用側の配線任せ
- 対応案: 少なくとも `Runtime` の docstring か `docs/execution-model.md` に「そのノードの Outlet と InputGroup は同じタイマーで作る」と書き、後続の `mujou.run(timer=...)` の作業の計画に引き継ぐ
- 確信度: 中（設計上のリスク）。深刻度: 低〜中

### 3. stub を整形する ruff が 2 か所から来る

- 対象: `justfile` の `stub`（`uv run ruff check --fix` / `uv run ruff format`）、`.pre-commit-config.yaml` の `ruff-pre-commit`（`rev: v0.17.0`）
- 問題: `just stub` は venv の ruff を使う（dev 依存は `ruff>=0.8.4`、lock の版は今 0.17.0）。ruff の hook は v0.17.0 に固定されている。`uv lock --upgrade` などで版がずれ、整形の結果が変わると、2 つの hook が `_core.pyi` を交互に書き換え、pre-commit が毎回「ファイルを変更した」で落ちる
- 根拠: 今は版がそろっていて `just run` は通る。ずれたときに壊れる構造になっている
- 確信度: 低〜中。深刻度: 低

## nit

1. `tests/mujou/test_core_slot.py:18` のコメント「`mujou._core.DLDataType` exists only in the stub.」が嘘になった。生成したスタブには `DLDataType` が無い。確信度: 高
2. スタブの `GlobalTimer(Timer)` は `Timer.__new__(cls) -> Self` を継承するので、型チェッカーは `GlobalTimer()` を通す。実行すると `TypeError` になる（`test_the_global_timer_is_a_timer_without_a_constructor` もそれを確かめている）。確信度: 高。深刻度: 低（private なモジュール）
3. `Timer` は `GlobalTimer` のために `subclass` を付けたので、Python から `Timer` を継承してメソッドを override できる。ただしコアは `Arc<Timer>` を直接読むので、override は効かない。確信度: 高。深刻度: 低
4. スタブの型が手書きのころより緩くなった。`Outlet.send` / `send_to` / `restore` / `InputGroup.write` の values は `Any` に（以前は `Sequence[SupportsDLPack]`）、`Schema(fields)` は `SupportsGetItem[...]` になった。判断ログのリスクに書いてあり、必要なら `hint.rs` の型で戻せる。確信度: 高
5. `justfile` の `stub` は `read -r library module output` で空白区切りに読むので、checkout のパスに空白があると壊れる。確信度: 高。深刻度: 低
6. `Timestamp(ns=-1)` は `ValueError` ではなく、u64 の抽出の `OverflowError` になる。スタブの `__lt__(value: object)` などは `ts < 1.0` を型の上で通すが、実行すると `TypeError` になる。確信度: 高。深刻度: 低
7. `a_capture_freezes_only_its_own_timer`（`crates/mujou-core/tests/runtime.rs`）と `test_a_capture_does_not_pause_another_timer` では、もう片方のタイマーでランタイムが走っていない。2 つ目のランタイムを走らせ、capture 中もそちらのノードが更新を続けることまで見ると強くなる。確信度: 中
8. `mujou-stubgen` の `add_exceptions` は、例外を無条件に足して `incomplete` を常に false にする。将来 pyo3 が `create_exception!` を introspect するようになると重複する。ほかに introspect できないメンバーが増えた場合は、スタブの `__all__` が「見つかったメンバー」から作られるので `test_lists_every_name_of_the_module` が検出する（`pyo3-introspection` の `dunder_all_stubs` で確認した）。生成器の側で「見つからなかったのは EXCEPTIONS だけ」を確かめれば、エラーがわかりやすくなる。確信度: 中
9. `test.yml` / `type-check.yml` の paths に `*.pyi` が無いので、スタブだけを変える PR では pytest（`test_core_stub.py`）も pyright も走らない。普通はスタブが `.rs` と一緒に変わるので、影響は小さい。確信度: 高
10. `test_core.py` には `GlobalTimer` のインスタンスを明示的に渡すケース（`Outlet(..., timer=_core.timer)` / `Runtime(..., timer=_core.timer)`）のテストが無い。省略した場合と同じ `Arc` を使うことは、コードを読めばわかる（`IntoPyObject for GlobalTimer` が `global()` を基底に入れる）。確信度: 中

## 確認した点（問題なし）

- `Arc<Timer>` 化: `PauseHold` は `Arc<Timer>` を持つので、ランタイムが生きている間はタイマーも生き、停止は保持される。`Freeze` → `PauseHold` → `Timer` に循環は無い
- LIVE 表: キーは `Arc::as_ptr`。`Live(Arc<Timer>)` を数えている間はタイマーが生きているので、アドレスは再利用されない。`Live::drop` は自分の `Arc` から同じキーを計算する。数えるのは `freeze_for_startup` の LIVE ロックの下で、`mark_finished` は control ロックの外で drop する。ロックの順序は変わっていない
- `global()` は `&'static Arc<Timer>`。`_core.timer` の基底は `global()` の clone で、`mujou.time`、省略時のスロット、省略時のランタイムが同じタイマーになる（`test_without_a_timer_they_use_the_global_one` と、既存のグローバルタイマーでの全テストが担保している）
- bindings の後方互換: `src/mujou/*.py` は変更なしで pyright strict が 0 errors。`Outlet(schema, channels, name)` / `InputGroup()` / `Runtime(nodes, names)` / `_core.timer` / `Message.time_ns` はすべて残っている。`__module__` は `test_every_class_belongs_to_this_module` が確かめている
- `Timestamp`: `float` は `ns / 1e9` で `Timer::time()` と同じ式、`int` は切り捨て。比較と hash は ns による。キーワード専用の引数で、`repr` から `eval` で戻る
- free-threaded: `Timer` / `GlobalTimer` / `Timestamp` は `frozen` で、中身は `Arc<Timer>`（Sync）か `Copy`。`gil_used = false` は declarative module でも保たれる（`test_import_keeps_gil_disabled`）
- 成果物の汚染（`</content>` など）: なし

## 検証結果

- just run: pass（pre-commit の全 hook が Passed、`mujou-stub` を含む / cargo test pass / pytest 335 passed, 11 skipped / pyright 0 errors）
- pytest `tests/mujou`: free-threaded 3 回、`python -X gil=1` 3 回、すべて 314 passed, 11 skipped（`sys._is_gil_enabled()` が True になることも確認した）
- cargo test --workspace: 10 回連続 pass（既知の `readers_see_whole_messages_in_order` の間欠的な失敗は対象外で、今回は出なかった）
- 未検証: GIL ありの通常ビルド（3.14）の拡張で `just stub` が同じスタブを出すか（CI の pre-commit は PR #14 で 3.14 になる）。中身に版に依存する箇所は見当たらない
