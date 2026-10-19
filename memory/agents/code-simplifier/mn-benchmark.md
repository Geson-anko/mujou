# code-simplifier: M:N PR 3（ベンチマーク）の仕上げ

対象: `git log --no-merges 2ee224f..HEAD`（PR 3 自身のコミット）。レビュー: `memory/agents/code-reviewer/mn-benchmark.md`、`memory/agents/algorithm-reviewer/mn-benchmark.md`（どちらも approve）。orchestrator の裁定に沿って直した。

## 直した指摘

- **code S1（到達できないと判断。orchestrator が同意）**: 「戻したワーカーが ready の hook を先に回すと、戻した期限が遅れる」経路は起こらない。ready に仕事を積むのは、(a) 制御の側（`place` が true を返し、`wake_pool` / `wake_node` が timekeeper を起こす）と、(b) ワーカー自身（自分のノードだけで、同じ lock のまま直後に自分で取る）の 2 つだけ。ワーカーが timekeeper になるのは ready が空のときだけで、これも同じ lock の中で決まる。そのため ready が空でなければ、timekeeper は (a) で積まれた時点ですでに起こされていて、lock を取り直した後に先頭の期限を見直す。reviewer の例（再開で ready = [A, B, C]、ワーカー 2 本）では、B を終えたワーカーは ready に C が残っているので timekeeper にならず、C を取る。timekeeper ができるのは C が取られた後になる。algorithm-reviewer の照合メモの「ready が空でなければ…」の段落と同じ結論
  - 防御的なチェックは足さず（CLAUDE.md「起こり得ないシナリオ向けの処理を増やさない」）、pool.rs の `run_worker` と `place` のコメントに不変条件を書いた。specs/native-core.md の起こし方の段落にも同じ説明を足した
  - 例どおりの回帰テスト `period::a_blocking_resume_does_not_delay_another_nodes_update`（ワーカー 2 本、周期 P / 1000P / P、ノード 2 の `on_resume` がブロック。capture で再開させてから P 進め、ノード 0 が更新されること）を足した。修正なしのコードで通ることを確かめた（10 回連続で pass）
  - pool.rs のロジックは変えていないので、mixed の `interrupt_waiters` の回数は plan-implementer の測定（修正後 0 回/s）から変わらない。計測用のパッチを当てての再測定はしていない
- **code S2 / algo N3**: bindings に private な `_available_parallelism()`（`#[pyfunction(name = "_available_parallelism")]`、`std::thread::available_parallelism()` を読む）を足し、`default_workers` もこれを使うようにした。テストの `default_workers()` は `os.process_cpu_count()` の代わりにこれを読むので、cgroup の quota があっても期待値がコアと一致する。公開 API は増えていない（`mujou._core._native` の中だけ。`just stub` で `.pyi` を更新）
- **code S3**: specs/native-core.md の制御の lock の数値を記録の表に合わせた（light は 14 本と 28 本で取得 1 回あたり 3.4 µs と 10.3 µs。tanh は待つ取得 4〜6%、取得 1 回あたり 0.2 / 0.9 / 5.7 µs。78 µs の更新に対する割合は、ワーカーがコア数以下で 3% 以下、28 本で約 15%）。記録の本文の「多くて 7%」も表と合わないので同じ形に直した
- **code S4**: `memory/project_torch_module.md`（PR 3 で測った結論）と `memory/project_native_core_rust.md`（GIL が有効なときは半分）
- **algo N1**: テスト用の `Mode` に `TwoWorkers` を足し、`a_target_a_worker_puts_back_before_the_timekeepers_is_kept` と新しいテストはこれで回す（CPU 数に依らずワーカー 2 本）
- **code N1 / algo N2**: 「GIL ありのビルド」を「実行時に GIL が有効なとき（`sys._is_gil_enabled()`）」に（run.py の docstring、specs/nodes.md、specs/native-core.md の 1 か所、docs/{en,ja}/user-guide/running.md）
- **code N2**: benchmarks/mn_scheduler.py の達成率を `中央値 × 周期` で求める。見出しは「of 1/period」
- **code N3**: `.venv-gil/` を .gitignore に入れた
- **code N8**: specs の「評価し直し 1 回につき 1 度だけ」は `wake_pool` の中の話で、制御語を変えるときは `set_command` が別に 1 度起こす、と書いた

## 直さずに記録だけ

- code N4: macOS では `count_threads` が計測の窓の途中で `ps` を起動し、fork と待ちのぶんが測定に混じる。量は小さい
- code N6: 半分という規則の根拠は 1 台（M4 Pro 14 コア、`tanh`）だけ。2〜4 vCPU の runner では GIL ありのワーカーが 1〜2 本になり、少ないコアで GIL ありの CI が通るかは未確認
- code N7: `test_the_default_number_of_workers_all_take_updates` は実時間 0.5 s の間に全ワーカーが 1 回以上更新することに頼る。負荷の高い runner では理屈の上で落ちうる
- algo N4: 「ワーカーが戻すたびに `interrupt_waiters` を呼ばない」という性能上の性質を固定するテストは無い（数はベンチマークの一時パッチで測っただけ）
- code N5: GIL ありの既定は「ユーザーへの確認待ち」の注記をそのまま残した

## 簡素化

PR 3 自身の差分（pool.rs の起こし方、period の `Option` の削除、GIL の既定）を読み直したが、公開 IF と振る舞いを保ったまま明確に簡素になる書き換えは見つからなかった。テストの `Ticker` に `Ticker::new` を足し、構築の 3 か所をそろえた（`hold_resume` を足したため）。pool.rs のスケジューリングは書き換えていない（コメントだけ）

## 検証

- `just run`: green（cargo test 全通過、pytest 595 passed / 20 skipped、pyright 0 errors、docs-build 成功）
- `cargo test -p mujou-core --test runtime period::a_` を 10 回: 全 pass
