# ステップ 3（ランタイムの状態機械を 1 本にする）: レビュー差し戻しへの対応

## must-fix

- プールの M=8 の性能低下（15〜20%）: 受け入れずに直した。ノードを戻したワーカーが `notify_one` で別のワーカーを起こしていたのが原因で、起きたワーカーは空のキューを見て眠り直すだけだった（vcsw/s が多い）。戻したワーカーは自分ですぐキューを見るので、ワーカー側の `place` は誰も起こさない（`c.sched.place` を直接呼ぶ）。起こすのは制御側（`Shared::schedule`）と、仕事を取った後に残りがあるとき（`run_work` の冒頭）だけにした。同じ時間帯に交互に測った結果: light・period 0・M=8・pool で、前（1f1a465）の 1.33M〜1.40M に対して 4.2M〜4.4M。M=64・pool で 0.61M〜0.66M に対して 0.87M。
- 起動中の Ctrl-C で、ランタイムが Python から辿れなくなっていた: bindings の `start` は、起動の待ちが失敗したら（ノードの例外か中断）`stop` と `join` を待ってから例外を返す。2 回目の中断だけは、裏で止まりに向かわせたまま返す。`Session.start` は `except BaseException` で lifetime から外す。回帰テストは `TestInterpreterExit::test_ctrl_c_while_starting_raises_once_the_network_has_stopped`（slow。`PYTHON_GIL=0/1` の両方で通した）。

## should-fix の扱い

- specs の古い名前（execution-model.md の周期、native-core.md の `RuntimeConfig::*`）: 直した
- 状態 None のあいまいさ: 「None は状態が無い（復元で load_state_dict を呼ばない）」を契約として docstring に書き、テストを足した（D23 の木では状態は dict なので、最終形でも同じ意味）
- 既定のワーカー数の規則のオラクル: `os.process_cpu_count()` を使った上限の検査を足した（quota は Python から見えない）
- Lifecycle の単体テスト: 遷移ごとに 9 本足した
- `Runtime::start` の分割: `lanes`、`Shared::new`、`spawn_workers` に分けた
- 4.3 の「子は自分の型のメソッドだけ」との食い違い: 設計書の側を実態に合わせる必要がある（orchestrator へ）
- worktree の後始末: 計測を終えてから外す
