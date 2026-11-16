# code-simplifier: manual-run-simulation のレビュー指摘の反映

対象: code-reviewer / algorithm-reviewer（どちらも approve）の should-fix と nit。公開 IF は変えていない。docs/・CHANGELOG.md・examples/simulation.py・CLAUDE.md・.claude/・memory/（agents 以外）は触っていない。

## 変更

- `src/mujou/execution.py`
  - `ManualRun.step`: 「ホストはステップ前の値を読む」を `nodes` なしの段落へ移した（cr-1）。逐次では各ノードの送信が更新の終わりに見えることを足した。Raises の「送信は公開しない」を「Without nodes」に限定（cr-1）。一致条件を「ノード間の流れに閉路が無ければ受け手を送り手より先に更新する順で一致、逆流があれば一般には一致しない」に（S4）。自分につないだポートへの自分の送信は同じ更新で見えないことを 1 文（N1）
  - `ManualRun` / `run_manually`: 「bit for bit, every time」に「with deterministic nodes that share no Python objects」を添えた（cr-6）。dt の文を「最後の更新からステップした時間」に合わせた（cr nit 1）
- `crates/mujou-core/src/slot.rs`: `Staging::open` / `commit` を `pub(crate)`（bindings と結合テストは `new` だけを使う）。`commit` を `mem::take` から `drain(..)` に（容量を使い回す。lock の順は pending → staged → value で、`stage` は staged を放してから pending を取るので入れ子にならない）
- `crates/mujou-python/src/slot.rs`: `PyStaging` を `PyOutlet` の struct と impl の間から、`PyOutlet` の `#[pymethods]` の後ろへ移した。`just stub` の出力は変わらない
- `tests/mujou/pc/test_layer.py`: `TestJacobi` → `TestFromTheOutput`、テスト名 `..._is_the_reference_update`、`from_next` の docstring とコメントを「前向きは前のステップ、credit は同じステップ」に（S1）
- `tests/mujou/pc/_reference.py`: `Reference` の docstring とループ前のコメントを同じ意味に（S1）
- `specs/execution-model.md`: `nodes` を渡す step の一致条件（DAG / トポロジカル順の逆 / 一般には）（S4）。送信の保留に自分の送信が見えない行（N1）、ホストの読み出しを全ノードのステップに限定（cr-1）。オラクルの節を「拡大系の Jacobi。送り手の更新後の状態だけで決まるなら陽的 Euler、そうでなければ 1 メッセージ 1 ステップ遅れの離散化」に（S3）、pc のオラクルは出力側からの逐次の混合であることを 1 項目（S1）
- `specs/pc.md`: テスト 2 を「出力側からの逐次」に、45・50・67・68・71 行の測定の「Jacobi」を「出力側からの逐次」に（S1）
- `specs/research/pc-experiments.md`: 73・98 行を同様に。14 行の「同期 Jacobi での MNIST」は u 形 / h 形の比較で、どの順で回したか確かめられないので残した
- `specs/scenarios.md`: K19 のテスト名を `test_a_node_of_another_run_moves_no_time`、K32 に `test_an_exception_in_a_step_stops_every_node`（cr-3、nit 5）
- `specs/open-decisions.md`（mujou.pc）: S2（`step(h)` で credit が 2 ステップに 1 ホップ）を事実と 2 案で 1 項目

## 前提にしたこと

- pc.md の scratch の測定（`credit/credit.py` など）で「Jacobi」と書いていたものは、当時の用法（`from_next` の逐次と `Reference` 型の混合）で測ったと見なした。scratch のコードは repo に無く確かめていない

## 検証

cargo fmt --check / clippy -D warnings / cargo test --workspace: pass。ruff check / format --check: pass。pytest（slow 以外）928 passed / 63 skipped。pyright 0 errors。pre-commit（変更したファイル）pass。`just stub` は差分なし。docs-build は docs/ を他の担当が編集中のため未実行
