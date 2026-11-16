# manual-run-simulation レビュー

対象: ブランチ `feature/2026-11-16/manual-run-simulation` の未コミットの変更（docs/ と CHANGELOG は対象外）。計画書 `memory/agents/implementation-planner/manual-run-simulation.md`、実装ログ `memory/agents/plan-implementer/manual-run-simulation.md`、テストノート `memory/agents/spec-test-author/manual-run-simulation.md`。

## verdict: approve（should-fix の反映を推奨。must-fix なし）

コアの保留→commit は、見た範囲で競合が無い。根拠:

- `Staging::open` は control lock の下で、許可と `wake` より前（`control.rs` の `step`）。ノードは lock と condvar を経て起きるので、`send` の `is_open` は必ず開いた状態を見る
- `commit` は `open_steps == 0` を lock の下で確かめた後に呼ぶ。`end_step` は `record` の中、つまり hook が戻った後なので、commit の間に hook を走らせているノードは無い。停止・取得・終了はすべて制御スレッドを通る（`Runtime::stop` も job）ので、commit と入れ違わない
- 失敗したとき、`failed()` を見て待ちを抜け、`end()` が残りのノードの終わりを待つ。走っていた更新の `record` は `end_step` で数を減らすだけで、`Staging` は開いたままなので `on_exception` / `on_stop` の送信も公開されない
- 正常に終わるとき、`shutdown` は step の間にしか来ないので `Staging` は閉じており、`on_stop` の送信は公開される
- `Slot::stage` は `staged` の lock を放してから `pending` を取るので、lock が入れ子にならない。保留中のメッセージは Arc で保持されるため、pool はそのバッファを再利用しない
- GPU: `codec.to_core` は送る側で stream を同期してからコアに渡す（`wait=True`）。保留しても読み手の `record_stream` は変わらない

## must-fix

なし。

## should-fix

1. **`src/mujou/execution.py` `ManualRun.step` の docstring（「While a step runs, the host reading a port from another thread gets its value from before the step」と Raises の「What the nodes sent in that step is not published」）**
   - 問題: どちらも、ノードを渡す逐次の step では成り立たない。逐次の step は 1 ノードのグループを順に回し、グループごとに commit する（`control.rs` の `step`、`mod.rs` の `Runtime::step`）。このため、ステップ中のホストの読み出しには前のノードがこのステップで送った値が見える。後のノードが例外を出しても、前のノードの送信はもう公開されていて、`subscribe` の待ち手には届く
   - 根拠: `specs/execution-model.md` の「逐次の step(dt, *nodes) は 1 ノードのグループの列として同じ機構で動く（グループごとに開いて公開する）」。どちらの文も「Without nodes」の段落に移すか、「ノードなしの step では」と限定する
   - 確信度: 高 / 深刻度: 低（docstring の不正確さ）

2. **`specs/execution-model.md:261`（「オラクルとしての使い方」の冒頭）と `specs/pc.md:374`（テスト 2）、`tests/mujou/pc/test_layer.py:89-90`（`HandChain.from_next` の docstring「a Jacobi update」）**
   - 問題: この変更で「Jacobi」は `step(dt)` の意味、つまり「メッセージをステップ開始時の値で読む」ことになった。ところが PC の `Layer` は `g_in` を更新**前**の δ から作って送る（`src/mujou/pc/layer.py:266-275`）。このため `step(dt)` では、後ろ向きの信号が ODE の陽的 Euler（Jacobi）より 1 ステップ遅れる。`_reference.py` の「Jacobi」と一致するのは `from_next` の逐次で、`step(dt)` ではない（spec-test-author の実測でも 5 ステップ目から 1e-2 ずれる）。それなのに 261 行は「全ノードを固定の Δt で同期的に回すと、元の離散アルゴリズム（Jacobi 型の陽的 Euler）に一致するはず」と書いており、`step(dt)` を指すと読める。pc.md の「Jacobi 更新」、`from_next` の docstring とも食い違う
   - 根拠: spec-test-author ノートの「計画書と食い違った点 1」。直し方は 2 つある。261 行に「送る値を更新後の状態から作るノードに限る」と書き添えるか、pc 側の「Jacobi」を「出力側からの逐次」と言い換える。学習則の離散化の解釈に関わるので、`algorithm-reviewer` にも見せるとよい
   - 確信度: 中〜高 / 深刻度: 中（オラクルの意味が仕様の中で 2 通りになる）

3. **`specs/scenarios.md` K19 の行**
   - 問題: テストの参照先 `test_rejects_a_node_of_another_run` は、もう無い。旧 `TestManualRun` から消え、`TestManualRunStep::test_a_node_of_another_run_moves_no_time`（`tests/mujou/test_execution.py:1586`）に統合された
   - 根拠: `grep -rn test_rejects_a_node_of_another_run tests/` が 0 件
   - 確信度: 高 / 深刻度: 低

4. **`.claude/skills/locality-and-async/SKILL.md:72`（「同期実行（全ノードを順番に固定の Δt で 1 ステップずつ回す）は、非同期実行の特殊ケースのスケジュールとして実装する。同期専用の別経路を作らない」）**
   - 問題: 新しく足した 74 行と食い違う。「順番に」はもう `step(dt)` の意味ではない。また `Staging`（送信の保留）は、手動の実行だけにある経路である。CLAUDE.md と 74 行で例外として認めたので、この行も「ステップの許可と送信の保留以外は自由実行と同じ状態機械」などに直さないと、後のレビューで違反と読まれる
   - 根拠: 同じファイルの 74 行と `specs/execution-model.md` の「送信の保留と公開」
   - 確信度: 中 / 深刻度: 低

5. **`CONTRIBUTING.md:69,97`、`memory/project_run_node_and_write_time.md:8`**
   - 問題: 現行の規約として `mujou.testing` を挙げたまま残っている。CONTRIBUTING は計画書の手順 6 に入っているが、今回の依頼範囲（docs/ と CHANGELOG を除く）にも、実装ログの「残り」にも挙がっていない。memory の方は計画書の memory の一覧から漏れている
   - 根拠: `grep -rn "mujou\.testing"` の結果（`memory/agents/` と docs/、CHANGELOG を除く）
   - 確信度: 高 / 深刻度: 低

6. **`src/mujou/execution.py` `ManualRun` のクラス docstring と `run_manually` の docstring（「the result is the same, bit for bit, every time」）**
   - 問題: 無条件に言い切っている。実際には、ノードが自分の generator だけを使い、Python のオブジェクトを共有せず、決定的な演算だけを使う（CUDA の atomics などを使わない）ことが前提になる。並列化で、利用者のノードの共有状態が非決定性として表に出るようになった（計画書のリスクの節）。docs で条件を書く予定でも、API リファレンスに出る docstring に「given deterministic nodes」程度の条件を添える
   - 根拠: `specs/execution-model.md` の「並列の性質」の 3 項目め
   - 確信度: 中 / 深刻度: 低

## nit

1. `src/mujou/execution.py` `run_manually` の docstring「Every update sees exactly the ``dt`` that was stepped」: `step(dt, *nodes)` で渡さなかったノードは、次の更新で dt を積み上げた値を見る。`step` の docstring の「Each dt a node sees is the time since its last update」と表現を合わせる（確信度: 中）
2. `crates/mujou-core/src/slot.rs` `Staging::open` / `Staging::commit` が `pub` になっている。使うのは同じ crate の `runtime` だけ（結合テストは `Staging::new` だけを使う）なので `pub(crate)` で足りる。外から開閉できると、走っているノードと commit が競合する経路を API が許してしまう（確信度: 中）
3. `crates/mujou-core/src/slot.rs` `Staging::commit`: `std::mem::take` は commit のたびに `pending` の Vec を新しく作り直す。`drain(..)` で回せば容量を使い回せる。また、並列のノードはスロットに初めて送るたびに 1 つの `pending` の mutex を取り合う。ステップあたりスロット数回なので実害は小さい（確信度: 中）
4. `crates/mujou-python/src/slot.rs`: `PyStaging` が `pub struct PyOutlet` と `impl PyOutlet` の間に入り、Outlet の定義と実装が離れた。`PyOutlet` の impl の後ろか前に置く（確信度: 高、好み寄り）
5. `specs/scenarios.md` K32 のテストの参照先が `TestManualRunStep`（クラス名）だけで、ほかの行のようにテスト名（`test_an_exception_in_a_step_stops_every_node`）を挙げていない。実装ログも直す必要に触れている（確信度: 高）
6. `tests/mujou/test_init.py` `test_the_manual_run_is_no_longer_in_mujou_testing`: skill `testing-strategy` の「書かない: import 可能性の追試」に近い。計画書の観点にあるので残してもよい（確信度: 低）
7. `tests/mujou/test_execution.py` `test_a_subscription_gets_the_value_once_the_step_ends` の `received.get(timeout=0.05)` は、保留が無い実装でも、50 ms の間に届かなければ通ってしまう（偽陰性）。偽陽性は無いので flaky ではない。決定的な検出は `test_the_host_reads_the_values_before_the_step_until_it_ends` が担っている（確信度: 中）

## 確認した点（指摘なし）

- lifecycle の許可の数え方: `granted[i]` と `open_steps` は `grant` で同時に立てる。`end_step` は自分の印が立っているときだけ数を減らし、最後の 1 つで `changed` を立てる。時刻が進んでいないノードは、`wake` → `schedule` → `advance` の中で `end_step` を通る。制御スレッドは `granted()` を lock の下で見てから待つので、取りこぼさない。グループは `None` から作る全ノードの列か、1 要素の列だけなので、`debug_assert` の重複は起こらない
- テストの決定性: 並列のテストの参照値は手で書いた参照（`noisy_jacobi`、閉路の手計算）か、逆順の逐次と比べた bit 一致で取っている。待ち合わせは Barrier / Meeting / Event に timeout を付け、実時間の sleep に結果を依存させていない（Rust の `a_waiting_reader_wakes_at_the_commit_not_at_the_send` の sleep は、正しい実装なら結果を変えない）
- `_` prefix（`_start` / `_Kind` / `_RUN` / `_MANUAL_RUN`）、`class TestXxx`、英語の docstring は守られている
- 成果物の汚染（`</content>` など）: 変更したファイルと新しい memory に無い

## 検証結果

- cargo fmt --check: pass。cargo clippy --workspace --all-targets -D warnings: pass。cargo test --workspace: pass
- ruff check / ruff format --check（変更した .py / .pyi）: pass
- pytest（slow 含む、全体）: pass（951 passed, 88 skipped。CUDA は skip）
- just run: 実行していない。`just format`（`pre-commit run -a`）は自動修正でファイルを書き換え得るため、読み取りだけの検査（上記）に分けた。mdformat と codespell はこの環境の venv に無く、未実行。`docs-build` は未実行。レビュー中に docs/ と mkdocs.yml が並行して編集され始めており、`docs/ja/{concepts,examples,topics/time}.md` にまだ `mujou.testing` が残っている（対象外）。pyright は依頼文の既知の結果（0 errors）による
