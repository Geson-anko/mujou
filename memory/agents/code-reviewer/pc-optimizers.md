# pc-optimizers レビュー

対象: worktree `.claude/worktrees/feature-pc-optimizers` の未コミットの変更（src、tests、examples、benchmarks/pc_dense.py、specs、memory、`docs/en/api/pc.md` の members）。学習則の数理と局所性は algorithm-reviewer の担当なので、ここでは仕様準拠・バグ・規約・テスト品質・公開 IF・docstring・specs と実装の整合を見た。

## verdict: approve

must-fix は無い。計画書の公開 IF（`Optimizer` / `BoundOptimizer` / `SGD` / `AdamW`、`TimeConstants(*, h, lam)`、`Layer` / `Output` / `Chain` の必須の `optimizer=`、`functional.learn_` の削除、`__all__`、`LAYERS`）、離散化の順序、K31 の例外とメッセージ、状態のキー `"optimizer"` は計画どおりに実装されている。下の should-fix は orchestrator が採否を決める。

## must-fix

なし

## should-fix

1. **`eps=0` を通すと、勾配が 0 の要素で θ が NaN になる**
   - 対象: `src/mujou/pc/optim.py:238-239`（`_check_non_negative("AdamW", "eps", ...)`）、`:272-273`
   - 問題: その要素の勾配がそれまでずっと 0 だと、`m = v = 0` のままなので、`denom = sqrt(0)/sqrt(c_v) + 0 = 0` になり、`addcdiv_(m, denom)` が `0/0 = NaN` を θ に書き込む。死んだ ReLU の重みのように、勾配が厳密に 0 の要素はよくある。NaN はその後、鎖全体に広がる
   - 根拠: 計画書の例外の表と `TestChecks.test_adamw_takes_zero_eps_and_weight_decay` で `eps=0` は意図して許可されている。torch も同じ振る舞い。ただし K31 の「防ぐ失敗」は「NaN になる」で、それと食い違う。`eps` を `> 0` にして断るか、docstring に「`eps=0` は勾配が 0 の要素を NaN にする」と書くかを決めてほしい
   - 確信度: 高（挙動について）／ 方針は要判断

2. **`BoundOptimizer.step_` の `elapsed` の説明が `Tick.elapsed` の意味と合わない**
   - 対象: `src/mujou/pc/optim.py:41-42`（"The time since the last step, in seconds (``Tick.elapsed``)"）
   - 問題: `Tick.elapsed` は「前に学習したと数えてから、`max_dt` で打ち切った dt の和」。入力が学習の前に切り替わったときや、mask が無くて学習を飛ばしたときは、その時間を捨てている（specs/pc.md「学習の時刻」）。どちらの場合も `step_` の前回の呼び出しからの経過時間とは一致しない。モジュールの docstring（`optim.py:8-9`、"``elapsed`` is the time the step weighs"）のほうが正確で、2 か所の説明が揃っていない。自作の optimizer を書く利用者は、この説明を契約として読む
   - 根拠: `src/mujou/pc/clock.py` の `Tick` / `tick`、specs/pc.md の「$\Sigma\Delta t$ を捨てる」の 2 項目
   - 確信度: 中
   - 補足（algorithm-reviewer 向け、確信度: 低）: AdamW のモーメントは、捨てられた時間の分だけ実時間より減衰が少なくなる。この意味付けでよいかは数理の担当で確認してほしい

3. **chain のテストが「ノードごとに別に bind する」ことを区別できていない**
   - 対象: `tests/mujou/pc/test_chain.py:139-152`（`test_each_node_binds_the_optimizer_to_its_own_parameters`）
   - 問題: layer0、layer1、output の module はどれも param が 2 つなので、`len(moments) == len(list(module.parameters()))` は、どのノードの params に bind しても通る。実際に別々の bind を保証しているのは「run が始まった（K24 で断られなかった）」ことだけ。各 moment の shape を、そのノードの param の shape と比べれば区別できる
   - 根拠: skill `testing-strategy`（オラクルが仕様の違いを検出できること）
   - 確信度: 中

## nit

- `src/mujou/pc/optim.py:283-288`: `_BoundAdamW.load_state_dict` は shape を確かめずに `copy_` する。broadcast で黙って通りうる（K22 の「防ぐ失敗」と同じ型）。module の shape の不一致は、状態の順序で先に読み込む `"module"` が先に断るので、実害はほぼ無い（確信度: 低）
- `tests/mujou/pc/test_optim.py:288-298`: `test_loading_copies_the_moments` は `tree["v"]` も書き換えているのに、確かめているのは `m` だけ（確信度: 高）
- `tests/mujou/pc/test_optim.py:175-181`: `test_a_long_step_makes_the_mass_one` は `c_m` しか見ていない。`c_v == 1.0` も同じ 1 行で確かめられる（確信度: 高）
- `src/mujou/pc/layer.py:90-91`、`:132-134`: docstring の折り返しが不揃い（"The\n nodes read ..."、"To\n watch the node\n while it runs"）。編集で生じた短い行（確信度: 高）
- `examples/pc_regression.py:26-27`: コメントが "SHOWN / optimizer.tau\n# = 0.3." と式の途中で折れている（確信度: 高）
- `src/mujou/pc/layer.py:164`: Args の `optimizer` が "(``pc.SGD`` or ``pc.AdamW``)" で、自作の `Optimizer` も渡せることに触れていない。`Optimizer` の docstring では自作を勧めている（確信度: 中）
- `examples/pc_merge.py:81`: 内包表記の `g` が、直前の行の `g`（次のノードから届いた和）と同じ名前。スコープは別なので動作に問題は無いが、読み違えやすい（確信度: 中）
- `examples/feedback_alignment.py:40`: 設定のインスタンスを `SGD` と名付けたので、`SGD.bind(...)` がクラスメソッドの呼び出しに見える。計画書がこの名前を指定している（確信度: 低）

## 検証結果

- just run: pass（format / test 902 passed, 1 skipped（`test_rejects_cuda_without_cuda`。CUDA があるため）/ pyright 0 errors / docs-build strict OK）。`just run` の後も、作業ツリーに差分の増減は無い
- `test_optim.py` は 50 件すべて pass。CUDA のテスト、slow の `TestTorchAdamW`（subprocess）、`TestFreeThreaded`（3.14t）も実行された。`src/mujou/pc/optim.py` のカバレッジは 95%（未到達は Protocol の `...` の本体）
- 古い API の残り: `src tests examples benchmarks specs .claude CLAUDE.md` に `theta=` / `learn_(` / `tau.theta` は無い（`docs/` の本文は対象外。後の段で直す）
- 成果物の汚染（`</content>` など）: 無し
