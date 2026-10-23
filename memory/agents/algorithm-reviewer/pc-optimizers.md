# algorithm-reviewer: pc-optimizers（未コミットの worktree）

> 2026-10-22。対象は worktree `.claude/worktrees/feature-pc-optimizers` の `git diff` と新規の `src/mujou/pc/optim.py`・`tests/mujou/pc/test_optim.py`。基準は skill `pc-alm`・`locality-and-async`、計画書 `memory/agents/implementation-planner/pc-optimizers.md`、`memory/agents/orchestrator/pc-optimizers.md`、`specs/pc.md` の「重みの optimizer」。

## verdict

approve（確信度: 高）。should-fix は残っているが、どれも更新式の誤りではない。

SGD と AdamW の離散化は、連続時間形と torch の AdamW の両方に 1 項ずつ一致する。対象は expm1、c の更新順、decay の順、ε の位置、`elapsed == 0`。`torch.optim.AdamW` との差は変位比で 4e-16〜4e-15 で、foreach の False / True / 既定のどれでも同じだった。可変の Δt でも、刻みを細かくすると同じ流れに 1 次で収束する。1/B の位置、frozen param、跳び Δ（`step_` の後に f を再計算）、jumps の連鎖、mask が全 False で学ばないことは、旧実装から変わっていない。ノード外の量の読み取り、ノードをまたぐ計算グラフ、グローバル同期、基盤への焼き付けも無い。

## must-fix

なし。

## should-fix

1. **`eps=0` を許すと、勾配がちょうど 0 の成分が NaN になる**（確信度: 高）
   - 該当: `optim.py` の `_check_non_negative("AdamW", "eps")`、K31、`test_adamw_takes_zero_eps_and_weight_decay`
   - `pc.AdamW(eps=0.0)` で勾配 `[1, 0, -2]` を 1 歩進めると、θ は `[0.99, nan, 1.01]` になった。m = v = 0 なので 0/0 になる
   - PC ではこうした成分がよく出る。batch 全体で ReLU が死んだユニットの行とバイアス、常に 0 の入力画素の列、学習率の更新で一度も勾配を受けない成分などである。NaN は跳び Δ を通して `h` に入り、次のノードへ伝わる
   - K31 の目的は「NaN になるか、学習しない」を構築時に断ることなので、eps = 0 を通すのはその目的と矛盾する。torch も eps = 0 を受け付けるが、そこに合わせる理由は弱い
   - 代案: (a) `eps` を有限で > 0 にし、K31・テスト・spec を直す。(b) 0 を残すなら、docstring と spec に「勾配が 0 の成分は NaN になる」と書く。(a) を推す
2. **mask が全 False で学ばないことのテストが SGD でしか回らず、`step_` を呼んでも検出できない**（確信度: 高）
   - 該当: `test_layer.py` の `TestUnlabeled.test_no_theta_moves_while_every_mask_is_false` と `test_no_target_skips_the_update_and_drops_the_time` の step 4、`test_output.py` の `test_no_target_skips_the_update_and_drops_the_time`
   - mask が全 False だと δ は `torch.where(labeled, …, 0)` でちょうど 0 になり、g_θ もちょうど 0 になる。SGD の `p.sub_(0)` はビット単位で何も変えないので、ゲート `learn = tick.learn and mask.any()` を外す退行が起きてもテストは通る
   - AdamW では、`step_` を呼ぶと weight decay と持ち越したモーメントで θ が動く。このゲートが今回初めて効く不変条件（`specs/pc.md` の「不変条件」）になった
   - 代案: 上のテストを `[SGD, ADAMW]` で parametrize する。ラベルのある入力で 1 回学ばせてモーメントを 0 から離してから、全 False の入力に切り替えれば decay と m̂ の両方が見える
3. **AdamW の docstring の例の `tau=0.3` は、Adam としては学習率が大きすぎる**（確信度: 中〜高）
   - 該当: `optim.py` の `AdamW` の Example（`pc.AdamW(tau=0.3, tau_m=0.3, tau_v=30.0)`、`learn_interval=24e-3`）
   - η = 24e-3 / 0.3 = 0.08 である。Adam の 1 歩は勾配の大きさに依らず、成分ごとに約 η 動く。そのため 0.08 は、torch の典型値 1e-3 の 80 倍になり、784 入力の Linear の初期値の幅（約 0.036）も超える。SGD の `tau=0.3` をそのまま写した値に見える
   - spec と open-decisions は「推奨の時定数は未測定」としている。例が具体的な値を示すと、利用者はそれを推奨と読む
   - 代案: 換算式で書く（`tau = learn_interval / lr`。例: lr = 1e-3、24 ms ごとなら `tau=24.0`）。値を勧めない方針なら、例の `tau` を `learn_interval / 1e-3` の形にする
4. **`BoundOptimizer.step_` の `elapsed` を「前の step からの時間」と書いているが、`Tick.elapsed` はそうではない**（確信度: 高）
   - 該当: `optim.py` の `BoundOptimizer.step_` の Args（「The time since the last step」）、spec の同じ記述
   - `Tick.elapsed` は「max_dt で打ち切った dt の和」で、次の時間を捨てる。入力が学ばれずに切り替わったとき、mask が全 False で学習を飛ばしたとき（このときは `step_` が呼ばれない）、max_dt を超えたぶんである。そのため AdamW のモーメントと c は、捨てた時間のあいだ減衰も成長もしない（流れが止まっている扱い）
   - weight decay も、ラベルのある時間にだけ掛かる。疎にしかラベルが来ないストリームでは、実時間あたりの decay がラベルの割合に比例する
   - この扱いは「`elapsed` は流れが重みづける時間」として一貫しているので、直すのは記述だけでよい。docstring を「the time the step weighs (`Tick.elapsed`: the capped time since the last update of the parameters, without the time the clock dropped)」のようにし、spec に「捨てた時間はモーメントも動かさない」「decay はラベルのある時間にだけ掛かる」を 1 行ずつ足す
5. **Adam の正規化は、緩和しきっていない δ の残差を全幅の歩みに変える。この性質を spec に書く**（確信度: 中。理論上の指摘で、実測はしていない）
   - BP と一致するのは不動点だけである。SGD なら、緩和の残差 e による θ の誤差は η·e で、小さい残差の影響は小さい
   - AdamW では、BP の勾配がほぼ 0 の成分に残差 e が乗ると、v̂ が小さいので約 η·sign(e) 動く（ε = 1e-8 では ε/|e| の項もほとんど効かない）
   - そのため、`learn_interval` を往復の時間（`2 * hops * period`）より短くしたときの崩れは、SGD より大きく出うる
   - 代案: `specs/pc.md` の「重みの optimizer / 決めたこと」か `open-decisions.md` の「AdamW の推奨の時定数」に、「AdamW は勾配の大きさで歩みを縮めないので、緩和の残差と θ の跳び Δ が不動点の近くでも小さくならない。`learn_interval` の推奨（往復より長く）は SGD より強く効く可能性がある（未測定）」と 1 文足す

## nit

1. **段階 2 の「不動点（r = 0、λ = −g）は保てる」には条件が要る**（確信度: 中）
   - 保てるのは、勾配が 0 の点だけを停留点に持つ則（SGD、モーメンタム、weight decay の無い Adam）の場合である。h や λ に decay や leak を入れると、λ の不動点は BP の credit からずれる（dual leak と同じ）
   - さらに、h は min、λ は max の鞍点問題なので、適応的な則やモーメンタムを主・双対の両方に掛けると、線形化系の固有値が変わって振動・発散しうる（GAN の Adam と同じ種類の問題）
   - 「陽的な結合の安定性」がこれを含むならそれでよいが、「勾配の零点を停留点とする則なら保てる（decay・leak を除く）」と条件を書くと正確になる
2. **`AdamW` の docstring と spec は「θ は陽的 Euler」と書いているが、厳密には半陰的**（確信度: 高）。Adam の項は、その歩みで更新した後の m・v・c（区間の終わりの値）を使う（decay は区間の始めの θ に掛かる）。torch と同じで、Δt → 0 で 1 次の精度なので害は無い。「decay は陽的、Adam の項は更新後のモーメントで（torch と同じ順）」と書けば正確になる
3. **`elapsed == 0` の早期 return が `zip(strict=True)` の検査より前にある**（確信度: 高）。0 秒の歩みでは、勾配の数が違っても `ValueError` にならない。0 秒の歩みで何もしないことは正しく、害は小さい。直すなら、長さの検査を先に置く
4. **weight decay は bind したすべての param（バイアス、LayerNorm の重みを含む）に掛かる**（確信度: 高）。torch の param group に当たる除外の手段が無い。今回の範囲外でよいが、docs に一言あるとよい
5. **`test_a_long_step_makes_the_mass_one` の「θ が有限」は弱い**（確信度: 高）。η = 1e6·τm/τ ≈ 6.7e4 で decay は 1 − 6667 倍になり、有限なだけで意味のある値ではない。c がちょうど 1.0 になることを見るのが主眼なら、θ の検査は外すか `weight_decay=0` で行う
6. **`Layer` / `Output` の AdamW の初回の閉じた形のテスト**（`TestOptimizer`）は、ε = 1e-8 なのでほぼ sign(g) しか見ていない。1/B の誤りは ε/|g| ≈ 1e-6 程度の差としてしか現れない（rtol 1e-12 なので検出はできる）。1/B は SGD のテストが別に押さえているので実害は無い。ε を勾配と同じ桁にした `ADAMW` を使えば、量の誤りにも感度が出る

## 検証結果

- 照合の手計算と数値実験（`uv run python`、scratchpad のスクリプト）
  - c の漸化式: 1 − c_k = (1 − a)(1 − c_{k−1}) なので、1 − c = Π e^{−Δt_i/τ} = e^{−ΣΔt/τ}。可変の Δt でも連続解 c(t) = 1 − e^{−t/τ} と一致する（テスト `test_the_mass_is_…` も同じ式）
  - m: g を Δt のあいだ一定とすると、τ ṁ = g − m の厳密解は m + (1 − e^{−Δt/τ})(g − m) で、`lerp_(g, a)` に一致する。g 一定・0 から始めると m(t) = g·c(t) なので m̂ = g（不偏）
  - torch の `_single_tensor_adam`（decoupled）は `param.mul_(1 − lr·wd)` → `exp_avg.lerp_` → `exp_avg_sq.mul_.addcmul_` → `denom = sqrt(v)/sqrt(bc2) + eps` → `addcdiv_(m, denom, −lr/bc1)` の順で、順序・ε の位置とも一致する
  - `torch.optim.AdamW` との比較（subprocess、float64、200 歩）: 変位の相対差は 4.3e-16（Δt = 4 ms）と 3.8e-15（Δt = 32 ms、τ = 24、τv = 32、ε = 1e-8）。foreach が False / True / None のどれでも同じ
  - 連続時間への収束: 時間に依存する g(t) を 2 s 積分した。Δt = 1e-2 / 5e-3 / 2.5e-3 の誤差は 1.56e-2 / 7.9e-3 / 3.9e-3（1 次）で、c の誤差は 1e-15。Δt を ±80% の乱数にした可変の刻みでも、1.6e-2 / 9.2e-3 / 4.4e-3 で同じ流れに収束した
  - eps = 0 で勾配 0 の成分は NaN になった（should-1）。float32 で g = 1e-25（v がアンダーフロー）でも、既定の ε なら有限
- `uv run pytest tests/mujou/pc tests/mujou/test_init.py tests/test_layers.py`（slow を含む）: 213 passed。`TestTorchAdamW`・`TestFreeThreaded`（3.14t）・cuda のテストも実行され、pass した
- `uv run pyright`: 0 errors。`ruff format --check`・`ruff check`: 通過
- `just run` は実行していない。`just format`（`pre-commit run -a`）はファイルを書き換えうるので、レビューで worktree を変えないよう、format の代わりに check のみのコマンドを使った。cargo test と docs-build も未実行で、Rust には差分が無い

## 照合メモ

- **SGD**: τθ θ̇ = −g、g = g_θ/B。1 歩は `p.sub_(g, alpha=elapsed/τ)`。旧 `learn_` の `p.sub_(g_sum, alpha=elapsed/τ/B)` と同じ実数で、B が 2 の冪ならビット単位で一致する
- **AdamW の連続時間形**: τm ṁ = g − m、τv v̇ = g² − v、τm ċm = 1 − cm、τv ċv = 1 − cv、τθ θ̇ = −(m̂/(√v̂ + ε) + wθ)
  - 離散化: a = −expm1(−Δt/τ)、c ← c + a(1 − c) を先に、η = Δt/τθ、θ ← θ(1 − ηw)、m, v を lerp、θ ← θ − (η/cm)·m/(√v/√cv + ε)
  - Δt 一定なら β = e^{−Δt/τ}、c = 1 − β^t で torch の AdamW(lr = Δt/τθ) と同じ
  - 初回の 1 歩: cm = am、m = am g なので m̂ = g、v̂ = g²。θ(1 − ηw) − η g/(|g| + ε) で、τm・τv に依らない（テストの閉じた形と一致）
- **1/B**: `Layer._learn` と `Output.on_update` が `g / self._samples` を渡す（`math.prod(())` = 1）。λ はサンプルごとの credit のまま（g_in には 1/B を掛けない）
- **frozen param**: `self._params` は `requires_grad` の param だけで、`vjp(..., self._params)` の g_θ と同じ順・同じ長さ。decay も frozen な param に掛からない（テストあり）
- **跳び Δ**: `step_` の後に `moved = module(h_in)` を同じ `h_in` で計算し、jump = moved − mu、`mu ← moved`。次の更新で h += jump。discrete なら jumps += 1。AdamW では g = 0 でも decay と残ったモーメントで Δ ≠ 0 になるが、付け替えは r を保つので r・λ を乱さない。下流は学習のたびに付け替えるが、これは SGD で g ≠ 0 のときと同じ
- **mask が全 False**: `learn = tick.learn and mask.any()` で `step_` を呼ばないので、AdamW でも θ は動かない（ゲートは正しい。テストの感度は should-2）
- **LearnClock**: theta の規則を消しただけ。学ばないのは `learn_interval=math.inf` だけで、`_interval_ns = inf` なので `learn_at` に届かない。`TimeConstants.max_dt` の docstring の「updates of the parameters alike」は打ち切りが揃う意味で、正しい
- **局所性**: 束縛が読むのは、自分の param（参照）と、ノードが渡す勾配・elapsed だけ。`torch.no_grad()` は `with` に閉じている。設定は frozen dataclass で、可変の状態は無い。束縛を 2 ノードで共有すると K24 が断る。`mujou` の基盤は optimizer を知らない（`mujou.pc` の部品）。`torch.optim` も import しない（GIL のテストあり）。新しい `.backward()` / `autograd.grad` は無い
- **段階 2 の文案**: 符号 λ = −g は mujou の記法（τh ḣ = −(δ + g)、g は次のノードのエネルギーの偏微分）での不動点として正しい。条件については nit-1
