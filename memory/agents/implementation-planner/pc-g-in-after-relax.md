# pc.Layer: credit g_in を緩和した後の δ から送る

ブランチ: `feature/2026-11-16/pc-g-in-after-relax`（main = PR #118 マージ後）

## 概要

`pc.Layer.on_update` の順序を「vjp（更新前の δ）→ g_in を送る → relax_」から「relax_ → vjp（緩和した後の δ）→ g_in を送る」に変える（`specs/open-decisions.md` の案 (b)、ユーザーの判断 2026-11-16）。受け取った credit をその更新のうちに前のノードへ渡せるので、どのスケジュールでも credit の戻りが 1 ホップあたり 1 ステップ（1 周期）縮む。`ManualRun.step(dt)`（Jacobi）では 2 ステップに 1 ホップだったものが 1 ステップに 1 ホップになる。不動点、vjp の回数、公開 IF、状態のキーは変わらない。

## 公開インターフェース案

公開 IF は変えない（シグネチャ・引数・状態のキーは同じ）。振る舞いの変化を docstring・docs・CHANGELOG で知らせる。

src（内部）:

- `pc.Layer.on_update(self, dt: float) -> None`: シグネチャは同じ。順序だけを変える（下の式）
- `pc.Layer._predict(self, act: Activity, learn: bool) -> tuple[torch.Tensor, list[torch.Tensor]]`: シグネチャは同じ。relax_ の**後**に呼ぶ。docstring の "from the ``delta`` before this update" を "from the ``delta`` this update has relaxed" に直す
- `pc.Output`、`functional.vjp` / `loss_vjp` / `relax_`、`_GraphedVjp`、`pc.relax`、`LearnClock` は変えない

tests（spec-test-author と plan-implementer はこの形を前提にする）:

- `Reference.__init__(self, layers: Sequence[nn.Module], head: nn.Module, widths: Sequence[int], learn_interval: float | None = None, *, jacobi: bool = False) -> None`
  - `jacobi=False`: 出力側からの逐次（`step(h, *chain.from_next())`）。層 i は、層 i+1 がこのステップで送った `g_in` を読む
  - `jacobi=True`: `ManualRun.step(h)`。層 i は、層 i+1 が前のステップで送った `g_in` を読む（ステップの初めの `self.g_in[i + 1]`。ステップの初めに複製しておく）
  - μ は両方とも、前のステップの `self.h[i]` から作る（今と同じ）
- `Reference.step(self, x, y, mask, time: int, dt: float, now: int, reset: torch.Tensor = KEEP) -> None`: シグネチャは同じ。層ごとの中身を次の順にする: tick → reset → `g`（上の 2 通り）で r、λ を緩和 → `delta = torch.where(mask_i[:, None], r + lam, 0.0)`（緩和の**後**）→ `torch.autograd.grad(mu, [h_in, *params], -delta)` → 学習 → `self.g_in[i] = g_in`。属性（`layers, head, clocks, r, lam, h, carried, g_in, y_hat, prediction_carried`）とその shape は変えない
- `layer_gradients(before: dict[str, Any], after: dict[str, Any], module: nn.Module) -> tuple[nn.Module, list[torch.Tensor]]`: θ、`h_in`、`mask` は `before` から（今と同じ）、δ は `after["nodes"]["l0"]["state"]` の `r + lam` から取る。学習は更新の最後なので、`after` の r、λ がその更新で使った δ そのもの
- `layer_update(before: dict[str, Any], after: dict[str, Any], module: nn.Module, elapsed: float) -> dict[str, torch.Tensor]`
- `adamw_first_update(before: dict[str, Any], after: dict[str, Any], module: nn.Module, elapsed: float) -> dict[str, torch.Tensor]`

## 理論上の対応

### 新しい更新（Layer の 1 回の on_update）

1. `act = prev.recv()`、`tick`。新しい入力なら、`reset` の立った行の r、λ を 0 にし、`relaxer.reset_` を呼ぶ（今と同じ）
2. `g = Σ next.recv()`。`relax_(r, lam, g, dt, tau, relaxer)` で $r' = r - \frac{dt}{\tau_h}\big((1+\frac{\tau_h}{\tau_r})r + \lambda + g\big)$、$\lambda' = \lambda + \frac{dt}{\tau_\lambda}r'$
3. $\delta' = \mathrm{where}(\mathrm{mask}, r' + \lambda', 0)$ で vjp を 1 回取る: $\mu = f(h_{\rm in};\theta)$、$g_{\rm in} = -J_h^\top\delta'$、学習するなら $g_\theta = -J_\theta^\top\delta'$。`g_in` を送る
4. `mu` を `Activity` で送る
5. 学習するなら `optimizer.step_(g_theta / B, elapsed)`

- **μ は変わらない**: μ は $h_{\rm in}$ と更新前の θ だけで決まり、δ に依らない。前向きのメッセージは今とビット単位で同じで、変わるのは `g_in` と `g_theta` だけ
- **コスト**: vjp は 1 回のまま（forward と backward が 1 回ずつ）。`g_in` と `g_theta` を同じ δ' から取るので、backward は増えない。1 回の更新の計算量は変わらない
- **g_θ も δ' から取る（決定）**: 公式実装（重みの勾配に使う λ は最後の dual 更新の前の値）からずれる。g_in と g_θ で別の δ を使うと backward が 2 回になるので採らない。学ぶ δ がより緩和しているので、ずれはよい方向。不動点では δ' = δ なので、不動点での θ の 1 回の更新が SGD に一致する性質（`TestFixedPoint::test_one_update_at_the_fixed_point_is_sgd`）は保たれる
- **入力を初めて見た更新では学ばない規則は残す**: その更新の δ' は、新しい入力をまだ見ていない次のノードが送った g で緩和した値で、新しい入力の credit ではない。理由の文言を「δ が前の入力のもの」から「緩和に使う g が前の入力のもの」に直す

### 不動点・BP 一致・α = 0

- 不動点（$r = 0$、$\lambda + g = 0$）では relax_ が r と λ を動かさないので、δ' = δ になる。送る値も学ぶ値も今と同じで、λ が BP の credit になる性質はそのまま
- α = 0（`lam=inf`、`r=inf`）: λ ≡ 0、$r' = r - \frac{dt}{\tau_h}(r + g)$ で、不動点 $r = -g$ は同じ
- mask が False の行の δ を 0 にすることは変えない（その行の不動点は λ = 0）
- 事前の確認: 順序を差し替えた Layer で `tests/mujou/pc` を slow 込みで回した。`TestFixedPoint`、`TestDeepChain`（増幅する線形の鎖 1.25 × 16 と ResBlock 8 ノード）、`TestPcLimit`、`TestSchedules`、`TestScalar`、`TestResidual`、`TestWithoutTeacher`、`TestLearnTime`、`test_chain.py` は変更なしで通った。落ちたのは、参照実装と「更新前の δ」を前提にした次の 12 本だけ: `TestFromTheOutput` 4 本、`TestLearnInterval` 3 本（`test_theta_moves_k_steps_after_each_input_by_the_elapsed_time`、`test_a_first_input_written_at_time_zero_is_learned`、`test_an_input_that_changes_before_it_is_learned_drops_its_time`）、`TestMaxDt::test_a_long_dt_counts_max_dt` 2 本、`TestUnlabeled` 2 本（`test_no_target_skips_the_update_and_drops_the_time`、`test_a_sample_without_a_target_sends_back_nothing`）、`TestOptimizer::test_the_first_update_with_adamw_is_its_closed_form`

### 安定性（project_pc_jacobian_stability.md の三角の議論）

- θ を固定すると、前向き（μ）は h_in だけで決まる feed-forward になる。後ろ向きでは、δ_i の次の値は δ_i と g_{i+1} = −J_{i+1}^⊤ δ_{i+1} だけで決まり、δ_{i+1} は δ_i に依らない。g_in を緩和の後に作っても、ノード i の (r, λ) の更新は i+1 の状態にしか依らず、依存は後ろ向きの一方向のまま。ステップ写像は（遅れを状態に入れた拡大系でも）ブロック三角で、固有値は各ノードの局所の 2×2（Jury 条件 $ak < 2$、$a(2k + b) < 4$）のものだけになる。よって「局所の陽的 Euler が安定ならヤコビアンによらず安定」は保たれる
- 変わるのは、遅れ（べき零の結合）が 1 段減ることだけ。臨界減衰の既定（$a = 1/2$、$b = 1$、$k = 2$）では、その更新のうちに δ' = −g になる。$g_{\rm in} = J^\top g$ をそのまま前へ渡すだけなので、行き過ぎは増えない

### credit の速さ（理論値と実測）

1 ステップ（1 周期）に、全ノードが 1 回ずつ更新する場合を考える。ホップ数は Input から Output までの接続の数（L 層の鎖で L + 1）。メッセージは、受け手が次に更新するまで待つ。

| スケジュール | 前向き 1 ホップ | 後ろ向き 1 ホップ（今） | 後ろ向き 1 ホップ（新） | 往復（今） | 往復（新） |
| --- | --- | --- | --- | --- | --- |
| `step(h)`（Jacobi） | 1 | 2 | 1 | 約 3 × hops | 2 × hops |
| 出力側からの逐次 `from_next` | 1 | 1 | 0（同じステップで全段） | 約 2 × hops | hops + 1 |
| 自由実行（周期が等しく位相が固定） | T − φ | T + φ | φ | 2 × hops × T | 約 hops × T |

- 自由実行の φ は「受け手の位相 − 送り手の位相」mod T。周期が等しく位相が固定の自由実行は、位相の順に逐次で回す `step(h, *order)` と同じになる（各ノードがその時点の最新値を読む）。前向き T − φ と後ろ向き φ の和はどの辺でも T なので、往復は位相によらず約 hops × T になる（今は 2T × hops）。ジッタのある実際の自由実行で言えるのは、1 ホップあたり最大 1 周期という上限だけ
- 実測（順序を差し替えた scratch。5 層 + Output = 6 ホップ、widths `[4, 5, 5, 5, 5, 5]`、λ が初めて 0 でなくなるステップを入力側から並べた）
  - `step(h)`: 今 [16, 14, 12, 10, 8] → 新 [12, 11, 10, 9, 8]（層 i は 2 × hops − i）
  - `from_next`: 今 [11, 10, 9, 8, 7] → 新 [7, 7, 7, 7, 7]（hops + 1）
  - `from_prev`: 今 [10, 8, 6, 4, 2] → 新 [6, 5, 4, 3, 2]（L + 1 − i）
  - 固定のランダム順 6 通り（位相が固定の自由実行に当たる）: l0 は今 10〜11 → 新 6〜7（hops + {0, 1}）
- BP の credit から相対 1% 以内に全層が着くステップ（臨界減衰 `TimeConstants(h=2H, lam=H, r=2H)`、dt = H、6 ホップ、linear と ResBlock で同じ値）: `step(h)` は 22 → 18、`from_next` は 17 → 13。λ は 1 歩ごとに残りの半分ずつ近づくので、波面が届いてから約 7 ステップかかることは変わらない

### learn_interval の目安

目安の式 `2 × hops × period` は**変えない**。根拠の説明だけを、「credit は 1 周期に 1 ホップ」から「前向きも後ろ向きも 1 ホップあたり最大 1 周期（`step(dt)` ではちょうど 1 ステップ）。往復は `step(dt)` で 2 × hops × period、周期の等しい自由実行で約 hops × period」に書き直す。最も遅いスケジュール（`step(dt)`）でも目安どおりに credit が戻りきるので、algorithm-reviewer の S2（`step(dt)` で目安を守っても、credit が戻りきる前に学ぶ）が解消する。examples の S と `learn_interval` は変えず、コメントだけを直す（決定）。

### ノードの局所性・非同期性

- ノードが読むのは、今までどおり自分の状態と隣のスロットだけ。メッセージの型・ポート・スケジューラは変えない。手動の実行やバリアに頼る変更ではない（どのスケジュールでも速くなる）
- `specs/execution-model.md` の「オラクルとしての使い方」は、`g_in` を「更新前の状態から作る」値の例に挙げているので直す。新しい `g_in` は送り手の更新後の δ' から作る。ただしヤコビアンは同じ更新で読んだ $h_{\rm in}$（中継した値）で取るので、`step(h)` はまだ陽的 Euler そのものではない（μ の中継による遅れは残る）

### CUDA graph

- `_GraphedVjp` は `h_in` と `delta` をバッファにコピーしてから再生するだけで、δ をいつ作ったかを知らない。記録の中身、キー（params の有無）、再生の回数は変わらない。返すバッファ（mu、g_in、g_theta）を同じ更新の中で使い切ることも変わらない（vjp の直後に g_in と mu を送るようになるだけ）
- `TestCudaGraph::test_the_replayed_update_is_the_eager_one_bit_for_bit` は両辺が同じ新しい順序で回るので、テストの変更は要らない。このコンテナには GPU が無く skip されるので、CI の cuda ジョブで通ることを確かめる（決定）

## 実装ステップ

各ステップの検証方法は括弧の中に書く。spec-test-author（ステップ 3）と plan-implementer（ステップ 1〜2）は並列に進められる。

1. `src/mujou/pc/layer.py`
   - `on_update`（237-252 行）: `g = sum(...)` と `relax_` を `_predict` の前に移す。他の段（tick、`_reset`、`_send`、`optimizer.step_`）は変えない
   - `_predict` の docstring（265-266 行）を直す
   - クラスの docstring を直す
     - 52-54 行: "delta follows -g in one update, so the credit moves one node down in each period" を「δ はその更新で −g になり、同じ更新のうちに前へ渡す。credit は 1 ホップあたり最大 1 周期」に
     - 63-76 行の段の列挙: 2 と 3 を入れ替え、"with the delta before this update" を "with the delta it has just relaxed" に
     - 80-90 行: 入力を初めて見た更新で学ばない理由を「その更新が緩和に使う g は前の入力のもの」に。往復の目安の説明を上の「learn_interval の目安」に合わせる
     - 174-176 行: `learn_interval` の引数の説明
   - （検証: `uv run pytest tests/mujou/pc -q`。ステップ 3 の前は、上の 12 本だけが落ちる。それ以外が落ちたら止めて報告する）
2. 周辺の docstring と examples
   - `src/mujou/pc/chain.py`: 86-87 行と 91-94 行の docstring（往復の目安と「1 周期に 1 ノード」）を同じ文言に
   - `examples/pc_merge.py` 59-71 行の `Merge.on_update`: 同じ順序（relax_ → vjp → g_in を送る）に直し、docs の「Nodes of your own」の手順と合わせる。examples はテストで回さないので、`just type` で確認する
   - `examples/cifar10/resnet.py` 10-16 行と `examples/cifar10/main.py` 22-23 行・90 行: credit の速さを述べたコメントだけを直す（S = 12 などの値は変えない）
3. tests（spec-test-author）
   - `tests/mujou/pc/_reference.py`: `Reference` を上のシグネチャにする。docstring の「each layer reads … the credit the next node sent in this one」を、`jacobi` の 2 通りに分けて書く
   - `tests/mujou/pc/test_layer.py`
     - `layer_gradients` / `layer_update` / `adamw_first_update` を `(before, after, ...)` の形にする。呼び出し元 8 本（`TestLearnInterval` 3 本、`TestMaxDt` 2 本、`TestUnlabeled` 2 本、`TestOptimizer` の AdamW 1 本）に `after` を渡す
     - `test_a_sample_without_a_target_sends_back_nothing` の「行 1 の δ が 0 でない」の検査は、`after` の r + λ で見る
     - `TestFromTheOutput` を `TestReference` に改名し、`schedule` を `["from_next", "jacobi"]` で parametrize する。`from_next` は `run.step(H, *chain.from_next())` と `Reference(jacobi=False)` の組、`jacobi` は `run.step(H)` と `Reference(jacobi=True)` の組。比べる量、許容誤差（1e-12）、入力の切り替え（k = 20、keep / reset）は今と同じ
     - 新しいテストを足す（下の「テスト観点」）
   - （検証: `uv run pytest tests/mujou/pc -q` が全部通る。ステップ 1 の前のコードでは、新しい `TestReference`、`TestCreditSpeed`、`TestCreditAfterRelax` が落ちることを確かめる。落ちれば、テストが新しい振る舞いを見ている）
4. 文書を直す（下の「文書の変更」）。`just docs-build` で確かめる
5. 速度と精度を確かめる（下の「速度の予算」）
6. `just run`（format → test → type → docs-build）を通す

## テスト観点

- 正常系:
  - `TestReference::test_stepping_is_the_reference_update[schedule × learn_interval × reset]`: 新しい参照実装と 1e-12 で一致する（r、λ、module、送った h・y・mask・reset、送った g_in、y_hat）。`jacobi` で、`step(h)` そのものをはじめてオラクルと突き合わせる
  - `TestCreditAfterRelax::test_g_in_is_made_from_the_relaxed_delta`（新規、test_layer.py）: 鎖を数ステップ回した後の 1 ステップ（`from_next`）を見る。`l0` が送った g_in は、`pc.functional.vjp(module_before, h_in_before, lambda _: torch.where(mask, r_after + lam_after, 0.0))` の g_in と 1e-14 で一致する。更新前の `r_before + lam_before` から作ったものとは一致しない（両者が 1e-6 以上離れる時点を選ぶ）
- 異常系: 新しい分岐は無い。mask が False の行が g_in を 0 で返すことは、緩和の後でも成り立つ（既存の `test_a_sample_without_a_target_sends_back_nothing`）
- エッジケース:
  - 入力を初めて見た更新では学ばない（既存の `TestLearnTime::test_a_node_does_not_learn_in_the_update_that_first_sees_an_input`。変更なしで通る）
  - `dt = 0`（`ManualRun.step(0.0)`、`Learning.step`）: relax_ は恒等で、δ' = δ になる。`Learning` を使う既存のテストが通ること
  - reset の立った行: リセットの直後の更新では、次のノードがまだ前の入力で送った g で緩和する。既存の `TestReference[reset]` が参照実装と一致することで押さえる
- 数値的オラクル:
  - BP 一致、SGD 一致、α = 0、どの順でも同じ不動点: 既存の `TestFixedPoint`、`TestDeepChain`、`TestPcLimit`、`TestSchedules`、`TestScalar` が変更なしで通る
  - `TestCreditSpeed::test_the_credit_moves_back_one_hop_per_step[schedule]`（新規、test_layer.py）
    - 設定: widths `[4, 5, 5, 5, 5, 5]`（L = 5、hops = 6）、`make_modules("linear", "tanh", widths)`、`learn_interval=math.inf`、`VirtualTimer`、書き込みは 1 回
    - 判定: 各ステップの後に、`state_dict` の `l{i}` の λ が初めて 0 でなくなるステップ（1 始まり）を集め、理論値と完全に一致させる
    - 期待値: `jacobi`（`run.step(H)`）は `2 * hops - i`、`from_next` は全層 `hops + 1`、`from_prev` は `L + 1 - i`。どれも scratch の実測（上表）と同じ
  - CUDA graph のあり・なしのビット一致（既存の `TestCudaGraph`）: CI の cuda ジョブで
  - `tests/mujou/pc/test_chain.py`（`Chain` と手組みの一致、workers 1 と 4 の一致）: 変更なしで通る

## 文書の変更

- `specs/pc.md`
  - 「上へ μ を送る」の臨界減衰の項（66 行）の末尾「credit は 1 周期に 1 ホップ進む」を、新しい速さに書き直す
  - 新しい節「g_in を緩和の後に送る（2026-11-16）」を足す。中身: 決定と理由（S2 の実測。案 (a) を採らなかった理由を 1 行）、新しい順序、μ が変わらないこと、g_θ も δ' から取ること、三角の議論、上の速さの表と実測
  - 「on_update の段」（158-164 行）: 段 2 と段 3 を入れ替え、「**更新前の** δ」を「**緩和した後の** δ」に
  - 「学習の時刻」180 行の理由と、「learn_interval の目安」196 行の根拠を直す
  - 公式実装との比較表 364 行の「重みの勾配の λ」: 「VJP を取ったときの更新前の値」を「その更新で緩和した後の値（最後の dual 更新の後）」に
  - テスト 2（374 行）を、両スケジュールで参照実装と一致することに直す。新しいテスト（credit の波面、g_in が緩和後の δ から作られること）を項目に足す
  - 過去の測定の記述（58 行、270 行など）は直さない
- `specs/execution-model.md` 262 行・266 行: `g_in` を「更新前の状態から作る」値の例から外す（μ の中継の例は残す）。PC の参照実装が `from_next` と `step(h)`（`jacobi=True`）の両方を持つことを書く
- `specs/open-decisions.md`: 43 行の「`step(h)` での credit の速さ」を消す（決定は specs/pc.md と memory に書く）
- skill `pc-alm`（`.claude/skills/pc-alm/SKILL.md`。編集は skill `edit-dot-claude` の手順で）
  - 7.1 の 126 行: `g_in` を、更新前の状態から作る値の例から外す
  - 7.2 の「状態はずれ r で持ち…」の「credit は 1 周期に 1 ノード戻る」を直す
  - 7.2 に箇条「g_in は緩和した後の δ から（2026-11-16）」を 1 項足す
- docs（en と ja を同じ内容で）
  - `topics/pc.md`
    - 57 行（周期の行）
    - 59-60 行（見せる時間と `learn_interval`。式は同じで、根拠だけを直す）
    - 68 行（credit の往復: `step(dt)` で 2 × hops × period、自由実行ではその約半分）
    - 71 行（この値にする理由）
    - 147 行（CIFAR の「1 周期に 1 ノード」。往復 8 周期の数字は目安の式のままなので残す）
    - 222-235 行の「Nodes of your own」: 手順 3 と 4 を入れ替える（relax_ を先にし、vjp は緩和した δ で取る）
  - `topics/pc-concepts.md`
    - 38 行
    - 119 行の Key idea を "read → `tick` → `relax_` → `vjp` → send `g_in` → send `Activity` → `optimizer.step_`" に
    - 122-129 行の表: 段 2 と段 3 を入れ替え、「The `delta` of step 2 is made from `r` and `lam` before step 3」を直す
    - 141-143 行の g_in の流れ方: "its next update" を、同じ更新で送ることに
    - 170 行（new の更新で学ばない理由）
    - 195 行
  - `tutorial/5-learn-with-pc.md` 28 行
  - MNIST の精度の数字（`examples/mnist/main.py` の docstring、`docs/*/topics/pc.md`、`specs/pc.md` 211 行）は、ステップ 5 の測定で変わったときだけ直す
- `memory/`
  - 新しいファイル `project_pc_g_in_after_relax.md` を作る（決定、理由、速さの表、g_θ も δ' から取ること、examples の値を変えないこと、CUDA graph は CI の cuda ジョブで確かめること）。`MEMORY.md` から 1 行リンクを張る
  - `project_pc_module.md` 20 行の「credit は 1 周期に 1 ノード戻る」を直す
  - `project_pc_jacobian_stability.md` に、三角の議論が保たれることを 1 行足す
- `CHANGELOG.md` の `## [Unreleased]` → `### Changed` に 1 項足す: "`pc.Layer` sends the credit `g_in` back, and learns, from the `delta` it has just relaxed in the same update, instead of from the `delta` before it: the credit goes back one hop per step under `ManualRun.step(dt)` (it took two), and about one period per hop forward and back together in a free run. The fixed point and the cost of an update are the same; the trajectories differ."

## 速度の予算

- 1 回の更新のコストは変わらない（vjp は 1 回、メッセージの数も大きさも同じ）
- `benchmarks/pc_dense.py` を、main（PR #118 の後）と変更後で、同じ機械・同じ引数（`uv run python benchmarks/pc_dense.py`、既定の repeats）で回す。manual の µs/step の中央値と、free の更新回数/秒が揺れ（±5% 程度）の内にあることを確かめる。結果は `specs/research/benchmarks.md` に 1 行で記録する（変化が無ければ「変化なし」と書く）
- 自由実行で credit の往復に要る周期は約半分になる。同じ S と `learn_interval` なら、より緩和した δ から学ぶ。BP の 10 倍まで・time scale 1 以上という予算に対しては改善の向きになる。S を縮めて速くするのは別のタスクにする
- 精度: `examples/mnist/main.py` を既定（3 epoch、seed 0、約 2 分）で 1 回回す。テスト精度が 97.56% から大きく下がらない（目安は 97.3% 以上）ことを確かめ、値を `specs/research/pc-experiments.md` に記録する。CIFAR-10 は回さない

## 想定リスク・トレードオフ

- 既存の数値（参照実装との一致、`layer_update` を使うテスト）は、すべて取り直しになる
- 軌道が変わるので、保存した状態を復元して続けた結果も前の版とは違う。状態のキーは同じなので読み込みはできる。リリース前なので互換は持たない（`feedback_no_compat_before_release.md`）
- リセットした行は、その入力を初めて見た更新で、次のノードがまだ前の入力で送った g で緩和し、その結果を g_in として前へ送る。今も 1 更新後には同じ値を送っていたので、新しい漏れではなく 1 更新早まるだけ。リセットした行の g を 0 にする案は設計の変更になるので採らない
- g_θ が δ' から取られるので、公式実装（dual 更新の前の λ）からずれる。不動点では同じ
- 自由実行の「hops × T」は、周期が等しく位相が固定のときの値。周期の違うノードやジッタがあるときに言えるのは、1 ホップあたり最大 1 周期という上限だけ
- 採らなかった案:
  - (a) 目安をスケジュールごとに書き分ける: 遅い `step(dt)` が既定のまま残る
  - g_in だけを δ' から、g_θ を δ から取る: backward が 2 回になる

## 確認事項

なし（2026-11-16、orchestrator 経由で次の 3 点が決まった）。

1. g_θ も緩和した後の δ' から取る
2. examples の S と `learn_interval` は変えず、コメントだけを直す
3. CUDA graph のビット一致は、CI の cuda ジョブで確かめる

## 参照

- 指摘: `memory/agents/algorithm-reviewer/manual-run-simulation.md` の S1・S2
- 現状のコード: `src/mujou/pc/layer.py:237-276`、`tests/mujou/pc/_reference.py:188-290`、`tests/mujou/pc/test_layer.py:157-243, 697-748`、`examples/pc_merge.py:52-74`
- 設計: `specs/pc.md` の「上へ μ を送る」「on_update の段」「learn_interval の目安」、`specs/execution-model.md` の「オラクルとしての使い方」、skill `pc-alm` の 7.1・7.2、`memory/project_pc_jacobian_stability.md`、`memory/feedback_pc_speed_budget.md`、`memory/project_no_cuda_graph.md`
