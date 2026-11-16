# pc-g-in-after-relax: テスト

計画書: `memory/agents/implementation-planner/pc-g-in-after-relax.md`

## 書いた・直したテスト

- `tests/mujou/pc/_reference.py`
  - `Reference.__init__(..., learn_interval=None, *, jacobi: bool = False)`
  - 層ごとの順を tick → reset → 緩和 → `delta = where(mask, r' + lam', 0)` → vjp → 学習 → `g_in` に変えた
  - `jacobi=True` のときは、ステップの初めに `g_in` を複製して（`g_sent`）層 i に `g_sent[i + 1]` を読ませる。`jacobi=False` は、このステップで更新した `self.g_in` を読む
- `tests/mujou/pc/test_layer.py`
  - `layer_gradients(before, after, module)` / `layer_update(before, after, module, elapsed)` / `adamw_first_update(before, after, module, elapsed)`。θ・`h_in`・`mask` は `before`、δ は `after` の `r + lam` から
  - 呼び出し元 8 本を直した（`TestLearnInterval` 3、`TestMaxDt` 1 本で 2 ケース、`TestUnlabeled` 2、`TestOptimizer` 1）
  - `test_a_sample_without_a_target_sends_back_nothing`: 行 1 の δ が 0 でないことを `after` の r + λ で見る
  - `TestFromTheOutput` → `TestReference::test_stepping_is_the_reference_update[schedule × learn_interval × reset]`（`schedule` は `from_next` / `jacobi`）
  - 新規 `TestCreditSpeed::test_the_credit_moves_back_one_hop_per_step[jacobi|from_next|from_prev]`: λ が初めて 0 でなくなるステップが、jacobi は `2 * hops - i`、from_next は `hops + 1`、from_prev は `L + 1 - i` に一致する
  - 新規 `TestCreditAfterRelax::test_g_in_is_made_from_the_relaxed_delta`: 5 ステップ回した後の 1 ステップで、l0 が送った g_in は after の r' + λ' からの vjp と 1e-14 で一致し、before の r + λ からの値とは 1e-6 以上離れる

## 期待される失敗（古い順序 vjp → relax_ のとき）

scratch のプラグインで `Layer.on_update` を古い順序に差し替えて確認した（src は変えていない）。slow を除くと次の 20 件が落ち、ほかは通る。

- `TestReference` 8 件（両スケジュール）。根拠: 新しい参照実装は緩和した後の δ' から g_in と g_θ を作る（計画書の「新しい更新」の段 2・3）
- `TestCreditSpeed` 3 件。根拠: 計画書の「credit の速さ」の表と実測（新しい値）
- `TestCreditAfterRelax` 1 件。根拠: g_in = −J_h^T δ'
- 既存の 8 本（パラメタ込みで 8 件）: `layer_update` / `adamw_first_update` が after の δ' で g_θ を作るため。根拠: 「g_θ も δ' から取る（決定）」

plan-implementer が変更した現在の作業ツリー（新しい順序）では、`tests/mujou/pc` は slow 込みで 250 passed、6 skipped（CUDA）。

## 実装側への要求

なし。現在の `src/mujou/pc/layer.py` の変更で全部通っている。
