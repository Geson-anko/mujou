# code-simplifier: mujou.pc の 3 回目の監査への対応（2026-10-20）

algorithm-reviewer の 3 回目の監査（`memory/agents/algorithm-reviewer/pc-module.md` の末尾）の should-1・検討 1・検討 2 に、挙動と公開 IF を変えずに対応した。

## should-1: learn_interval の安定性の条件

- `Layer` の docstring（本文と `learn_interval` 引数）: 「目安」を「安定性の条件（これより短いと θ がほぼ止まっていても深い鎖が発散しうる）」に書き換え、理由（ほぼ毎回の更新で学習して epoch が進み、上のノードが毎回付け替える）を足した
- `specs/pc-module.md`
  - 「θ の更新間隔」に安定性の条件の箇条を足し、監査の数値（None・16H 安定、4H は credit の約 4 倍、2H・H は発散）を残した
  - 「跳びの連鎖（epoch）」に、短い間隔で u の形に戻る理由（付け替えは μ(h_in) − μ_last なので 1 周期分の h_in のドリフトも乗る）を足した
  - 未決定事項に根本の代案（Activity に Δ を載せ μ(h_in) − μ(h_in − Δ) だけ動く。未検証）を足した
  - テスト 14 の記述を None と 16 周期の両方に更新した
- テスト: `TestDeepChain::test_learning_in_every_update_keeps_lam_at_the_credit` を `learn_interval` で parametrize（`None` と `16 * H`）し、`test_learning_keeps_lam_at_the_credit` に改名した。構成・オラクル（100 ステップごとに max|λ| < 2·credit、最後に動いた θ の credit と 1e-8）は同じ。新しいテストを別に書くより重複が無い
  - 検出力の確認: 一時的に `2 * H` にすると max|λ| = 3.45 > 1.25 で落ちた（戻してある）

## 検討 1: module の決定性

- `Layer` の docstring に「module は決定的で状態を持たない（Dropout・学習モードの BatchNorm は不可）。学習した更新で同じ h_in に対してもう一度呼ぶため」を足した。以前は Dropout をサンプルの独立性の例に入れていたが、理由が違うので分けた
- spec の「符号・バッチ・損失の規約」にも同じ箇条を足した

## 検討 2: 過負荷

- spec の未決定事項に busy 32・スケール 1 の発散は既に記録済み。example が有限のままの発散（精度が偶然の水準に落ちる）を検出しないことと、過負荷への対応が範囲外で未着手であることを追記した

## 検証

- `uv run pytest tests/mujou/_core/pc/test_layer.py -k TestDeepChain -m "slow or not slow"`: 3 passed
- `just run`: グリーン（pytest 758 passed・21 skipped、pyright 0 errors、docs-build 成功）
