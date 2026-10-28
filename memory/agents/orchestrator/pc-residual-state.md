# pc の状態を残差 r にする（2026-10-27）

PR #112（cifar10-fine-relaxation）の上に積む。ユーザー指示: 「mujou の pc を h ではなく残差 r 形式に変更し、確認された安定性問題に対処する」。

## 計画（段階 1）

形: ノードは $r$ と $\lambda$ を状態に持ち、毎回の更新で $h = \mu(h_{\mathrm{in}}) + r$ を送る。

$$\tau_h \dot r = -(r + \lambda + g) - \frac{\tau_h}{\tau_r} r,\qquad \tau_\lambda \dot\lambda = r$$

- 予測 $\mu$ が動いても（入力のドリフト・新しいサンプル・前のノードや自分の θ の跳び）$r$ は変わらないので、付け替え（reanchor）、`jumps`、θ の跳びの 1 更新遅延、学習後の 2 度目の forward がすべて不要になる
- 安定性: 減衰なしの r 形は深い鎖で崩れる（下の測定）。`TimeConstants.r` を `lam` と同じく必須にし、推奨を `r = h` とする（`math.inf` は減衰なしとして残す）

公開 IF の変更（公開前なので互換は残さない）:

- `pc.TimeConstants(*, h, lam, r)`: `r` が必須に
- `pc.functional.relax_(r, lam, g, dt, tau, relaxer=None)`: `h` と `mu` を取らず `r` を動かす
- `pc.relax.Relaxer.bind(r)`、`BoundRelaxer.step_(r, force, dt, tau_h)`: 引数名を `r` に。`reset_` は新しいサンプルのときだけ呼ぶ
- `pc.Layer`: 属性 `r`, `lam`（`h` は送るだけで持たない）。state は `generator`, `module`, `r`, `lam`, `clock`, `optimizer`, `relaxer`（`h`, `mu`, `jump`, `jumped`, `jumps`, `seen_jumps` を削除）
- `pc.msg.Activity` から `jumps` を削除。`pc.Input` は状態なし（`jumps`, `seen_write` を削除）
- `pc.clock.LearnClock.discrete` を削除（使い道が jumps だけ）

テスト観点:

- 参照実装 `tests/mujou/pc/_reference.py` を r 形の Jacobi 更新に書き直し、TestJacobi で一致させる
- TestDeepChain: 減衰ありで不動点に着く／減衰なしは崩れる（設計の理由をオラクルで残す）。h の確認は r = 0
- relax_ の 1 歩、減衰、relaxer への引数
- Input が jumps を送らない、Activity の spec に jumps が無い
- 入力が変わったとき h が新しい forward + r に乗る（付け替え不要の性質）、θ を更新した直後の送る h

## 測定（段階 1、scratch rstate/deep.py）

TestDeepChain と同じ ResBlock 8 段（tanh、幅 8、τ_h=4e-3、τ_λ=8e-3、周期 τ_h/4、Jacobi、4000 ステップ）。λ と BP credit の最大誤差:

| τ_r | 結果 |
| --- | --- |
| ∞ | 653 ステップで発散 |
| 4τ_h | 誤差 3.7、cos −0.64（未収束） |
| 2τ_h | 4e-16 |
| τ_h | 5e-15 |
| τ_h/2 | 4e-13 |
| τ_h/3 | 5e-11 |

## 実装（段階 2〜3）

- テストを先に書き、50 件の red を確認してから実装した。`_reference.py` は r 形の Jacobi 更新に書き直した
- 計画外の判断: `pc.relax.HeavyBall` は r 形で効果が見えず（4 ノード ResBlock、60 ステップで Euler 0.214 対 0.216〜0.243、τ=3e-3 は減衰と合わせて発散）、`test_the_front_of_the_credit_crosses_the_chain_in_fewer_periods` を削除し、HeavyBall のテストの τ を 2e-3 にした。削除はユーザー判断として open-decisions へ
- `TestDecayOfR` は TAU が減衰を持つようになったので `TestFixedPoint` と重複し削除。代わりに `TestDeepChain.test_without_the_decay_of_r_it_diverges`

## 実測（段階 3〜5）

- MNIST（3 epoch、例そのまま、r = h）: テスト 97.58%（h 形 97.60%、BP 97.54%）
- pc_merge（Merge だけ毎回学ぶ）: r 形・h 形とも 4 回に 1 回ほど損失が 0.7 前後に跳ね上がる。r 形に固有ではない
- CIFAR-10 の重みを固定した r 形の同期緩和（PC 10 epoch の重み、τ_λ = τ_h/4、8 τ_h 後の cos）: 周期 τ_h/4 で 0.96〜0.98（τ_r = τ_h/3）、τ_h/8 と τ_h/16 で 0.99〜1.00

## レビュー（段階 4）

- code-reviewer: approve。should-fix（r < h で既定の period が作れない、docstring の h 形の残り、例の数値）を対応。nit も対応
- algorithm-reviewer: request-changes（軽め）。M1「r = h は safe」は増幅する鎖では成り立たない（線形 w = 1.25 × 16 段で ρ > 1、自分でも再現: τ_r = τ_h で 1.045）。docstring・specs・open-decisions に限界として記録。S1 PC 極限の記述を修正。S2 period と max_dt は文書化し open-decisions へ。S3 HeavyBall は削除を推奨 → ユーザー判断
- CIFAR-10（例の既定、3 epoch）: prequential 57.22%・テスト 61.36%（h 形 54.83%・59.05%、BP 54.45%・59.02%）

## ユーザーの判断と追加の依頼（2026-10-28、PR #113 の後）

- 1（増幅する鎖の不安定）: ドキュメントにする。doc-teacher-student と yomiyasu で docs を改善
- 2（HeavyBall）: 削除する → 削除。`Relaxer` の差し替え口は残し、状態を持つ relaxer のテストは `_reference.Inertial` で書いた
- 3（既定の周期と max_dt）: OK（docstring と specs の記述のまま。open-decisions から外した）
- 追加: 学習の高速化（CIFAR の周期 τ_h/8、提示時間の短縮）を試す。安定性の強化もこの PR で

## 安定性の強化の調査（scratch rstate/ct*.py、kappa*.py）

- 原因: r 形の流れは $\dot r = -A^\top\nabla_r E$ で降下方向でない。連続時間で不安定（w = 1.25 × 16、τ_λ = τ_h/2: k = 1, 3, 10, 30 で実部の最大 0.72, 0.24, 0.054, 0.012）。周期を縮めても直らない
- 試した: ドリフトを κ だけ通す形、δ に罰則 (1+k) r を入れる形、h 形 + 減衰。連続時間では「罰則 + κ ≤ 0.5」が安定だが、離散（周期 τ_h/4）では増幅する鎖と ResBlock の両方を安定にする設定は無かった
- 結論: この PR では r 形の則は変えない。specs/pc.md「減衰が要る理由と限界」と open-decisions に記録。長時間の学習（CIFAR 10 epoch）で崩れないかを実測で確かめる
