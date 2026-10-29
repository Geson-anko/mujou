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

## 速度実験の続き（2026-10-28）

- d8s128（τh/8、16 τh）: epoch 1〜3 の prequential 32.80 / 34.35 / 13.88%、test 43.70 / 22.19 / 17.67%。2 epoch 目から崩れる
- d8s192e10（τh/8、24 τh、10 epoch）: 3 epoch 目までは 55.37% / 56.61% で BP と並ぶが、4 epoch 目 50.06% / 34.23%、5 epoch 目 22.15% / 29.60%、10 epoch 目 14.39% / 19.58% に崩れた。376 s/epoch
- → τh/8 は 3 epoch では h 形と同じ精度に見えたが、長くは保たない。学習でゲインが育つと r 形が落ち着かなくなる（open-decisions の予想どおり）と見られる
- 次: 既定（τh/16、384 周期）で 10 epoch（d16s384e10）。これが崩れるなら r 形の既定そのものに問題がある
- d16s384e10 はユーザーの指示で 3 epoch 目（56.21% / 59.00%）のあと止めた

## 取り組み直し（2026-10-28 15 時〜、ユーザーの新しい制約）

制約: time scale 1 以上（2 まで余裕）、学習時間は BP の 10 倍まで、小さいバッチでも比べる。非連続なデータではバッチごとに r（h）のリセットが必要で、リセットはフラグで行う（memory/feedback_pc_speed_budget.md）。

### 予算（scratch budget/bench*.py、RTX 4090）

- BP の 1 バッチ: B = 16〜128 でほぼ一定の 5.0〜5.4 ms（起動律速）
- 全ノード 1 周期のコスト（11 スレッド並列、各自の stream で、最も遅いノード）: eager 2.1 ms（B ≤ 64）、3.4 ms（B = 128）。ノード 1 つだけなら 0.4 ms で、並列にすると CPU 側の競合で 5 倍に延びる。CUDA graph にすると 0.43 / 0.72 / 1.56 / 3.6 ms（B = 16 / 32 / 64 / 128）
- → 余裕 2 倍込みで 1 バッチ 50 ms に収めるには、eager では 1 バッチ約 12 周期しか使えない

### リセットフラグ

- `Sample` と `Activity` に `reset`（バッチの形の bool）を足した。ノードは入力を初めて見た更新で、`reset` の立ったサンプルの r と λ を 0 にし、relaxer に `reset_(samples)` を伝える
- テスト: 参照実装（Jacobi）にリセットを入れ、keep / reset の両方で一致。リセットを外すと 4 本が落ちることを確かめた

### credit の届き方（scratch credit/credit.py、重み固定・Jacobi・初期の重み）

- r 形（τh/16、τλ = τh/4、τr = τh/3）: credit は約 8 周期（τh/2）で 1 ノード進む。layer0 の cos が 0.9 になるのは 96 周期。収束先の cos は 0.91〜0.94 止まり（ノルム比 1.06〜1.1）
- リセットありは持ち越しより早く向きがそろう（96 周期の layer0: 0.90 対 0.40）が、波が来るまでノルムはほぼ 0
- **FPA（上へ h = μ + r でなく μ を送る、fixed prediction assumption）**: 不動点は同じで、cos 1.00 に収束する。上向きのループが切れて三角（上から下だけ）になるので、局所の 2 変数系が安定なら J によらず安定
  - τh/4: 32〜48 周期で全層 cos ≥ 0.93、64 周期でノルム比 0.99
  - dt = τh = τλ、減衰なし（デッドビート: λ ← −g、r → 0）: 32 周期で全層 cos 1.00・ノルム比 1.00（1 周期 1 ホップ）。波面より下は過渡のごみ（古い g から）が大きいが、波が来ると消える
  - 減衰なしの r 形を同じ刻みにすると収束しない
- 増幅する線形の鎖（1.25 × 16）: FPA なしは τh/4 でも τh/16 でも発散。FPA ありはデッドビートで 64 歩、τh/4 で 200 歩で credit に一致（相対誤差 1e-10 以下）
- **臨界減衰の FPA**: dt = τh/2、τλ = dt（= τh/2）、τr = τh（a = 0.5、b = 1、k = 2）にすると δ' = r' + λ' = −g となり、δ が 1 歩で −g に一致する（行き過ぎなし）。スカラーの鎖（J = 1、10 層）の走査で最速かつ peak 1.00。ResNet（初期の重み）では 20 周期で全層 cos 1.00、26 周期でノルム比 1.00。デッドビート（a = b = 1）は δ = −λ − 2g が 2 倍に行き過ぎ、層ごとに複利で増える（波面で ±1、ノルム 768 まで）
- 古い g（次のノードがまだ前のサンプルにいる間の g）を時刻のタグで捨てる案: 臨界減衰なら不要（タグなしでも波面の前のごみは credit の 0.24〜0.48 倍で増えない）

### CIFAR の学習（B = 128 は lr 1.6e-2、11 ノード、FPA は scratch のモンキーパッチ）

| 構成 | epoch 1 | epoch 2 | epoch 3 | s/epoch |
|---|---|---|---|---|
| r 形・リセット・τh/16・96 周期・scale 0.25 | 18.36 / 14.01 | 12.15 / 19.38 | 11.80 / 13.22 | 189 |
| FPA デッドビート・32 周期・scale 1 | 21.36 / 32.00 | 32.16 / 37.78 | 37.16 / 42.42 | 64 |
| FPA 臨界減衰・32 周期・scale 1（周期 5 ms） | 31.22 / 43.13 | 46.56 / 54.41 | 55.04 / 56.04 | 64 |
| FPA 臨界減衰・24 周期・scale 1 | 30.41 / 40.75 | 45.79 / 52.89 | 53.94 / 57.67 | 48 |
| （参考）r 形・τh/16・384 周期・scale 0.25 | 31.78 / 45.73 | 48.09 / 51.08 | 56.21 / 59.00 | 750 |
| （参考）BP | | | 54.45 / 59.02 | 2.8 |

- B = 32（lr 8e-3）、FPA 臨界減衰、24 周期、周期 5 ms、scale 2（実時間 2.5 ms）: 29.23 / 39.85、42.21 / 42.49、48.78 / 53.59、95 s/epoch。BP は 33.17 / 42.16、49.30 / 55.78、56.74 / 61.47、9 s/epoch（10.6 倍）。1 更新 1.5 ms（B = 128 と同じ: CPU と起動で律速）で、平均 dt 1.03 周期、ときどき 10〜34 周期の停止 → 追いつくのがやっと
- 同じ B = 32 で周期 4 ms（実時間 2 ms）: 追いつかない（平均 dt 1.13〜1.30 周期、最大 106 周期）→ 34.34 / 37.62%
- **5 ノード（3 ブロックずつの Layer 3 つ + 出力）、B = 32、12 周期、周期 5 ms、scale 2**: 33.92 / 45.28、49.19 / 52.79、56.86 / 61.76%（BP 32.48 / 42.99、48.50 / 51.75、56.29 / 60.60）。48.2 s/epoch（BP 9.0 s の 5.4 倍）。1 更新 1.7 ms、平均 dt 1.00 周期。scale 1 なら 10.7 倍
- 次: 同じ構成で 10 epoch（g3s12x2e10）。FPA を pc に入れるかはユーザーに確認する

### 実装（2026-10-29、ユーザーの判断: FPA だけにする、例は 5 ノード・B = 32）

- `Layer._send` が `mu` を送る。`TimeConstants.period` = h/(1+h/r)、`max_dt` = min(period, 2 lam)
- テスト: 参照実装が μ を送る。α = 0 のオラクルは「r が BP の credit、送る値が forward」に、減衰なしで発散するテストは「増幅する鎖が credit に届く」に変えた。pc のテスト 237 本が通る
- 例: CIFAR はステージごとの 5 ノード、B = 32、12 周期、τh = 10 ms、scale 2、ワーカー 5。例そのものの 10 epoch: 73.05 / 73.90%（BP 73.20 / 74.35%）、48.2 s/epoch（BP 8〜9 s）。3 epoch の別の実行は 55.35 / 60.79%（BP 57.62 / 61.88%）。MNIST は周期 τh/2・16 周期で 97.56%（BP 97.54%）、scale 2 でも 99% が 1.25 周期以内
- 自己レビューの気づき: コミットは src とテストで reset と FPA を分けられなかった（layer.py の同じ箇所に両方が入る）
