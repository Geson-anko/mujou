# cifar10-fine-relaxation（solo-dev-cycle の記録）

2026-10-27。ブランチ `feature/2026-10-27/cifar10-fine-relaxation`。

ユーザーの /goal:「アナログ回路で作られることを考慮しつつ（積分のリークなど）残差の枝のスケールを小さく初期化する、周期を短くするなどを行い、安定的に学習できつつ、BP の性能に近くなるようなものを目指しましょう。BP より遅くなっても良い」。途中でユーザーが「バッチを変更した際に lambda のリセットは必要ではないですかね」と質問。

結論と数値の正本は `specs/pc.md`「ResNet のブロックごとのノード（2026-10-27）」。ここには手順と測り方を残す。

## 測り方（scratch のスクリプト。セッションの scratchpad にあり、リポジトリには入れていない）

- `bp_sigma.py`: モデルの変種（枝のスケール α、GroupNorm の eps、加算後の ReLU の有無）を BP で 10 epoch 学び、次のノードのモジュールのサンプルごとの最大特異値（べき乗法、`torch.autograd` の二重 backward。`torch.func` は triton を import して GIL を有効にするので使わない）を測る
- `oracle.py`: 重みを固定し、`mujou.pc.functional` で全ノードを同時に更新する同期の緩和（Jacobi）。λ と δ を BP の credit `-dl/dh` と cos・大きさの比で比べる。dt/τ_h、τ_λ、h と λ のリーク・オフセット、μ に向かう減衰を変えられる
- `switch.py`: バッチ A を緩和した後に B へ切り替え、λ を持ち越す場合とリセットする場合の cos の時間変化
- `sim_train.py`: 同期の緩和で学習まで回す（遅延なし。1 バッチ約 48 秒と遅い）
- `pc_train.py`: 非同期の実行（mujou.run）でノードを手で組み、周期・τ_λ・リーク（`layer.relax_` を差し替え）・リセット（`Layer._relax` を差し替え）を変える。`--diag` で数十バッチごとにモジュールの複製の σ と |λ| を出す（生きたモジュールを autograd で読むと in-place 更新と競合して落ちた）
- 非同期の実行は 1 本ずつ流す。3 本同時では dt の中央値が 2.8 周期になり測定が無効になった。単独なら実時間の周期 5 ms で dt の 99% 点 1.01 周期

## 経過

1. 同期の緩和で dt/τ_h = 1/4・1/16・1/64 を比べ、離散化が原因と確定
2. BP の変種: α = 0.1・0.25、eps 0.01・0.1 は σ を下げず BP を悪くした。加算後の ReLU を外す変種（α = 0.25、eps 0.1 と組み合わせ）は BP が 77.5% / 76.6% と良かったが σ は大きい（採らず、モデルは変えない）
3. τ_h/16・16 τ_h・Euler: 3 epoch 目は 41.3%（崩壊）と 47.4%。τ_h/4 + HeavyBall は σ が 1〜1.7 のまま 42.8%
4. リセット: 非同期では 200 バッチ 19%（持ち越し 26%）。同期の学習では差なし。未解明
5. τ_λ と μ に向かう減衰を同期の緩和で探索 → τ_λ = τ_h/4、τ_r = τ_h（3 epoch 54.24%、減衰なし 51.75%）
6. 10 epoch: τ_r = τ_h・16 τ_h・学習 0.75 は 67.28% / 66.07%。10 epoch の重みで探索し直して τ_r = τ_h/3・24 τ_h・学習 0.9 → 71.56% / 71.41%（BP 73.27% / 72.61%）
7. アナログの誤差（リーク・オフセット）を同期の緩和で測った（specs の表）

## 実装

- `TimeConstants.r`（既定 inf）と `relax_` の減衰の項、`max_dt` の既定の式。テスト: `TestTimeConstants`、`TestRelax.test_the_decay_of_r_pulls_h_towards_mu`、`TestDecayOfR`（不動点で λ = BP の credit）
- `pc.Chain(period=)`。テスト: `test_every_node_runs_at_the_period_given`
- 例: 周期 τ_h/16、`--tau-lam`・`--tau-r`・`--div`、`--momentum` は外した（τ_h/16 で測っていない）、テストの見せる時間は 32 周期

## 確認待ち（ユーザーへ）

- `src/mujou/pc` の学習則に `TimeConstants.r` を足した（memory の方針で src の pc の変更は確認する）。名前と置き場所（relaxer ではなく時定数）
