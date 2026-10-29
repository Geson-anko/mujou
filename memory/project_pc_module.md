---
name: project-pc-module
description: PC-ALM は PyTorch で実装し、src/mujou/pc に置いて mujou.pc から公開する（公開 API だけに依存、トップレベルには出さない）。任意の nn.Module を包み、状態はずれ r = h − μ と λ で持ち予測 μ を送る（固定予測の仮定、τ_r は必須）、reset フラグで行ごとに消す、目標は Activity で運ぶ、学習の時刻は書いた時刻から learn_interval ごと（LearnClock）、周期 τ_h/(1+τ_h/τ_r)、テストは tests/mujou/pc
metadata:
  type: project
---

設計の正本は `specs/pc.md`、実験の数値は `specs/research/pc-experiments.md`。

- **PC-ALM は PyTorch で実装する（2026-10-15、ユーザーの判断）。** 計算はノードの内側の行列演算なので、Rust では書かない。Rust のコアは学習則を知らない基盤にとどめる
- **置き場所は `src/mujou/pc/`、公開元は `mujou.pc`。** pc は `mujou`・`mujou.time`・`mujou.testing` の公開名だけを import し、基盤のモジュール（`mujou.node` など）は pc を知らない（`tests/test_layers.py`）。トップレベルに再公開しない（基盤と更新手法の区別を名前に残す）。α=0 で PC に退化するのでモジュール名は `pc`
- **公開名:** 正本は `mujou.pc.__all__` と `tests/mujou/test_init.py`。`Layer` / `Output` / `TimeConstants` / `Input` の名前はユーザーが承認（2026-10-20）。重みの optimizer `Optimizer` / `BoundOptimizer` / `SGD` / `AdamW` を足し、`functional.learn_` は消した（2026-10-22、[[project-pc-optimizers]]）
- **ノードは任意の微分可能な `nn.Module` を包む。** ノードの内側は VJP、ノードの間は PC-ALM。VJP はバッチの和、1/B は θ の勾配にだけ（ユーザーの承認）
- **状態はずれ r = h − μ と λ で持つ（2026-10-28、ユーザーの指示「pc を h ではなく残差 r 形式に変更し、確認された安定性問題に対処する」）。** 跳びの付け替えと `jumps` は消した。2026-10-27 までは h で持ち跳びだけ付け替えていた
- **送るのは予測 μ（固定予測の仮定、2026-10-29、ユーザーの判断「FPA だけにする」）。** h = μ + r を送ると r がヤコビアンの積を通って戻るループができ、増幅する鎖で発散した。μ を送ると credit は後ろへだけ流れ、局所の刻みが安定ならどの鎖でも安定。不動点は同じ。`TimeConstants.r` は既定の周期を決めるので必須のまま（推奨 τ_h）
- **リセットはフラグで（2026-10-28、ユーザーの指示）。** `Sample` / `Activity` の `reset`（`mask` と同じ形の bool）。ノードは入力を初めて見た更新で、立った行の r と λ を 0 にし `relaxer.reset_(reset)` を呼ぶ。シャッフルしたデータは True、ストリームは False（[[feedback-pc-speed-budget]]）
- **目標は入力に付随させる（2026-10-20、ユーザーの決定）。** `Activity` は `{h, y, mask, time}` で、各ノードが y・mask・time を次へ写す（コアの carry は廃止した: [[project-carry]]）。`time` は `Timestamp`（`TimestampSpec()`、書き込みの前は `Timestamp(ns=0)`）（2026-10-22）
- **ノードは `act` で繋ぐ。** `Input` と `Layer` が送る spec を `act` で公開し、次のノードが `act=` で受け取る。バッチの照合は `activity_spec` の 1 か所（K28）。バッチの shape は [[project-batch-shape]]
- **学習の時刻は、ホストが入力を書いた時刻から `learn_interval` ごと**（plan-implementer の判断をユーザーが承認）。`LearnClock(*, period, learn_interval, max_dt)` と `Tick` に切り出した。`learn_interval=None` は毎回、`math.inf` は学習しない（学ばないノードの表し方はこの 1 通り。`TimeConstants.theta=inf` の規則は τθ を optimizer に移したときに消した、2026-10-22）。目安は 2 × ホップ数 × period より長く（検査はしない）
- **dt は `max_dt`（既定 min(周期, 2τ_λ)）で打ち切る。** 周期の既定は τ_h/(1+τ_h/τ_r)（τ_r = τ_h で τ_h/2、2026-10-29。それまでは τ_h/4）。τ_λ = τ_h/2 と組むと δ が 1 歩で −g になり、credit は 1 周期に 1 ノード戻る。λ は更新後の r で更新（公式実装と同じ順）
- **過負荷で発散するのは許容する**（ユーザー「過負荷では仕方ない」）

**Why:** Rust で書く利点（並行処理の安全性）はノードの間の基盤で効くもので、ノードの内側の行列演算には要らない。pc を公開 API だけで書けば、pc でできることは利用者の更新手法でもできる。

**How to apply:** pc のために基盤へ特別な口を足す必要が出たら、利用者にも要る口として公開 API に足す。テストは `tests/mujou/pc/`。

関連: [[project-pc-optimizers]], [[project-pluggable-update-rule]], [[project-torch-module]], [[project-node-lifecycle]]
