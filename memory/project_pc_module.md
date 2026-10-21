---
name: project-pc-module
description: PC-ALM は PyTorch で実装し、src/mujou/pc に置いて mujou.pc から公開する（公開 API だけに依存、トップレベルには出さない）。任意の nn.Module を包み、状態は h と λ で持ち離散的な跳びだけ付け替えて jumps で連鎖、目標は Activity で運ぶ、学習の時刻は書いた時刻から learn_interval ごと（LearnClock）、周期 τ_h/4、テストは tests/mujou/pc
metadata:
  type: project
---

設計の正本は `specs/pc.md`、実験の数値は `specs/research/pc-experiments.md`。

- **PC-ALM は PyTorch で実装する（2026-10-15、ユーザーの判断）。** 計算はノードの内側の行列演算なので、Rust では書かない。Rust のコアは学習則を知らない基盤にとどめる
- **置き場所は `src/mujou/pc/`、公開元は `mujou.pc`。** pc は `mujou` の公開名だけを import し、`mujou._core` は pc を知らない（`tests/test_layers.py`）。トップレベルに再公開しない（基盤と更新手法の区別を名前に残す）。α=0 で PC に退化するのでモジュール名は `pc`
- **公開名:** 正本は `mujou.pc.__all__` と `tests/mujou/test_init.py`。`Layer` / `Output` / `TimeConstants` / `Input` の名前はユーザーが承認（2026-10-20）
- **ノードは任意の微分可能な `nn.Module` を包む。** ノードの内側は VJP、ノードの間は PC-ALM。VJP はバッチの和、1/B は θ の勾配にだけ（ユーザーの承認）
- **状態は h と λ で持ち、入力が変わってもリセットしない。** μ が離散的に跳ぶとき（新しいサンプル、前のノードの h の跳び、自分の θ の更新）だけ h を付け替え、跳びの回数 `jumps` を `Activity` で次へ連鎖させる。ずれ u = h − μ を状態にする形は深い鎖で発散したので採らない
- **目標は入力に付随させる（2026-10-20、ユーザーの決定）。** `Activity` は `{h, jumps, y, mask, time}` で、各ノードが y・mask・time を次へ写す（コアの carry は廃止した: [[project-carry]]）
- **ノードは `act` で繋ぐ。** `Input` と `Layer` が送る spec を `act` で公開し、次のノードが `act=` で受け取る。B の照合は `activity_spec` の 1 か所（K28）
- **学習の時刻は、ホストが入力を書いた時刻から `learn_interval` ごと**（plan-implementer の判断をユーザーが承認）。`LearnClock(*, period, learn_interval, max_dt)` と `Tick` に切り出した。`learn_interval=None` は毎回、`math.inf` は学習しない。目安は 2 × ホップ数 × period より長く（検査はしない）
- **dt は `max_dt`（既定 min(τ_h/2, 3τ_λ)）で打ち切る。** 周期の既定は τ_h/4（ユーザーの提案）。λ は更新後の h の r で更新（公式実装と同じ順）
- **過負荷で発散するのは許容する**（ユーザー「過負荷では仕方ない」）

**Why:** Rust で書く利点（並行処理の安全性）はノードの間の基盤で効くもので、ノードの内側の行列演算には要らない。pc を公開 API だけで書けば、pc でできることは利用者の更新手法でもできる。

**How to apply:** pc のために基盤へ特別な口を足す必要が出たら、利用者にも要る口として公開 API に足す。テストは `tests/mujou/pc/`。

関連: [[project-pluggable-update-rule]], [[project-torch-module]], [[project-node-lifecycle]]
