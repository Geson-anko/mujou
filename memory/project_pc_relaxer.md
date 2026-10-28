---
name: project-pc-relaxer
description: mujou.pc の r（もとは h）の更新手法は relaxer（pc.relax.Euler / pc.relax.HeavyBall）で差し替える。重みの optimizer と同じ形（設定 + bind）。TimeConstants.momentum は載せ替えて削除した
metadata:
  type: project
---

設計の正本は `specs/pc.md` の「h の relaxer」。実装は `src/mujou/pc/relax.py`。

- **きっかけ（2026-10-26、ユーザー）。** `TimeConstants.momentum`（h の慣性）を見て「h にも θ の optimizer のように更新の仕方を自由に変えられる仕組みが必要」と言われ、別 PR で実装した。慣性という 1 つの手法を時定数の 1 項目にするより、手法そのものを差し替える部品にする、という判断
- **設定 + bind。** `pc.relax.Euler()`（既定）と `pc.relax.HeavyBall(tau=)` は状態を持たない frozen dataclass（ABC の `Relaxer`）。`Layer` が `relaxer.bind(r)` で束縛（ABC の `BoundRelaxer`、`mujou.Stateful`: `step_(r, force, dt, tau_h)`、`reset_()`、`state_dict`、`load_state_dict`）を作り `self.relaxer` に置く。状態の木のキーは `"relaxer"`。`Layer` と `Chain` の省略できる引数で、`Output` は r を持たないので取らない
- **差し替えるのは r だけ（2026-10-28 に h から r へ）。** λ は動かした後の r で進める。`relax_(..., relaxer=None)` の None は素の 1 歩
- **`reset_()` の呼びどころ。** 新しいサンプルだけ（r 形では予測の跳びで r も力も跳ばない）。理由は「力が跳ぶ」ではなく「速度が積んだのは前の入力の力」（付け替えの瞬間に力は連続する。algorithm-reviewer の指摘）
- **自作の relaxer の条件。** 力が 0 で relaxer 自身の状態が静止したところ（HeavyBall なら F=0 かつ v=0）で h を動かさない（不動点が動くと λ が credit にならない）。r 形では r を動かすのは step_ だけなので、位置の型の状態も持てる。r 形では HeavyBall の効果が見えず、存続は未決定（open-decisions）
- `TimeConstants.momentum` は削除（PR #108 で入れたが未 merge の名前だったので互換は残さない）。この PR は #108 のブランチの上に積んだ。#108 が merge されたら base を main に付け替える

**Why:** optimizer と同じ形にすれば、利用者は h の動かし方も同じ作法で差し替えられ、`TimeConstants` に手法ごとの項目が増えない。

**How to apply:** h や λ の緩和の手法を足すときは `Relaxer` の実装として足す。λ の差し替えと h・λ の組は研究から（`specs/open-decisions.md`）。

- 関連: [[project-pc-optimizers]], [[project-pc-jacobian-stability]]
