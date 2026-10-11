---
name: reference-pc-alm
description: PC-ALM の出典 URL と、skill pc-alm の式がどこまで記事に依拠しているか
metadata:
  type: reference
---

- 出典: Sakana AI, "PC-ALM: Augmented Lagrangian Predictive Coding" <https://pub.sakana.ai/pc-alm/>
- skill `pc-alm` の 1〜6 節は記事の定義を勾配まで展開したもの。勾配式は 2026-10-11 に有限差分で検証済み（tanh、4 層、誤差 3e-9）。不動点で重み更新が BP 勾配と一致することも同時に数値で確認した（非線形でも成立。ただし不動点への収束保証は線形のみ）
- 7 節「mujou の研究課題」は記事の範囲外。理論として扱わない
- 記事が触れていない点: 双対の初期化・リセット方針、非同期スケジューリング、ペナルティ係数の調整（½ 固定）

関連: [[project-purpose]] [[project-open-decisions]]
