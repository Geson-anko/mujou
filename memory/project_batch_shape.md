---
name: project-batch-shape
description: mujou.pc のバッチは shape で既定は ()（バッチなし）。損失は MSELoss / CrossEntropyLoss(batch)、mse は特徴の平均で ½ なし（2026-10-21）
metadata:
  type: project
---

2026-10-21 ユーザーの決定。`mujou.pc` のバッチは `batch: Size = ()`（`type Size = int | tuple[int, ...]`、汎用の型なので公開モジュール `mujou.types` に置く。ユーザーの提案と決定）。`()` はバッチの無い 1 サンプル、int の `B` は `(B,)`、多次元も取る。`activity_spec(h, y, batch)`・`Input(batch=)`・`Chain(batch=)` で宣言し、`Layer` / `Output` は `act_spec` の mask の shape から読む。

損失は torch の `F.mse_loss` / `nn.MSELoss` に倣い、関数 `pc.functional.mse` / `pc.functional.cross_entropy`（`batch=()`）とクラス `pc.losses.MSELoss(batch=())` / `pc.losses.CrossEntropyLoss(batch=())`。`pc.mse` / `pc.cross_entropy` は消した。

**Why:** 常にバッチを要求する API は、1 サンプルずつ学ぶストリーム（mujou の主な用途）にダミーの次元を書かせる。`()` を許せば多次元への一般化は追加のコストなしで付いてくる。mse は名前どおり特徴で平均し（½ なし）、1/B と合わせて θ の更新が `nn.MSELoss()` の勾配降下と一致する。交差エントロピーも、1 サンプルの目標が構造を持つとき畳む次元とクラスの次元（バッチの直後）をバッチから決めるので、バッチを受け取る。

**How to apply:** pc の API・例・docs ではバッチを `batch=` で明示する。既定が `()` なので、バッチ付きの spec で `batch` を書き忘れると全体が 1 サンプルとして黙って走る。設計の正本は specs/pc.md の「符号・バッチ・損失の規約」。[[project-pc-module]]
