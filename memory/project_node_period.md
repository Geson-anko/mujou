---
name: project-node-period
description: ノードの on_update の実行周期 period の決定（2026-10-17、推奨案で実装、ユーザー確認待ち）
metadata:
  type: project
---

ノードは `Node.__init__(*, device="cpu", seed=None, period=None)` の `period`（その実行のタイマーの秒、読み取り専用 `Node.period`）で `on_update` の周期を持つ。`None` はできるかぎり速く（Δt > 0 になりしだい）回す。2026-10-17 に推奨案で実装し、ユーザー確認待ち。

- 待つのはランタイム（Rust の `node_loop`）。前回の更新の直前に読んだ時刻 + period まで `Timer::wait_until_ns_or` で待つ。スケール・仮想時計に従い、停止・終了の要求ですぐ起きる
- 数え方は pamiq-core の `SleepIntervalAdjustor` に倣う（ユーザーの指示）: 計算時間は周期に含める。遅れたら 1 回だけ更新し、その時刻から数え直す（取り戻さない）
- Δt は実際の経過時間（≥ period）。`max_dt` は持たない
- 手動モード（テストのステップ）は周期を無視する
- pamiq-core と変えた点: 最初の更新は開始から period 後（Δt = 0 の更新に意味が無い）、`offset` は持たない

**Why:** hook の中からの `sleep` を禁じた（2026-10-17）ので、周期はフレームワークが持つ必要がある。待ちをランタイムに置けば、停止・終了の要求とスケール・仮想時計にそのまま従う。

**How to apply:** 周期の振る舞いを変えるとき、`specs/nodes.md#実行の周期` と `crates/mujou-core/tests/runtime.rs` の `mod period` を合わせる。未決定の残り（`offset`、クラス属性の既定値、M:N での扱い）は [[project-open-decisions]]。
