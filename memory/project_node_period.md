---
name: project-node-period
description: ノードの on_update の実行周期 period の決定（2026-10-17。数え方と予約名はユーザーの決定）
metadata:
  type: project
---

ノードは `Node.__init__(*, device="cpu", seed=None, period=None)` の `period`（その実行のタイマーの秒、読み取り専用 `Node.period`）で `on_update` の周期を持つ。`None` はできるかぎり速く（Δt > 0 になりしだい）回す。2026-10-17 に実装した（シグネチャは推奨案のまま、ユーザー確認待ち）。`period` は `device` / `generator` / `timer` と同じ予約名で、クラス属性で置き換えると実行の開始で `TypeError`（ユーザーの決定。クラス属性の既定値は認めない）。

- 待つのはランタイム（Rust の `node_loop`）。前回の更新の直前に読んだ時刻 + period まで `Timer::wait_until_ns_or` で待つ。スケール・仮想時計に従い、停止・終了の要求ですぐ起きる
- 数え方（2026-10-17、ユーザーの決定）: 次の目標は前回の目標 + P。P 以上遅れたとき（目標 + P をすでに過ぎていたとき）だけ、その更新の時刻 + P に数え直す。まとめて回らず、平均の頻度は 1/P。pamiq-core の `SleepIntervalAdjustor`（起きた時刻から数え直す）からの意図した変更で、理由は macOS で起床が遅れ、周期 10 ms が約 80 Hz になった実測
- Δt は実際の経過時間（おおむね P、遅れれば長く、その次は短い）。`max_dt` は持たないので、`on_update` は長い Δt でも安定でなければならない（必要ならノードの内側で刻みを分割）
- 手動モード（テストのステップ）は周期を無視する
- pamiq-core と変えた点: 目標を前回の目標から数える、最初の更新は開始から period 後（Δt = 0 の更新に意味が無い）、`offset` は持たない

**Why:** hook の中からの `sleep` を禁じた（2026-10-17）ので、周期はフレームワークが持つ必要がある。待ちをランタイムに置けば、停止・終了の要求とスケール・仮想時計にそのまま従う。

**How to apply:** 周期の振る舞いを変えるとき、`specs/nodes.md#実行の周期` と `crates/mujou-core/tests/runtime.rs` の `mod period` を合わせる。未決定の残り（M:N での扱い）は [[project-open-decisions]]。
