---
name: project-node-period
description: ノードの on_update の実行周期 period の決定（2026-10-17。既定 0.01・数え方・予約名はユーザーの決定）
metadata:
  type: project
---

ノードは `Node.__init__(*, device="cpu", seed=None, period=0.01)` の `period`（その実行のタイマーの秒、読み取り専用 `Node.period: float`）で `on_update` の周期を持つ。`period=0.0` はできるかぎり速く（Δt > 0 になりしだい）回す。2026-10-17 に実装した。同日、既定を `None`（できるかぎり速く）から 0.01 に変え、`None` をやめた（ユーザーの決定:「周期を持たないノードを想定しない。持たないノードは period=0.0 を明示する」。周期を持たないノードは CPU のコアを使い切り、ほかのノードの dt を乱すため）。native へは 0.0 もそのまま秒で渡し、bindings の `period_to_ns` が 1 ns に丸める（2026-10-19 まで 0.0 を `None` にして渡していたが、`None` を廃止したときに Rust を変えずに済ませた名残なので消した。native の内側の `Option` も 2026-10-19 に M:N のブランチ（PR 3）で消した。bindings の `periods` は `list[float] | None`、コアの `RuntimeConfig::period_ns` は `Option<Vec<u64>>` で、外側の `None` は「全ノード 1 ns」の既定）。`period` は `device` / `generator` / `timer` と同じ予約名で、クラス属性で置き換えると実行の開始で `TypeError`（ユーザーの決定。クラス属性の既定値は認めない）。2026-10-19、`None` を渡したときに `period=0.0` を案内する専用の `TypeError` を消した（[[feedback-no-compat-before-release]]）。注釈は `float` なので型チェッカーが弾く。

- 待つのはランタイム（Rust の `node_loop`）。目標の時刻（最初は最初の Δt の起点 + period、その後は前回の目標 + period）まで `Timer::wait_until_ns_or` で待つ。スケール・仮想時計に従い、停止・終了の要求ですぐ起きる
- 数え方（2026-10-17、ユーザーの決定）: 次の目標は前回の目標 + P。P 以上遅れたとき（目標 + P にすでに達していたとき）だけ、その更新の時刻 + P に数え直す。まとめて回らず、更新と起床の遅れが P に収まる間は平均の頻度が 1/P。pamiq-core の `SleepIntervalAdjustor`（起きた時刻から数え直す）からの意図した変更で、理由は macOS で起床が遅れ、周期 10 ms が約 80 Hz になった実測
- 復元では目標が木に入らないので、目標は「保存した最後の更新 + P」に置き直される。止めずに続けた場合と次の更新の時刻が変わりうる（Δt は実経過のまま）。2026-10-17 に方向 (a)（このまま仕様に書く）を orchestrator の推奨案で採った。ユーザー確認待ち。代案 (b) は次の目標を木に入れる
- Δt は実際の経過時間（おおむね P、遅れれば長く、その次は短い。遅れが P に近いほど 0 に近づく）。`max_dt` は持たないので、`on_update` は長い Δt でも安定でなければならない（必要ならノードの内側で刻みを分割）
- 手動モード（テストのステップ）は周期を無視する
- pamiq-core と変えた点: 目標を前回の目標から数える、最初の更新は開始から period 後（Δt = 0 の更新に意味が無い）、`offset` は持たない

**Why:** hook の中からの `sleep` を禁じた（2026-10-17）ので、周期はフレームワークが持つ必要がある。待ちをランタイムに置けば、停止・終了の要求とスケール・仮想時計にそのまま従う。

**How to apply:** 周期の振る舞いを変えるとき、`specs/nodes.md#実行の周期` と `crates/mujou-core/tests/runtime.rs` の `mod period` を合わせる。未決定の残り（M:N での扱い）は [[project-open-decisions]]。
