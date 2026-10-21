---
name: project-node-lifecycle
description: ノードは分散処理の最小単位で、hook（on_start / on_update(dt) / on_pause / on_resume / on_exception / on_stop）と状態の 2 メソッド（state_dict / load_state_dict）を override して書く。呼ぶ順は on_start → load_state_dict → on_update。on_start では状態に依存する出力をしない。ループと private な処理は Driver が持つ。インスタンスが定義と実行を兼ね、1 回だけ走る
metadata:
  type: project
---

2026-10-14 決定。詳細は `specs/nodes.md` の「ライフサイクル hook」。

- **ノードが実行時に存在する分散処理の最小単位。** ネットワークは組み立てだけの構造で hook を持たない
- **hook を override する方式。手本は Unity の `Update(deltaTime)`。** Lightning ではない（エポック・バッチ前提の hook が不変の原則に反する）
- **hook:** `on_start()`、`on_update(dt)`（読む → dt ぶん更新 → 書く）、`on_pause()` / `on_resume()`（引数なし）、`on_exception()`（どれかのノードの例外で終わるとき、全ノードに 1 回、`on_stop()` の前。アクチュエータの安全停止など）、`on_stop()`（資源の解放。状態を変えない）。状態は `state_dict()` / `load_state_dict(state)` で、hook とは分けて書く
- **呼ぶ順は `on_start` →（復元なら）`load_state_dict` → `on_update` の繰り返し**（ユーザーの決定「on_start は状態の確保」を保つ）。代わりに `on_start` では状態に依存する出力（送信、アクチュエータへの指令）をしない。例と docs は状態を `__init__` で作る書き方にそろえ、`on_start` は I/O を開くのに使う
- **private な処理は Driver が持つ。** CUDA の device・stream、contextvars、同期、状態の複製は、Rust が hook のたびに呼ぶ Python の `Driver` が hook の外で行う。利用者の MRO に private hook は無い（[[project-torch-module]]）
- **`state_dict()` は状態の木を返し、フレームワークが停止中に複製する。** ノードもフレームワークもファイルを扱わない（ユーザーの指摘）。既定は自動で集めた状態（テンソル、`state_dict` を持つ属性、`register_state` した名前、乱数生成器）
- **`on_pause` / `on_resume` は引数を取らない。** 停止中はその実行の時間が進まないので、停止と再開は同じ時刻。停止していた実時間はダイナミクスの外の量なので、要る I/O ノードが自分で測る
- **持たない hook:** エポック・学習/推論モード、受信ごとに呼ぶ hook（受信を溜めて queue に戻る）、固定刻みの更新（テストで仮想時計から作る）
- **インスタンスは定義と実行を兼ねる（`nn.Module` と同じ）。** 1 回だけ走り（K14）、終わった後は属性を読んでよい
- **更新手法は `Node` のサブクラスとして提供する**（PC-ALM のレイヤーなど）。差し替え可能の要件は「基盤が学習則を知らない」こと

**Why:** ループをフレームワークが持たないと、M:N のワーカーに載せられず、仮想時計で決定的に回すテストのオラクルも書けない。順を `__init__` → `load_state_dict` → `on_start` に変えると、`on_start` で状態を作るノードが復元した値を黙って上書きする。

**How to apply:** ノードの API を変更するときはこの hook 構成に合わせる。hook が例外を投げたらシステム全体を終了する（[[project-shutdown-on-exception]]）。

関連: [[project-pause-and-state]], [[project-compute-granularity]], [[project-native-core-rust]]
