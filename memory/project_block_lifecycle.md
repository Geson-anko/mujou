---
name: project-block-lifecycle
description: ブロックは分散処理の最小単位で、ライフサイクル hook（on_start / on_update(dt) / on_pause / on_resume / on_stop / state_dict / load_state_dict）を override して書く。ループはフレームワークが持つ。インスタンスが定義と実行を兼ねる
metadata:
  type: project
---

2026-10-14 決定。詳細は `docs/blocks.md` のライフサイクル hook の節。

- **ブロックが実行時に存在する分散処理の最小単位。** ネットワークは構築時だけの構造で hook を持たない。Block と Network の共通基底が持つのは Port と connect だけ（[[project-network-ports]]）
- **hook を override する方式。手本は Unity の `Update(deltaTime)`。** Lightning ではない（エポック・バッチ前提の hook が不変の原則に反する）
- **hook:** `on_start()`（実行スレッド上で 1 回、状態の確保）、`on_update(dt)`（1 ステップ。読む → dt ぶん更新 → 書く）、`on_pause()` / `on_resume()`（停止・再開。引数なし）、`on_stop()`（資源の解放。例外による終了でも呼ぶ）、`state_dict()` / `load_state_dict(state)`（状態の取得・復元）
- **`state_dict()` は pickle できるオブジェクトの木を返す。** pamiq-core の `save_state(path)` のようにブロックがパスへ直接書く方式は採らない（ユーザーの指摘）。フレームワークもファイルを扱わず、保存方法はユーザーが決める（[[project-pause-and-state]]）。木には torch テンソルや optimizer の状態、更新し続ける値への参照を入れてよい（フレームワークが停止中にコピーする）。コアの Tensor は pickle 可能にする
- **`on_pause` / `on_resume` は引数を取らない。** 停止中はグローバル時間が進まないので、グローバル時間では停止と再開が同じ時刻 T になり、時刻を渡しても情報がない。T はタイマーから読める。停止していた実時間はダイナミクスの外の量なので、必要な I/O ブロックが自分で測る
- **持たない hook:** エポック・学習/推論モード、受信ごとに呼ぶ hook（受信を溜める必要があり queue に戻る）、固定刻みの更新（テストで仮想時計から作る）
- **インスタンスは定義と実行を兼ねる（`nn.Module` と同じ）。** 構築で Port が実スロットに結び付き、同じインスタンスが hook を呼ばれる。凍結スキーマから実行時オブジェクトを作り直さない
- **更新手法は `Block` のサブクラス群として提供する**（PC-ALM のレイヤーなど）。Strategy の注入は最初はしない。差し替え可能の要件は「基盤が学習則を知らない」こと

**Why:** ループをフレームワークが持たないと、M:N のワーカープールに載せられず、仮想時計で決定的に回すテストのオラクルも書けない。hook は Rust（trait）と Python（専用スレッド）の境界も兼ねる。

**How to apply:** ブロックの API を実装・変更するときはこの hook 構成に合わせる。名前やシグネチャは docs ではイメージ扱いなので、確定させる前にユーザーに確認する。hook が例外を投げたらシステム全体を終了する（[[project-shutdown-on-exception]]）。

関連: [[project-pause-and-state]], [[project-compute-granularity]], [[project-native-core-rust]]
