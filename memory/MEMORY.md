# MEMORY

このプロジェクトのメモリ索引。1 メモリ 1 ファイル、ここには 1 行のリンクだけを置く。詳細は各ファイルへ。

エージェント固有のメモリ・中間メモは [agents/](agents/) 配下にある。

## user

- [ユーザー像](user_role.md) — 日本語でやり取り。mujou の作者。参考リポジトリは silicone-casting / pcb-assembly

## project

- [プロジェクトの目的](project_purpose.md) — 推論と学習を同時に行い、知能に「無常」を実現するフレームワーク。既定の更新手法は暫定的に PC-ALM
- [更新手法の差し替え](project_pluggable_update_rule.md) — PC-ALM は暫定の既定。ブロックの基盤と更新手法を分離し、他の手法も与えられるようにする
- [計算単位の粒度](project_compute_granularity.md) — 基盤は非同期のブロック（内側は同期、ブロック間は非同期メッセージ）。大きさはニューロン/レイヤー/スタック、既定はレイヤー。BP も 1 ブロックとして取り込める
- [コアの責務](project_core_message_framework.md) — ブロックを非同期・分散的に回すこと（プロトコル + 実行フレーム）と、非同期・分散的な学習のフレームワーク（PC-ALM）自体の提供。ブロック内部は PyTorch などを使ってよい
- [非同期ブロックの実行](project_async_block_execution.md) — queue・pipe を使わず lock 付きの値の読み書きだけで共有。グローバルタイマー上の微分方程式として各ブロックが独自に更新する
- [Rust コアと M:N 実行](project_native_core_rust.md) — コアは Rust、Python はインターフェース。ブロックはタスクと専用スレッドの混在で実行。値の型は独自、PyTorch 連携は future work
- [メッセージの型](project_message_types.md) — Tensor（f32・不変・入れ物+変換）、Tensor / Tensor | None / TypedDict、初期値必須、片方向スロット 2 本、時刻（秒 f64 + ns u64）自動付与、バッファプールのみ、define-and-run のネットワークオブジェクト
- [free-threaded Python](project_python_free_threaded.md) — 開発は 3.14t。GIL ありの 3.14 もサポートし CI で両方を検証
- [未決定事項](project_open_decisions.md) — ネットワークスキーマの API・時定数・双対の持ち越しなど、まだ決めていない設計判断

## feedback

- [文書の言語](feedback_doc_language.md) — 仕様書・計画書・CLAUDE.md・skill・memory は日本語。コミットと PR タイトルは英語の規約形式

## reference

- [PC-ALM の出典](reference_pc_alm.md) — Sakana AI の記事 URL と、skill pc-alm がどこまで記事に依拠しているか
