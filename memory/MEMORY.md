# MEMORY

このプロジェクトのメモリ索引。1 メモリ 1 ファイル、ここには 1 行のリンクだけを置く。詳細は各ファイルへ。

エージェント固有のメモリ・中間メモは [agents/](agents/) 配下にある。

## user

- [ユーザー像](user_role.md) — 日本語でやり取り。mujou の作者。参考リポジトリは silicone-casting / pcb-assembly

## project

- [プロジェクトの目的](project_purpose.md) — 推論と学習を同時に行い、知能に「無常」を実現するフレームワーク。既定の更新手法は暫定的に PC-ALM
- [更新手法の差し替え](project_pluggable_update_rule.md) — PC-ALM は暫定の既定。ブロックの基盤と更新手法を分離し、他の手法も与えられるようにする
- [計算単位の粒度](project_compute_granularity.md) — 基盤は非同期のブロック（内側は同期、ブロック間は非同期メッセージ）。大きさはブロックのクラスで決まり設定では分割しない、既定はレイヤー、ニューロン単位は future work。BP も 1 ブロックとして取り込める
- [コアの責務](project_core_message_framework.md) — ブロックを非同期・分散的に回すこと（プロトコル + 実行フレーム）と、非同期・分散的な学習のフレームワーク（PC-ALM）自体の提供。ブロック内部は PyTorch などを使ってよい
- [非同期ブロックの実行](project_async_block_execution.md) — queue・pipe を使わず lock 付きの値の読み書きだけで共有。グローバルタイマー上の微分方程式として各ブロックが独自に更新する
- [Rust コアと M:N 実行](project_native_core_rust.md) — コアは Rust、Python はインターフェース。ブロックはタスクと専用スレッドの混在で実行。値の型は独自、外部ライブラリとはコピーで変換（DLPack は使わない）
- [メッセージの型](project_message_types.md) — Tensor（f16/bf16/f32/f64・整数・bool、不変、入れ物+変換）、Tensor / TypedDict（Tensor | None なし）、TensorSpec と mujou.default.dtype、zeros/rand、初期値必須、片方向スロット 2 本、時刻（秒 f64 + ns u64）自動付与、バッファプールのみ、define-and-run のネットワークオブジェクト
- [ブロックのライフサイクル](project_block_lifecycle.md) — ブロックが分散処理の最小単位。on_start / on_update(dt) / on_pause / on_resume / on_stop / state_dict / load_state_dict を override、ループはフレームワーク、state は pickle 可能な木、インスタンスが定義と実行を兼ねる
- [停止と状態の取得・復元](project_pause_and_state.md) — フラグ → すぐタイマー凍結 → 全ブロック停止 → 木をコピー → 再開 → ホストに返す。保存はユーザー。スロット・最後の更新時刻・タイマーも木に入る。グローバルなバリア禁止の例外
- [例外によるシステム終了](project_shutdown_on_exception.md) — どの hook が例外を投げてもシステム全体を終了（pamiq-core 参考）。その場で終了フラグ、全ブロックの on_stop、ホストへ最初の例外を送出。設定で state tree を取得（保存はユーザー）。部分停止・再起動・スキップはしない
- [ネットワークと Port](project_network_ports.md) — Block と Network は Composite、a.connect(b) は NextPort（既定で複数）と PrevPort をつなぐ、Port 名は prev / next、向きなしの Port（send/recv 型の組）、接続 1 回で両端に登録する Channel、単一/複数を宣言で区別、相手は接続ハンドルで識別、入出力は未接続 Port の読み書き、片方向 Port 可、集約は更新手法の責任
- [ホストの入出力（暫定）](project_host_io.md) — run はバックグラウンド、ルートのネットワークの write / read / subscribe / state_dict / shutdown。run 前の write が初期値。複数 Port の write は版番号で一斉切り替え。入力はグローバル時間上の信号
- [GPU の扱い](project_gpu_policy.md) — mujou は GPU をサポートしない。Tensor はホストだけ、GPU のブロックは境界で device → host → device、DLPack は使わない。ブロック内の GPU 利用は自由
- [PyTorch の採用と mujou.Block](project_torch_module.md) — torch 非依存は private な mujou._core だけ、それ以外は PyTorch を全面採用（mujou.torch に分けない）。利用者が継承する mujou.Block が on_start で勾配停止・ブロックごとのストリーム・乱数生成器を設定。状態はその場で更新。CPU と CUDA のみ、MPS は考えない。JAX は多数スレッドで伸びず不採用
- [free-threaded Python](project_python_free_threaded.md) — 開発は 3.14t。GIL ありの 3.14 もサポートし CI で両方を検証
- [未決定事項](project_open_decisions.md) — ネットワークスキーマの API・時定数・双対の持ち越しなど、まだ決めていない設計判断

## feedback

- [文書の言語](feedback_doc_language.md) — 仕様書・計画書・CLAUDE.md・skill・memory は日本語。コミットと PR タイトルは英語の規約形式

## reference

- [PC-ALM の出典](reference_pc_alm.md) — Sakana AI の記事 URL と、skill pc-alm がどこまで記事に依拠しているか
