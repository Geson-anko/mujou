# MEMORY

このプロジェクトのメモリ索引。1 メモリ 1 ファイル、ここには 1 行のリンクだけを置く。詳細は各ファイルへ。

エージェント固有のメモリ・中間メモは [agents/](agents/) 配下にある。

## user

- [ユーザー像](user_role.md) — 日本語でやり取り。mujou の作者。参考リポジトリは silicone-casting / pcb-assembly

## project

- [プロジェクトの目的](project_purpose.md) — 推論と学習を同時に行い、知能に「無常」を実現するフレームワーク。既定の更新手法は暫定的に PC-ALM
- [更新手法の差し替え](project_pluggable_update_rule.md) — PC-ALM は暫定の既定。ノードの基盤と更新手法を分離し、他の手法も与えられるようにする
- [計算単位の粒度](project_compute_granularity.md) — 基盤は非同期のノード（内側は同期、ノード間は非同期メッセージ）。大きさはノードのクラスで決まり設定では分割しない、既定はレイヤー、ニューロン単位は future work。BP も 1 ノードとして取り込める
- [コアの責務](project_core_message_framework.md) — ノードを非同期・分散的に回すこと（プロトコル + 実行フレーム）と、非同期・分散的な学習のフレームワーク（PC-ALM）自体の提供。ノード内部は PyTorch などを使ってよい
- [非同期ノードの実行](project_async_node_execution.md) — queue・pipe を使わず lock 付きの値の読み書きだけで共有。その実行のタイマー上の微分方程式として各ノードが独自に更新する
- [Rust コアと M:N 実行](project_native_core_rust.md) — コアは Rust、Python はインターフェース。ノードはタスクと専用スレッドの混在で実行。コアの値の型は内部に閉じ、torch とは DLPack で必ずコピー
- [実行ごとのタイマー](project_timer_per_run.md) — 時間軸は実行ごとに 1 本、既定はグローバルタイマー。1 つの実行のノード・スロット・ホストの入力は同じタイマー。時刻は Timestamp
- [_native のスタブは生成物](project_core_stub.md) — `src/mujou/_core/_native.pyi` は `just stub` で生成してコミット。説明は Rust の doc comment に書く
- [メッセージの型](project_message_types.md) — 利用者の値は torch.Tensor だけ（mujou.Tensor・mujou.dtypes・mujou.default.dtype は廃止）、torch.Tensor / TypedDict、TensorSpec(shape, torch.dtype)、zeros/rand、初期値必須、片方向スロット 2 本、時刻（秒 f64 + ns u64）自動付与、バッファプールのみ、define-and-run のネットワークオブジェクト
- [PC-ALM の実装と公開元](project_pc_module.md) — PC-ALM は Rust でなく PyTorch で実装し、mujou.pc から公開（トップレベルには出さない）
- [ノードのライフサイクル](project_node_lifecycle.md) — ノードが分散処理の最小単位。on_start / on_update(dt) / on_pause / on_resume / on_stop / state_dict / load_state_dict を override、ループはフレームワーク、state は pickle 可能な木、インスタンスが定義と実行を兼ねる
- [停止と状態の取得・復元](project_pause_and_state.md) — フラグ → すぐタイマー凍結 → 全ノード停止 → 木をコピー → 再開 → ホストに返す。正常終了後も取得できる。保存はユーザー。スロット・最後の更新時刻・タイマーも木に入る。グローバルなバリア禁止の例外
- [例外によるシステム終了](project_shutdown_on_exception.md) — どの hook が例外を投げてもシステム全体を終了（pamiq-core 参考）。その場で終了フラグ、全ノードの on_stop、ホストへ最初の例外を送出。例外時の state tree は取得しない。部分停止・再起動・スキップはしない
- [ネットワークと Port](project_network_ports.md) — Node と Network は Composite、a.connect(b) は NextPort（既定で複数）と PrevPort をつなぐ、Port 名は prev / next、向きなしの Port（send/recv 型の組）、接続 1 回で両端に登録する Channel、単一/複数を宣言で区別、相手は接続ハンドルで識別、入出力は未接続 Port の読み書き、片方向 Port 可、集約は更新手法の責任
- [ホストの入出力（暫定）](project_host_io.md) — run はバックグラウンド、ルートのネットワークの write / read / subscribe / state_dict / shutdown。run 前の write が初期値。複数 Port の write は版番号で一斉切り替え。入力はグローバル時間上の信号
- [GPU の扱い](project_gpu_policy.md) — mujou は GPU をサポートしない。コアの値はホストだけ、GPU のノードは境界で device → host → device、DLPack は CPU 上の受け渡し（必ずコピー）にだけ使う。ノード内の GPU 利用は自由
- [PyTorch の採用と mujou.Node](project_torch_module.md) — 実装は private な mujou._core パッケージ、torch 非依存は拡張モジュール mujou._core._native だけ、それ以外は PyTorch を全面採用（mujou.torch に分けない）。利用者が継承する mujou.Node が on_start で勾配停止・ノードごとのストリーム・乱数生成器を設定。状態はその場で更新。CPU と CUDA のみ、MPS は考えない。JAX は多数スレッドで伸びず不採用
- [free-threaded Python](project_python_free_threaded.md) — 開発は 3.14t。GIL ありの 3.14 もサポートし CI で両方を検証
- [ドキュメントのサイト](project_docs_site.md) — specs/ は開発者向けの設計仕様、docs/ はユーザー向けの mkdocs（英語と日本語）。サンプルは examples/ でテストが実行。今は公開しない
- [テスト用の公開の口](project_testing_api.md) — 仮想時計と手動ステップを mujou.testing などで公開し、mujou 自身と利用者のテストが同じ口を使う。実装は後
- [実行の API とモジュール構成の見直し](project_runtime_api_revision.md) — 2026-10-16 の決定。実装は mujou._core パッケージ、Rust は _core._native。build が Graph、mujou.run がハンドル、Compose、自動の状態収集、hook の名前など
- [未決定事項](project_open_decisions.md) — ネットワークスキーマの API・時定数・双対の持ち越しなど、まだ決めていない設計判断

## feedback

- [文書の言語](feedback_doc_language.md) — 仕様書（specs/）・計画書・CLAUDE.md・skill・memory は日本語、docs/ の本文は英語と日本語、docstring とコメントは英語。コミットと PR タイトルは英語の規約形式

## reference

- [PC-ALM の出典](reference_pc_alm.md) — Sakana AI の記事 URL と、skill pc-alm がどこまで記事に依拠しているか
