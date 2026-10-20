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
- [タイマーの一時停止と now への一本化](project_timer_pause.md) — Timer / mujou.time の pause・resume・is_paused を公開（複数の実行の状態を同時刻で取る）。時刻は now() だけ、Timestamp の差は符号付きの秒
- [_native のスタブは生成物](project_core_stub.md) — `src/mujou/_core/_native.pyi` は `just stub` で生成してコミット。説明は Rust の doc comment に書く
- [メッセージの型](project_message_types.md) — 利用者の値は torch.Tensor だけ（mujou.Tensor・mujou.dtypes・mujou.default.dtype は廃止）、torch.Tensor / TypedDict、TensorSpec(shape, torch.dtype)、zeros/rand、初期値必須、片方向スロット 2 本、時刻（秒 f64 + ns u64）自動付与、バッファプールのみ、define-and-run のネットワークオブジェクト
- [PC-ALM の実装と公開元](project_pc_module.md) — PC-ALM は Rust でなく PyTorch で実装し、mujou.pc から公開（トップレベルには出さない）。実装計画の決定（ずれ u で持ちリセットしない、教師は入力の時刻で揃える、learn_interval、周期 τ_h/4、register_state）
- [ノードのライフサイクル](project_node_lifecycle.md) — ノードが分散処理の最小単位。on_start / on_update(dt) / on_pause / on_resume / on_exception / on_stop / state_dict / load_state_dict を override、ループはフレームワーク、state は pickle 可能な木、インスタンスが定義と実行を兼ねる
- [停止と状態の取得・復元](project_pause_and_state.md) — フラグ → すぐタイマー凍結 → 全ノード停止 → 木をコピー → 再開 → ホストに返す。正常終了後も取得できる。保存はユーザー。スロット・最後の更新時刻・タイマーも木に入る。グローバルなバリア禁止の例外
- [例外によるシステム終了](project_shutdown_on_exception.md) — どの hook が例外を投げてもシステム全体を終了（pamiq-core 参考）。その場で終了フラグ、全ノードの on_exception → on_stop、ホストへ最初の例外を送出。例外時の state tree は取得しない。部分停止・再起動・スキップはしない
- [ネットワークと Port](project_network_ports.md) — Node と Network は Composite、接続は Port 同士（a.next.connect(b.prev)）、NextPort は既定で複数・PrevPort は 1 つ、Port 名は prev / next、向きなしの Port（send/recv 型の組）、接続 1 回で両端に登録する Channel、単一/複数を宣言で区別、相手は接続ハンドルで識別、入出力は未接続 Port の読み書き、片方向 Port 可、集約は更新手法の責任
- [ホストの入出力（暫定）](project_host_io.md) — run はバックグラウンド、mujou.run が返す Run ハンドルの write / read / subscribe / wait_for / state_dict / shutdown / join。初期値と復元は inputs= / state=。複数 Port の write は版番号で一斉切り替え。入力はその実行のタイマー上の信号
- [GPU の扱い](project_gpu_policy.md) — mujou は GPU をサポートしない。コアの値はホストだけ、GPU のノードは境界で device → host → device、DLPack は CPU 上の受け渡し（必ずコピー）にだけ使う。ノード内の GPU 利用は自由。ノードのデバイスは CPU と CUDA だけ（MPS などは 2026-10-17 に取りやめ）
- [PyTorch の採用と mujou.Node](project_torch_module.md) — 実装は private な mujou._core パッケージ、PyTorch を必須の依存として全面採用（mujou.torch に分けない）、torch を使わないのは拡張モジュール mujou._core._native と時刻まわりだけ。利用者が継承する mujou.Node が内部の hook _on_enter で勾配停止・ノードごとのストリームを設定、乱数生成器は super().__init__ で作る（2026-10-18）。状態はその場で更新。CPU と CUDA だけ。JAX は多数スレッドで伸びず不採用
- [free-threaded Python](project_python_free_threaded.md) — 開発は 3.14t。GIL ありの 3.14 もサポートし CI で両方を検証
- [ドキュメントのサイト](project_docs_site.md) — specs/ は開発者向けの設計仕様、docs/ はユーザー向けの mkdocs（英語と日本語）。サンプルは examples/ でテストが実行。GitHub Pages に公開（2026-10-18〜）。mike でバージョンごと（main は dev、タグは X.Y.Z と latest）
- [テスト用の公開の口](project_testing_api.md) — 仮想時計と手動ステップを mujou.testing などで公開し、mujou 自身と利用者のテストが同じ口を使う。2026-10-16 実装（run / ManualRun / VirtualTimer）
- [実行の API とモジュール構成の見直し](project_runtime_api_revision.md) — 2026-10-16 の決定。実装は mujou._core パッケージ、Rust は _core._native。build が Graph、mujou.run がハンドル、Group、自動の状態収集、hook の名前など
- [ノードの実行周期](project_node_period.md) — `Node(period=0.01)` が既定、`0.0` は明示したときだけ、タイマーの秒、待つのはランタイム。目標は前回の目標 + P、P 以上遅れたら数え直す（平均 1/P、まとめて回らない）。Δt は実経過。`period` は予約名。手動モードは周期を無視
- [run がノードを受け付け、write が時刻を返す](project_run_node_and_write_time.md) — 2026-10-17。`Run.write` が `Timestamp` を返す。`mujou.run(component, *components)` / `Group(*components)` がノードを受け付け混在可。1 つだけなら包まず根にする（ノードのパス `""`、ユーザーの決定）
- [Compose を Group に、型は mujou.typing に](project_group_and_typing.md) — 2026-10-17 のユーザーの決定。並べるだけなので直列の合成に読める名前を避けた。`Single` / `Multi` / `MessageSpec` を `mujou.typing` から公開
- [接続は Port 同士だけ、Port と子は代入時に登録](project_port_registry.md) — 2026-10-17。Component.connect と PrevPort / NextPort を 1 つずつに限る制約を廃止。nn.Module と同じく __setattr__ で登録し build は vars() を探索しない。init 前の代入は AttributeError、構築後の変更は RuntimeError
- [M:N スケジューリング](project_mn_scheduler.md) — 2026-10-18 のユーザーの要求。Python のノードも既定で実行ごとのワーカープールに載せる。専用スレッドは Node(dedicated_thread=True) で opt-in（on_thread_start は 2026-10-19 に廃止）、mujou.run(workers=None)、ノードはワーカー間を移ってよい（保つのは contextvars と CUDA の device/stream）
- [勾配の記録は切り替えない](project_grad_on_by_default.md) — 2026-10-19 ユーザーの決定。_before_hook を廃止、hook は通常の PyTorch と同じく勾配を記録
- [未決定事項](project_open_decisions.md) — ネットワークスキーマの API・時定数・双対の持ち越しなど、まだ決めていない設計判断

## feedback

- [文書の言語](feedback_doc_language.md) — 仕様書（specs/）・計画書・CLAUDE.md・skill・memory は日本語、docs/ の本文は英語と日本語、docstring とコメントは英語。コミットと PR タイトルは英語の規約形式
- [日本語の文体](feedback_japanese_style.md) — 翻訳調にしない。読点を減らす、主語と補足を削る、短い項目は体言止め。基準はユーザーが直した docs/ja/index.md
- [質問は最後にまとめる](feedback_questions_at_end.md) — 確認事項は推奨案で進め、質問は報告の最後に一覧で出す（2026-10-19）
- [example は手引き](feedback_examples_are_guides.md) — examples/ は利用者への手引きで動作確認用ではない。テストで全部を実行しない（2026-10-20）
- [向きの名前は Prev / Next](feedback_direction_names.md) — Up / Down、Bottom / Top は重力方向の直感と食い違うので使わない（2026-10-20）
- [super() を呼ぶ](feedback_super_calls.md) — Node/Network のサブクラスは `__init__` と override した hook の最初で super() を呼ぶ。サンプル・docs・docstring の例も常に
- [公開前は互換を残さない](feedback_no_compat_before_release.md) — PyPI への正式公開までは、破壊的変更で古い名前の受け付け・移行の案内・deprecation の警告を入れず単に消す

## reference

- [PC-ALM の出典](reference_pc_alm.md) — Sakana AI の記事 URL と、skill pc-alm がどこまで記事に依拠しているか
