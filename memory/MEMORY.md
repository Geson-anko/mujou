# MEMORY

このプロジェクトのメモリ索引。1 メモリ 1 ファイル、ここには 1 行のリンクだけを置く。詳細は各ファイルへ。

エージェント固有のメモリ・中間メモは [agents/](agents/) 配下にある。

## user

- [ユーザー像](user_role.md) — 日本語でやり取り。mujou の作者。参考リポジトリは silicone-casting / pcb-assembly

## project

- [モジュールは公開する](project_public_modules.md) — `_core` と `pc/_*.py` をやめ、`mujou/` 直下に 1 責務 1 ファイル。private は `mujou._native` だけ。`run.py` は関数 `mujou.run` と衝突するので `execution.py`（2026-10-21）
- [作り直し（2026-10-20）](project_restructure_2026_10_20.md) — 作り直しの決定の正本は specs/architecture.md と specs/scenarios.md。R1〜R8 は 2026-10-21 にユーザーが承認。テンソルの容器は register_state、pc.Chain.layers を公開
- [プロジェクトの目的](project_purpose.md) — 推論と学習を同時に行い、知能に「無常」を実現するフレームワーク。既定の更新手法は暫定的に PC-ALM
- [更新手法の差し替え](project_pluggable_update_rule.md) — PC-ALM は暫定の既定。ノードの基盤と更新手法を分離し、他の手法も与えられるようにする
- [計算単位の粒度](project_compute_granularity.md) — 基盤は非同期のノード（内側は同期、ノード間は非同期メッセージ）。大きさはノードのクラスで決まり設定では分割しない、既定はレイヤー、ニューロン単位は future work。BP も 1 ノードとして取り込める
- [コアの責務](project_core_message_framework.md) — ノードを非同期・分散的に回すこと（プロトコル + 実行フレーム）と、非同期・分散的な学習のフレームワーク（PC-ALM）自体の提供。ノード内部は PyTorch などを使ってよい
- [非同期ノードの実行](project_async_node_execution.md) — queue・pipe を使わず lock 付きの値の読み書きだけで共有。その実行のタイマー上の微分方程式として各ノードが独自に更新する
- [Rust コアと M:N 実行](project_native_core_rust.md) — コアは Rust（仕組み）、Python はインターフェース（方針）。Rust は Python の Driver を hook ごとに 1 回呼ぶ。例外は Python で定義。コアの値の型は内部に閉じ、torch とは DLPack で必ずコピー
- [実行ごとのタイマー](project_timer_per_run.md) — 時間軸は実行ごとに 1 本、既定は global_timer、復元した実行は保存した時刻から始まる専用の Timer。ランタイムの時刻源は RunClock。時刻は Timestamp | None
- [タイマーの一時停止と now への一本化](project_timer_pause.md) — Timer.pause / resume / is_paused を公開（複数の実行の状態を同時刻で取る）。時刻は now() だけ、Timestamp の差は秒の float
- [_native のスタブは生成物](project_core_stub.md) — `src/mujou/_native.pyi` は `just stub` で生成してコミット。説明は Rust の doc comment に書く
- [メッセージの型](project_message_types.md) — 利用者の値は torch.Tensor と Python のスカラー（IntSpec(bits) / FloatSpec / BoolSpec / TimestampSpec、2026-10-22）、1 つのフィールド / TypedDict、DictSpec.field、TensorSpec(shape, dtype=torch.float32)、zeros/empty/rand(generator=)、初期値必須、スロットの書き手は 1 つ、時刻（ns、None はまだ書かれていない）自動付与、バッファプールのみ
- [pc の重みの optimizer](project_pc_optimizers.md) — τθ を TimeConstants から pc.SGD / pc.AdamW に移した。設定 + bind、時間で進める AdamW（質量 c で bias 補正）、torch.optim は使わない（GIL）、h と λ は段階 2（2026-10-22）
- [PC-ALM の実装と公開元](project_pc_module.md) — PyTorch で実装し src/mujou/pc に置く（公開 API だけに依存）。h と λ で持ち跳びだけ付け替えて jumps で連鎖、目標は Activity で運ぶ、act で繋ぐ、LearnClock、周期 τ_h/4、Chain と show
- [pc のバッチの shape](project_batch_shape.md) — `batch: mujou.types.Size = ()`（`type Size = int | tuple[int, ...]`）（既定はバッチなし）、`MSELoss` / `CrossEntropyLoss(batch)`、mse は特徴の平均で ½ なし（2026-10-21）
- [ノードのライフサイクル](project_node_lifecycle.md) — hook は on_start / on_update(dt) / on_pause / on_resume / on_exception / on_stop、状態は state_dict / load_state_dict。順は on_start → load_state_dict → on_update、on_start では状態に依存する出力をしない。private な処理は Driver
- [停止と状態の取得・復元](project_pause_and_state.md) — フラグ → すぐタイマー凍結 → 全ノード停止 → device の上で複製 → 再開 → CPU へ移して返す。木は素の dict（timer と nodes）。正常終了後も取得できる。保存はユーザー。グローバルなバリア禁止の例外
- [例外によるシステム終了](project_shutdown_on_exception.md) — どの hook の例外でも全体を終了。end は失敗を検出した時刻、全ノードの on_exception → on_stop、以後のホストの呼び出しが NetworkError。例外時の状態は取得しない
- [ネットワークと Port](project_network_ports.md) — 接続点は Port（受け取る相手は高々 1、送るだけなら何本でも）と MultiPort、向きなしの send / recv の spec の組、接続は相手 → Channel の dict、Network の Port は参照、集約は更新手法の責任
- [コアの carry は廃止](project_carry.md) — 目標 y・mask・time は Activity のフィールドで運ぶ（R1）
- [ホストの入出力](project_host_io.md) — run はバックグラウンド、Run の write / read / read_with_time / subscribe / wait_for / state_dict。時刻 None はまだ書かれていない、has_received は無い。初期値は initial / inputs= / write。複数 Port の write は版番号で一斉切り替え
- [GPU の扱い](project_gpu_policy.md) — コアは GPU をサポートしない。GPU のノードは境界で device → host → device、DLPack は CPU 上の受け渡しだけ。ノードのデバイスは CPU と CUDA だけ
- [PyTorch の採用と mujou.Node](project_torch_module.md) — PyTorch を必須の依存として全面採用、torch に依存させないのは _native と mujou.time だけ。mujou.Node は純 Python で Driver が CUDA と同期を受け持つ（R3）。乱数生成器は super().__init__ で作り状態に入る
- [free-threaded Python](project_python_free_threaded.md) — 開発は 3.14t。GIL ありの 3.14 もサポートし CI で両方を検証
- [ドキュメントのサイト](project_docs_site.md) — specs/ は開発者向けの設計仕様、docs/ はユーザー向けの mkdocs（英語と日本語、tutorial / topics / api）。サンプルは examples/（snippet で取り込み、テストでは実行しない）。GitHub Pages に mike で版ごと
- [テスト用の公開の口](project_testing_api.md) — mujou.testing.run と ManualRun.step(dt, *nodes)。mujou 自身と利用者のテストが同じ口を使う。VirtualTimer は mujou.time
- [実行の API とモジュール構成](project_runtime_api_revision.md) — モジュールは公開、private は Rust の mujou._native だけ。build と Graph は持たず、mujou.run がハンドル Run を返す。state_dict は実行中と正常終了の後
- [ノードの実行周期](project_node_period.md) — `Node(period=0.01)` が既定、`0.0` は明示したときだけ、タイマーの秒、待つのはランタイム。目標は前回の目標 + P、P 以上遅れたら数え直す。Δt は実経過。手動モードは周期を無視。シグネチャと復元後の目標は確認待ち
- [run はルート 1 つ、write は時刻を返す](project_run_node_and_write_time.md) — `mujou.run(root)`（並べるのは Group、R4）。ノード単体もルート（パス ""）。`Run.write` が `Timestamp` を返す
- [Group と mujou.typing の廃止](project_group_and_typing.md) — 並べるだけの入れ物は `Group(**named)`。`mujou.typing` は廃止し、`MessageSpec` はトップレベル。汎用の注釈の型は公開モジュール `mujou.types`
- [`__dict__` が唯一のレジストリ](project_port_registry.md) — 子とポートはインスタンスの属性に代入するだけ。run の段 1 で tree.walk が代入の順に読む（R7）
- [M:N スケジューリング](project_mn_scheduler.md) — Python のノードも既定で実行ごとのワーカープールに載せる。専用スレッドは Node(dedicated_thread=True)、mujou.run(workers=None)。既定は available_parallelism（GIL ありは半分、確認待ち）で、実効の数は 0 もありうる。ノードはワーカー間を移ってよい
- [勾配の記録は切り替えない](project_grad_on_by_default.md) — 2026-10-19 ユーザーの決定。hook は通常の PyTorch と同じく勾配を記録、ノードをまたぐグラフは Port の detach コピーで作れない
- [pc の緩和の安定とヤコビアン](project_pc_jacobian_stability.md) — 周期 τ_h/4 の緩和は次のノードの σ(J) ≲ 2.6 でだけ安定。ResNet のステージは学習で 5〜12 に育つので CIFAR-10 の例は 2 ノード。GPU のノードは境界のコピーが律速（2026-10-24）
- [グラフの可視化](project_viz.md) — `mujou.viz.to_html(root)` で自己完結の HTML。CDN なし、JS は目視で確かめる（2026-10-21）
- [未決定事項](project_open_decisions.md) — 一覧の正本は specs/open-decisions.md

## feedback

- [文書の言語](feedback_doc_language.md) — 仕様書（specs/）・計画書・CLAUDE.md・skill・memory は日本語、docs/ の本文は英語と日本語、docstring とコメントは英語。コミットと PR タイトルは英語の規約形式
- [日本語の文体](feedback_japanese_style.md) — 翻訳調にしない。読点を減らす、主語と補足を削る、短い項目は体言止め。基準はユーザーが直した docs/ja/index.md
- [解説ページは教材サイト級に](feedback_docs_teaching_quality.md) — 仕組みの解説は節ごとに要点・図・表・コード。基準は docs/{en,ja}/concepts.md（2026-10-21）
- [質問は最後にまとめる](feedback_questions_at_end.md) — 確認事項は推奨案で進め、質問は報告の最後に一覧で出す（2026-10-19）
- [example は手引き](feedback_examples_are_guides.md) — examples/ は利用者への手引きで動作確認用ではない。テストで実行せず、just type が静的に検査する（2026-10-20）
- [向きは属性名の prev / next](feedback_direction_names.md) — クラスは Port / MultiPort（R2）。Up / Down、Bottom / Top は重力方向の直感と食い違うので使わない（2026-10-20）
- [super() を呼ぶ](feedback_super_calls.md) — Node/Network のサブクラスは `__init__` と override した hook の最初で super() を呼ぶ。サンプル・docs・docstring・テストの例も常に
- [公開前は互換を残さない](feedback_no_compat_before_release.md) — PyPI への正式公開までは、破壊的変更で古い名前の受け付け・移行の案内・deprecation の警告を入れず単に消す
- [ガードは提供シナリオの中だけ](feedback_guards_within_scenarios.md) — 入力検証・状態の検査は提供するシナリオで起こりうる誤りに止め、1 か所で知らせる。それ以外は YAGNI（2026-10-20）
- [良いセンスのコード](feedback_good_taste.md) — Linux カーネルメンテナーのように書く。特別扱いはデータ構造の見直しで消す、短い関数、使用者のいない抽象化を作らない（2026-10-20）
- [ダックタイピングを使わない](feedback_no_duck_typing.md) — Protocol と hasattr をやめ ABC の継承にする。状態は mujou.Stateful（nn.Module は register）、pc.Optimizer / BoundOptimizer、pc.Shown、current.Running（2026-10-23）

## reference

- [PC-ALM の出典](reference_pc_alm.md) — Sakana AI の記事 URL と、skill pc-alm がどこまで記事に依拠しているか
