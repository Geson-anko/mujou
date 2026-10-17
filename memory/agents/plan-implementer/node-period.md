# node-period: ノードの `on_update` の実行周期

ブランチ `feature/2026-10-17/node-period`（`feature/2026-10-16/node-state` の上）。solo-dev-cycle で進めた記録。

## 1. 計画

### pamiq-core の `interval_adjustors.py` を読んで分かったこと

- `SleepIntervalAdjustor.adjust()` は「前回の `reset` の時刻 + (interval - offset)」まで `sleep` し、起きた時刻で `reset` し直す。`FixedIntervalInteraction` は `setup` で `reset` し、`step` の**後**に `adjust` を呼ぶ
- 周期は「前の step の開始（= 前の待ちが明けた時刻）から次の step の開始まで」。step にかかった時間は周期の中に入るので、計算時間は自動で差し引かれる
- 遅れたとき（step が interval より長いとき）は待たずにすぐ次へ進み、そこで `reset` する。遅れを取り戻すために続けて回すことはせず、次の周期はその時刻から数え直す
- オーバーヘッドの補償は `offset`（利用者が与える定数）を待ち時間から引くだけ。起床の遅れ（sleep の寝過ごし）や `adjust` 自身のコストは自動では補償しない
- `adjust` は経過時間（スケール済み）を返し、時刻は pamiq-core の `time` モジュール（スケール・一時停止つき）の `perf_counter`

### mujou での形

- `Node.__init__(*, device="cpu", seed=None, period=None)`。`period` はその実行のタイマーの秒。`None` は従来どおり（Δt > 0 になりしだい回す）。有限の正の数でなければ `ValueError`。読み取り専用 property `Node.period`
- コア: `RuntimeConfig.period_ns: Option<Vec<Option<u64>>>`。`node_loop` は自由実行のとき「時刻が `last_ns + period_ns`（周期なしは `last_ns + 1`）に達するまで」タイマー上で待つ。`last_ns` は前回の更新の直前に読んだ時刻 = pamiq-core の `reset` の時刻と同じ位置。待ちは `wait_until_ns_or` で、停止・一時停止・終了の要求（制御語の変化）ですぐ起き、スケール変更・仮想時計の前進にも追従する
- 遅れたとき: pamiq-core と同じく、取り戻さずに「その更新の時刻」から数え直す。手元の推奨案（数え直す）とも一致
- Δt は従来どおり実際の経過時間（`now_ns - last_ns`）。遅れれば P より長い
- 最初の更新: 開始時刻（または復元した `last_update_ns`）から P 後
- 手動モード（`step`）では周期を使わない（Δt > 0 なら更新）
- binding: `Runtime.start(..., period_ns: list[int | None] | None = None)`。session が各ノードの `period` を ns に丸めて渡す（1 ns 未満に丸まる値は 1 ns）

### テスト観点（Rust の結合テストを主に、Python は少数）

Rust（`crates/mujou-core/tests/runtime.rs` の `mod period`）:
1. 仮想時計: P 未満の前進では更新しない／P で 1 回だけ、Δt = P（ns 厳密）
2. 遅れ（時刻が飛んだ、更新中に P 以上進んだ）: 1 回だけ更新し Δt = 実経過、その時刻から数え直す（P/2 進めても更新しない。位相固定方式なら更新する、で区別）
3. 周期なしは 1 ns 進めるだけで更新する
4. 手動モードは周期を無視する
5. 停止要求・capture（一時停止）で周期の待ちがすぐ起きる（実時計、P = 1000 s、実時間の上限）
6. 実時計: 回数 ≤ 経過/P + 1、回数 ≥ 粗い下限、各 Δt ≥ P
7. スケール: 実時間あたり回数 ≤ 実経過 × scale / P + 1 かつ、スケール 1 なら起こりえない回数を超える
8. 引数検証: 0 の周期、長さ違いは InvalidArgument

Python: `Node(period=...)` の検証と property、仮想時計での 1 本の結合（P 未満で更新なし、P で Δt == P）。

### リスク

- 起床の遅れが周期ごとに積もる（pamiq-core と同じ）。実時計テストは上限側を厳密、下限側を緩く

## 2. テスト実装

- Rust `crates/mujou-core/tests/runtime.rs` の `mod period` に 9 本。コンパイルエラー（`RuntimeConfig::period_ns` が無い）で red を確認
- Python `tests/mujou/_core/test_node.py` の `TestPeriod` に 4 本（パラメータ込み 7）。`run-handle` ブランチの merge で書き直す前提で、検証と結合 1 本に絞った
- 遅れの扱いは「数え直し」と「位相固定（P の倍数の格子）」を区別できるように、1.5 P 進めた後に 0.5 P 進めても更新しないことを確かめる。取り戻す方式なら 2 回目がすぐ来るので、`assert_quiet` で落ちる

## 3. 機能実装と計画外の判断

- コアは `Shared::min_dt_ns`（周期なしは 1 ns）に統一した。従来の「Δt > 0 まで待つ」は「1 ns の周期」と同じ式になるので、分岐を増やさずに済む
- 周期 0 は `InvalidArgument`。Python 側は `Node.__init__` で有限の正の数でなければ `ValueError`（利用者の入力の境界）。ns への丸めは session の `_period_ns` で行い、1 ns 未満は 1 ns に切り上げる
- 周期はノードの `period` property から session が読む（コアは Python を呼ばない）。binding の `Runtime.start(period_ns=...)` を足した
- クラス属性での既定値: `_period = None` をクラス属性に置いた（`super().__init__()` を呼ばないノード用、既存の `_device` / `_seed` と同じ）。サブクラスが `period = 0.01` のようなクラス属性で既定を置く API は入れなかった。`period` は property なので、上書きすると検証を通らない値も入りうる。設定の口を `__init__` の 1 つにした。必要ならユーザーに確認（未決定事項に記載）
- `max_dt` は入れない（指示どおり）。specs/open-decisions.md の「時定数」と memory に記録

## pamiq-core から取り入れたもの・変えたもの

| 項目 | pamiq-core | mujou | 理由 |
| --- | --- | --- | --- |
| 周期の起点 | 前の待ちが明けた時刻（= step の開始） | 前の更新の直前に読んだ時刻 | 同じ位置。計算時間は周期に含まれる |
| 遅れ | 待たずに次へ、起きた時刻から数え直す | 同じ | pamiq-core 優先の指示。推奨案とも一致 |
| オーバーヘッド補償 | 利用者の定数 `offset` を待ち時間から引く | 持たない | 計算時間はもともと周期に入る。残る起床の遅れは実時間の量で、タイマーの秒の定数はスケール・仮想時計と噛み合わない。未決定事項に残した |
| 最初の step | 開始直後 | 開始から P 後 | `on_update(dt)` は Δt ぶん進める処理で、開始時刻では Δt = 0 |
| 待ち方 | スケール済みの `time.sleep`（停止中はすぐ戻る） | タイマーの条件変数で絶対時刻まで待つ | スケール変更・仮想時計・停止/終了の要求にそのまま従う |
| 戻り値 | `adjust()` が経過時間を返す | Δt を `on_update` に渡す | 既存どおり |

## 4. 自己レビュー

- diff を通読。要求からトレースできない行なし。局所性: 待ちはノードごとで、グローバルなバリアを持ち込んでいない
- 気付き（未対応）: 非常に大きい `period`（約 584 年超）は ns が u64 を超え、`run` で `OverflowError` になる。起こりえない範囲なので対処しない
- 整理点なし

## 5. 検証

- `just run` green
- `cargo test --test runtime` を 10 回、`mod period` を 4 並列 × 3 回、pytest を free-threaded と `-X gil=1` で 3 回ずつ。すべて green

## 6. レビュー後の作業（2026-10-17）

code-reviewer・algorithm-reviewer とも approve（`memory/agents/{code,algorithm}-reviewer/node-period.md`）。

### merge（`feature/2026-10-16/node-state`、run-handle を含む）

- 衝突は 3 ファイル
  - `session.py`: node-state 側の `Session.start` を採り、`period_ns` は `runtime.start(...)` の引数として渡した（`restored` の dict には入れない。復元しない実行でも周期が渡るように）
  - `node.py`: `on_update(dt)` の行で、node-state の「the run's ``timer``」の文言に周期の記述を合わせた
  - `specs/execution-model.md`: 手動モードの行を `mujou.testing.run` / `ManualRun.step` の表記にし、「周期は使わない」を足した
- `TestPeriod` の結合テストを新しい口に書き直した（自由実行の `mujou.run(..., timer=VirtualTimer())` と `advance`）。手動の `mujou.testing.run` は周期を使わない
- 文書の手動モードの表記を `mujou.testing` に合わせた

### ユーザーの決定 1: 数え方を変えた

- 次の目標 = 前回の目標 + P。更新の時刻が「前回の目標 + P」以上（P 以上遅れた）なら、目標をその時刻 + P に数え直す。最初の目標は最初の Δt の起点（開始 / 復元した `last_update_ns`）+ P
- pamiq-core（起きた時刻から数え直す）からの意図した変更。理由: macOS では起床の遅れ（周期 10 ms で中央値約 2.5 ms、algorithm-reviewer の実測）が周期ごとに積もり、10 ms が約 80 Hz になった。前回の目標から数えれば平均の頻度が 1/P に保たれ、`offset` も要らない
- 結果として Δt は P より短くなりうる（少し遅れた次の更新）。docstring・specs の「Δt ≥ period」を「おおむね P」に直した
- 境界: 「ちょうど P 遅れ」は数え直す。`>` にすると Δt = 0 の更新が起きる（ミューテーションでテストが落ちることを確認）
- 復元では、状態に入るのは最後の更新時刻だけで目標は入らないので、起点は最後の更新 + P になる（位相は数え直しと同じ扱い）。木の形を変えないための選択

### ユーザーの決定 2: クラス属性での既定の周期は入れない

- `period` を予約名の一覧（`device`・`generator`・`period`・`timer`）に足した
- session は `node.period_ns(node)`（node.py のモジュール関数）で検証済みの `_period` を読む。サブクラスが `period` を置き換えていたら（`inspect.getattr_static` で `Node.period` の property と違う）、実行の開始の最初（復元・書き込みの前）で `TypeError` を出す。開始は何も始めずに失敗するので、置き換えを直せば同じネットワークで実行し直せる

### ほかの指摘への対応

| 指摘 | 対応 |
| --- | --- |
| code should-fix 1・3 | merge で対応（上記） |
| code should-fix 2・4 | 予約名にした（決定 2） |
| code should-fix 5 / algorithm should-fix 1 | Rust に `a_capture_does_not_move_the_target`・`a_restored_node_counts_its_period_from_its_last_update` を追加（仮想時計） |
| algorithm should-fix 2 | `specs/nodes.md#実行の周期` と docstring に「長い Δt でも安定に。必要ならノードの内側で刻みを分割。打ち切ると時間が失われる」 |
| algorithm nit 1 | 停止の時間は数えないこと、復元の起点を docstring と specs に書いた |
| algorithm nit 3 | `each_node_keeps_its_own_period`（P と 3P） |
| algorithm nit 4 / code nit 2 | `bool` は `TypeError` |
| code nit 3 | docstring の起点（復元を含む）と「every period」→「once every period on average」 |
| code nit 4 | specs を「前回の更新から合計で目標まで進めたとき（1 回の advance である必要はない）」に。Rust テストでも P/2 ずつ進めて確かめる |
| code nit 5 | `period_ns` は 0.5 を 0 から遠い側へ丸める（コアの `seconds_to_ns` と同じ） |
| code nit 6 | 実時計テストの下限を `RUN`（sleep の長さ）で計算 |
| code nit 8 | Python に `test_a_period_below_a_nanosecond_is_one_nanosecond` |
| code nit 1（584 年超で OverflowError）・nit 7（u64::MAX 付近の saturating）| 記録のみ。起こりえない範囲 |
| algorithm nit 2（起床の遅れで頻度が下がる）| 数え方の変更で解消。実時計テストの下限を「平均で 1/P の半分以上」に上げた |

### 検証

- 負荷試験はしていない（coordinator の指示。CPU を埋める並列は回さない）。実時計テストの不等式は、上限が「目標は P ずつ離れているので回数 ≤ 経過/P」で負荷に依らず、下限は期待値の半分で、遅れが積もらない方式なので余裕が大きい

## 7. 2 巡目のレビュー後（2026-10-17）

code-reviewer の 2 巡目は approve（`memory/agents/code-reviewer/node-period-r2.md`）。このブランチはすでに `docs/2026-10-17/docs-content` と `feature/2026-10-17/api-polish` に merge されているので、振る舞いは変えずに文書・記録・コメントだけ直した。

- should-fix 1（復元で目標の位相が失われる）: 方向 (a)（orchestrator の推奨案、ユーザー確認待ち）。状態の木は目標を持たないので、復元すると目標は「保存した最後の更新 + P」に置き直される。少し遅れて更新した後に取った状態からは、止めずに続けた場合より次の更新が最大で P 近く遅れうる（Δt は実経過のまま）。取得してそのまま続ける場合は目標を保つ。`specs/nodes.md`・`Node` の docstring・`memory/project_node_period.md` に書いた。代案 (b) は次の目標を木に入れて `RuntimeConfig` に渡す（木の形が変わる）
- should-fix 2: `memory/project_node_period.md` の待ちの記述を目標の方式に直した
- nit 1: 「平均で P に 1 回」に「`on_update` の時間と起床の遅れの和が P に収まる間は」を足した（docstring、`RuntimeConfig::period_ns` の doc、specs）。超えると頻度はおよそ 1/(C + L) に下がる
- nit 2: 「すでに過ぎていたら」を「すでに達していたら」に（実装の `>=` と一致）
- nit 3: 遅れが P に近いほど次の Δt は 0 に近づく（常に > 0）ことを docstring と specs に
- nit 4: `exactly_a_period_late_counts_again_from_the_late_update` のコメントを、`>` では Δt = 0 の更新が起きることを区別していると読めるように直した
- nit 6: docstring の `on_update(dt)` の折り返しを直した
- 記録だけ:
  - nit 7: 同じ P・同じ起点のノードは位相がそろい続け、同じ ns に起きてタイマーの lock を取り合う。正しさには影響しない。M:N の検討材料
  - nit 8: 約 584 年を超える `period` は `runtime.start` で `OverflowError`。何も起動せずに戻るので実害なし
  - nit 9: 予約名の検査は実行の開始で行う。`Node.__init_subclass__` なら定義時に落とせるが、今の位置でも何も始まらずに `TypeError` になり走らせ直せる
