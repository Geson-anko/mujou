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
