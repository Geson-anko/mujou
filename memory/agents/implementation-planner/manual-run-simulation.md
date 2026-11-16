# ManualRun をシミュレーション用の正式な実行に格上げする

## 概要

`ManualRun` をテスト専用から「シミュレーション用の実行」に格上げする。入口は `mujou.run_manually` にして `mujou` 直下から公開し、`mujou.testing` は消す。ノードを渡さない `step(dt)` は全ノードをワーカーで並列に更新する Jacobi にする（ステップ開始時の値だけを読むので決定的）。実時間の `mujou.run` はデプロイ用、`mujou.run_manually` はシミュレーション用の 2 本立てになる。

## 公開インターフェース案

`src/mujou/execution.py` に置き、`mujou/__init__.py` の `__all__` に `"ManualRun"` と `"run_manually"` を足す。`src/mujou/testing.py` は削除する（互換の名前は残さない）。

- `def run_manually[P: Port[Any, Any]](root: Node | Network, *, timer: VirtualTimer | None = None, inputs: Mapping[P, Message] | None = None, state: Mapping[str, Any] | None = None, workers: int | None = None) -> ManualRun`
  - 引数名と順は `mujou.run` と同じ（`timer` の型だけ `VirtualTimer`）。`workers` の意味と既定も `mujou.run` と同じ（`None` は利用できる CPU 数、GIL が有効なら半分、プールのノード数が上限）
  - docstring の要旨: 「`mujou.run` と同じように `root` を走らせるが、ノードは `ManualRun.step` が許したときだけ更新する。シミュレーション用（実時間で回す `mujou.run` は現実世界へのデプロイ用）。既定のタイマーは時刻 0 の新しい `VirtualTimer`（`state` を渡したなら保存した時刻から）。他の実行とタイマーを共有すれば `step` がそちらの時間も進める」。Example は `testing.py` の module docstring の例を移す
  - Raises は `mujou.run` と同じ（`RuntimeError` 既に走ったノード／hook からの呼び出し、`TypeError`/`ValueError` 不正な木・inputs、`ValueError` `workers < 1`／`timer` と `state` の同時指定／形の合わない `state`、`NetworkError` 起動中の例外）
- `class ManualRun(Run)`（名前は変えない）
  - クラス docstring: 「`run_manually` が返す実行。ノードは起動・停止こそ通常の実行と同じだが、更新は `step` を待つ。各 `dt` はちょうど進めた時間で、結果は毎回 bit 単位で同じ。シミュレーションにもテストにも使う」
  - `timer -> VirtualTimer`（property、現状のまま。docstring の「`mujou.testing.run` が作った」を `run_manually` に）
  - `def step(self, dt: float, *nodes: Node) -> None`
    - ノードなし: タイマーを `dt` 進めてから、全ノードを 1 回ずつ同時に（ワーカーと専用スレッドで並列に）更新する。どのノードもステップ開始時の値を読み、このステップで送った値は全ノードの更新が終わってから一斉に見える（Jacobi）。結果はワーカー数やスレッドの順に依らない
    - ノードあり: 渡した順に 1 つずつ更新し、後のノードは前のノードがこのステップで送った値を読む（Gauss-Seidel）。同じノードを 2 回渡すと 2 回目は時刻が進んでいないので飛ばす（現状どおり）。全ノードを渡しても `step(dt)` とは別物である旨を明記（chain なら出力側からの逆順が `step(dt)` と一致する）
    - 周期は無視（現状どおり）。ステップの間ほかのスレッドからホストが読むと、そのポートはステップ前の値を返す
    - Raises: `ValueError`（run に無いノード、`dt` が負か非有限）、`RuntimeError`（正常に止まった後、タイマーが一時停止中、hook からの呼び出し）、`NetworkError`（ノードが例外を出し実行が止まった。そのステップで送った値は公開しない）
- 内部名: `testing.py` が消えると `start` / `Kind` / `RUN` / `MANUAL_RUN` は `execution.py` の中だけで使う名前になるので `_start` / `_Kind` / `_RUN` / `_MANUAL_RUN` にする（skill `refactor-conventions`）

### `_native`（private。`just stub` で `_native.pyi` を再生成）

- 新クラス `Staging()`（frozen、引数なし、Python から呼ぶメソッドなし）: 1 つの手動の実行で、ステップ中の送信を保留する箱
- `Outlet(schema, channels, name, *, clock, staging: Staging | None = None)`
- `start(nodes, clock, *, staging: Staging | None = None, workers: int | None = None) -> Runtime`: `manual: bool` を `staging` に置き換える。`staging` を渡した実行が手動モード（手動なのに保留が無い状態を型で作れなくするため）
- `Runtime.step(order: list[int] | None = None)`: シグネチャはそのまま。`None` の意味を「全ノードを並列に 1 ステップ」に変える
- `src/mujou/wiring.py:97` `wire(tree, clock, staging: _native.Staging | None = None) -> Wiring`、`_outlet` に `staging` を通す

### Rust コア（`mujou-core`）

- `crates/mujou-core/src/slot.rs` に `pub struct Staging`
  - `pub fn new() -> Arc<Self>`
  - `pub(crate)` 相当の操作: `open()`（保留を始める）、`commit()`（保留を公開して閉じる）。`Outlet::send` から使う `stage(slot, msg)`
- `Outlet::new(schema, channels, name, clock, staging: Option<Arc<Staging>>)`
- `RunConfig { nodes, manual: Option<Arc<Staging>>, workers }`（`manual: bool` を置き換え。doc: 「`Some` なら手動モード。ノードは `Runtime::step` が許したときだけ更新し、ステップ中の outlet の送信は `Staging` が保留する」）
- `Runtime::step(&self, order: Option<&[usize]>) -> Reply<()>`: シグネチャ不変。`None` は全ノードを 1 つのグループとして並列に、`Some(order)` は 1 ノードずつのグループの列

## 理論上の対応

- 更新手法には触れない。PC-ALM の式は変わらない。`step(dt)` は「全ノードを固定の Δt で同期的に回す」離散アルゴリズム（Jacobi 型の陽的 Euler）にそのまま一致する。従来 pc のテストが `step(h, *chain.from_next())`（出力側からの逆順）で作っていた Jacobi が既定になる
- 局所性: ノードが読むのは自分の状態と隣接ノードのスロットだけで変わらない。保留はスロットの「いつ見えるか」だけを変え、ノードは自分が手動の実行にいることを知らない（ノード・メッセージ・Driver に手法を焼き付けない）
- 非同期性: ステップ境界は全ノードのバリアになる。これを不変原則の例外として **`ManualRun` にだけ** 認める。理由は停止と同じで、ステップの間はその実行のタイマーが止まっており（`VirtualTimer` は `advance` でしか進まない）、ダイナミクスから見れば各ノードは Δt ごとに 1 回ずつ更新されるだけだから。条件: 更新手法の正しさがこのバリアに依存してはならない（自由実行でも正しく動くこと）。自由実行（`mujou.run`）にはバリアも保留も入れない

## Rust コアの設計

採用案: **送信をスロットごとに 1 つだけ保留し、グループの全ノードが終わったら制御スレッドがまとめて公開する**（ユーザーの想定案を、メモリとロックの点で詰めたもの）。

1. `Staging` は `open: AtomicBool` と `pending: Mutex<Vec<Arc<Slot>>>`（保留を持つスロットの一覧）を持つ
2. `Slot` に保留用の `staged: Mutex<Option<Arc<Message>>>` を足す（読み手が取る `value` の lock とは別。読み手は保留を見ない）
3. `Outlet::send`: 中身を詰めて時刻を刻むところまでは現状どおり（`slot.rs:287`）。`staging` が開いていれば、各対象スロットの `staged` を置き換え、`None` から埋めたときだけ `pending` にそのスロットを積む。閉じていれば（自由実行・起動・停止・取得の hook の送信）今までどおり `publish`。同じステップで同じスロットに何度送っても保留は最後の 1 つで、メモリはスロットあたり高々 1 メッセージ
4. `commit`: `pending` を取り出して各スロットの `staged` を `publish`（`publish` が `wait_newer` の待ち手を起こすので subscribe はそのまま動く）し、`open` を下ろす。この間どのノードも走っていない
5. 制御スレッド（`control.rs:83` の `step`）を `step(groups: Vec<Vec<usize>>)` にする。グループごとに: `staging.open()` → lock の下で `life.grant(group)` と各ノードの `wake` → 全員の許可が終わるか失敗するまで待つ → 失敗なら `end()`、成功なら `staging.commit()`。Gauss-Seidel は「1 ノードのグループの列」なので、1 本のコードで両方を表す（特別扱いを作らない）
6. `lifecycle.rs`: `grant: Option<usize>` を `granted: Vec<bool>` と未完の数に置き換える。`grant(&[usize])`、`granted()` は未完が 1 つでもあれば真、`end_step(i)` は自分の印を下ろし、未完が 0 になったときに `changed` を立てる。`advance` の `Command::Run if self.grant != Some(i)` を `!self.granted[i]` に。時刻が進んでいないノードは今までどおり何もせず自分の許可だけ終える
7. `Runtime::step`（`mod.rs:197`）: `None` → `vec![0..n]`、`Some(order)` → 各要素を 1 要素のグループに。手動でない runtime と範囲外の index を断る検査は現状のまま

決めたこと（理由を 1 行ずつ）:

- **失敗時**: 失敗したステップの保留は公開しない。`Staging` は開いたまま残し、以後の送信（`on_exception` / `on_stop`）も保留されて実行とともに捨てられる。失敗後のホストの読み出しは `NetworkError`（`Run._check`）なので、利用者からは見えない。片付けのコードを増やさないため
- **正常終了**: `shutdown` はステップの間にしか来ない（制御スレッドが直列化）ので `Staging` は閉じており、`on_stop` の送信はそのまま公開されて終了後の `read` / `state_dict` に出る
- **状態の取得・復元**: 取得はステップの間なので保留は空。`Slot::restore` は起動前なので無関係
- **GPU テンソル・stream**: 書き手は `send` の中で自分の stream を待ってからコアに渡す（S2）ので、保留中のメッセージは完成している。保留は公開を遅らせて Arc を長く持つだけで、読み手の `record_stream` も device の上の扱いも変わらない。追加の device メモリはスロットあたり 1 メッセージ
- **ホストの read / subscribe**: ステップ中にほかのスレッドから読むと、各ポートはステップ前の値（公開の途中ならポートごとに前か後）。subscribe は commit の `publish` で起きる。ホストの `write`（`InputGroup`）は保留しない。ステップ中に別スレッドから書いた値が同じステップで読まれるかは決まらないので、docs で「write はステップの間に」と書く
- **dedicated_thread のノード**: 許可されると自分のレーンのワーカーが起きる（`Shared::wake` が既にレーンを見る）。プールのノードと同じグループで並列に走る
- **workers の既定**: `mujou.run` と同じ（`crates/mujou-python/src/runtime.rs:238` の `default_workers`）。手動の実行でも並列に回るようになったので、`testing.run` が `workers` を持たなかった理由（`tests/mujou/test_testing.py:17` のコメント）が消える
- **メッセージの時刻**: ステップの時刻（タイマーはステップ中動かない）。並列でも逐次でも同じ

比べて採らなかった案:

- 版番号で切り替える（`InputGroup` と同じ `Pair.next` + 公開の版）: 公開が原子的になるが、`Source::Own` のスロットにも版の参照と、wake 用のスロット一覧が要り、結局一覧を持つ。ステップ中のホストの読み出しが原子的である必要は無いので、単純な方を採る
- ノードがステップ開始時に入力を写し取る 2 相のステップ: lifecycle に新しい Work と Driver の変更が要り、ホストの読み出しも途中の値を見る
- `RunClock` に保留を持たせて配線の引数を増やさない: 時間の型がスロットに依存し（`time` → `slot` の循環）、自由実行のコードにも手動の概念が入る

## 実装ステップ

各ステップの検証は記した順に。最後に `just run`。

1. **slot の保留**（`crates/mujou-core/src/slot.rs`）: `Staging`、`Slot.staged`、`Outlet::new` の引数、`send` の分岐。既存の `Outlet::new` 呼び出し（`slot.rs` のテストと `runtime/mod.rs` の doc、`crates/mujou-python/src/slot.rs:415`）に `None` を渡す
   - 検証: `cargo test -p mujou-core slot`（下記テスト観点の slot の項）
2. **lifecycle の複数許可と制御のグループ**（`lifecycle.rs`、`control.rs`、`shared.rs:73` の `manual: bool`、`mod.rs` の `RunConfig` と `Runtime::step`、モジュール doc の「stepping by hand」の説明）
   - 検証: `lifecycle.rs` の単体テスト、`crates/mujou-core/tests/runtime/manual.rs` に並列の結合テスト、`main.rs:297` と `period.rs:322` の `manual: true` を `Some(Staging::new())` に。`just test-rs`
3. **bindings**（`crates/mujou-python/src/slot.rs` に `PyStaging`、`PyOutlet::new` の `staging`、`runtime.rs:296` の `start` の `manual` → `staging`、`step` の doc comment、`time.rs:156` の doc の `mujou.testing.run` → `mujou.run_manually`）→ `just stub`
   - 検証: `just test-rs`、`tests/mujou/native/test_runtime.py`（`manual=True` 15 か所を `staging=_native.Staging()` に）と `tests/mujou/test_wiring.py`、`test_state.py` の `wire(` 呼び出しが通ること
4. **Python の実行**（`wiring.py` の `wire` / `_outlet`、`execution.py` の `_start` で `staging = _native.Staging() if kind.manual else None` を作って `wire` と `_native.start` に渡す、`run_manually`、`ManualRun` の docstring、module docstring の `mujou.testing.run`、内部名の `_` prefix。`__init__.py`。`testing.py` を削除）
   - 検証: `uv run pytest tests/mujou/test_execution.py -v`、`just type`
5. **テストの移行**（下記「既存テストの直し方」）
   - 検証: `just test-fast`
6. **docs / examples / benchmarks**（下記「docs の変更」）
   - 検証: `just docs-build`、`just type`（examples を含む）
7. **specs / memory / CLAUDE.md / skill**（下記）。`.claude/` の編集は skill `edit-dot-claude` の手順で
   - 検証: `just format`（mdformat・codespell）
8. CHANGELOG を書いて `just run`

### 既存テストの直し方

調べた結果、ノードなしの `step(dt)` は 107 か所あり、Jacobi で結果が変わるのは **2 か所だけ**（依頼文の「約 38 か所」より少ない。chain の多くのテストは `net.a.x` だけを見るか、`read(net.next)` と `net.b.x` の一致を見るので Jacobi でも成り立つ）。

- `tests/mujou/test_port.py:478` `TestSend.test_reaches_every_peer`: 送り手と受け手が同じステップで更新される前提。テストの関心は fan-out なので `run.step(H, source, *sinks)` と順を明示する
- `tests/mujou/test_testing.py:48` `test_advances_the_timer_by_dt_then_updates_in_the_given_order`: 後半 2 つが木の順を仮定。前半（渡した順）はそのまま、後半は「a と b が 1 回ずつ」（順は問わない）に直し、コメント「every node, in the order of the tree」も直す
- 注意して見る所: `tests/mujou/pc/test_chain.py:91` / `:231` の「学習した」の assert（Jacobi ではサンプルが出力に届くまで約 3 ステップ。20 ステップ・`LEARN_INTERVAL = 3*H` なので通る見込み）
- `tests/mujou/native/test_runtime.py` の `order=None` のテストはポートの無いノードの hook 記録だけを見るので影響なし
- pc のテストの `step(h, *chain.from_next())` は今回は書き換えない（変更範囲を絞る。`step(h)` と一致することは新しいテストで 1 回だけ示す）

ファイルの移動・置換:

- `tests/mujou/test_testing.py` → 中身を `tests/mujou/test_execution.py` の `class TestRunManually`（公開名・シグネチャ・既定のタイマー）と `class TestManualRunStep` に移して削除（`tests/` は `src/` をミラーするため）。`test_run_takes_the_arguments_of_mujou_run_but_workers` は「引数が `mujou.run` と同じ（`workers` を含む）」に反転
- `tests/mujou/conftest.py`: `import mujou.testing` と `from mujou.testing import ManualRun` を `mujou.run_manually` / `from mujou import ManualRun` に。`StartManual` と `start_manual` に `workers: int | None = None` を足す
- `mujou.testing` を参照する残り: `tests/mujou/test_driver.py`、`test_node.py`、`test_port.py`、`test_state.py`、`test_pyright.py`（171〜195 行の `mujou.testing.run` と、`run(a, b)` の型エラーの例）、`test_execution.py`、`pc/test_input.py` / `test_layer.py` / `test_output.py`、`tests/mujou/test_init.py:39`（サブモジュールの一覧から外す）、`tests/test_layers.py:11,37,64`（層と `PUBLIC` から `mujou.testing` を外す）。機械的に `mujou.run_manually` へ
- `benchmarks/ports.py:30,109`、`benchmarks/pc_dense.py:16,38,124`: `mujou.run_manually` に。`pc_dense.py` の「manual」の計測は並列の Jacobi ステップを測ることになる旨を docstring に（過去の数値とは比べられない）

## テスト観点

### Rust（`crates/mujou-core`）

- 正常系
  - slot: 保留を開いた outlet の `send` は `load` に見えず、`commit` 後に見える。`Target::All` と `Target::Channel(k)` の両方
  - slot: 1 ステップで同じスロットに 2 回送ると、commit 後に見えるのは 2 回目だけ
  - slot: `wait_newer` で待つ読み手が `commit` で起きる（保留の段階では起きない）
  - slot: 保留を閉じた outlet の `send` は今までどおりすぐ見える
  - lifecycle: 2 ノードを同時に許可し、1 つ目が終わっても `granted()` は真、2 つ目で偽
  - runtime（`tests/runtime/manual.rs`）: `step(None)` で全ノードが 1 回ずつ同じ dt で更新される。ワーカー 2 のプールで、更新の中で互いを待つ 2 ノード（タイムアウト付きの待ち合わせ）が両方とも完了する（並列に走っている証拠。逐次なら待ち合わせがタイムアウトする）
  - runtime: プールのノードと `dedicated_thread` のノードが同じ並列ステップで更新される
- 異常系
  - 並列ステップで 1 ノードの更新が失敗すると、ほかのノードの更新は最後まで走り、`step` は失敗を返し、全ノードが `exception` → `stop` の順で 1 回ずつ終わる
- エッジケース
  - 時刻が進んでいない `step(None)` は誰も更新せずに返る（許可が全員分終わる）
  - `Some(&[2, 0, 1])` の逐次は今までどおり渡した順

### Python（`tests/mujou/test_execution.py` ほか）

- 正常系
  - `mujou.__all__` に `ManualRun` と `run_manually` がある。`import mujou.testing` が `ModuleNotFoundError`
  - `run_manually` の引数の名前と順が `mujou.run` と同じ。既定のタイマーは時刻 0 の新しい `VirtualTimer`、`state` を渡すと保存した時刻から
  - 並列ステップのノードはそれぞれ `dt == h` を見る。送ったメッセージの時刻はステップの時刻
  - 周期 `math.inf` のノードも `step(dt)` で更新される
  - `workers=1` と `workers=4` で同じ結果（`torch.equal`）
  - ステップ中にほかのスレッドからホストが読むと、ステップ前の値が返る（`on_update` で送った後 `threading.Event` で止まるノードを使い、止まっている間にホストが読む）。`step` が返った後は新しい値
  - `subscribe` は並列ステップの後の値を受け取る
  - 状態を途中で取って新しいインスタンスで復元し、並列ステップを続けた結果が止めずに続けた結果と bit 一致
- 異常系
  - 並列ステップで 1 ノードが例外 → `step` が `NetworkError`（`__cause__` がその例外）。以後の `read` も `NetworkError`。全ノードの `on_exception` → `on_stop`
  - `workers=0` は `ValueError`（`mujou.run` と同じ文言）
  - 既存の K19（run に無いノード → `ValueError`）、負・非有限の `dt`、正常終了後の `step` の `RuntimeError` は `TestManualRunStep` に移して維持
- エッジケース
  - 起動直後の `step(0.0)` は誰も更新しない（既存の `test_execution.py:1332` 相当を並列で維持）
  - ノード 1 つの実行では `step(dt)` と `step(dt, node)` が同じ
- 数値的オラクル（skill `testing-strategy`）
  - **chain の一致**: 3 ノードの chain（`x_next = a·x_prev + b` のような決定的な更新を float64 で）を 2 つ作り、片方を `step(h)`、もう片方を `step(h, c, b, a)`（出力側から逆順）で k 回。全ノードの状態・全ポートの値と時刻が `torch.equal`
  - **閉路の Jacobi**: a ↔ b の 2 ノードの閉路（各ノードが受け取った値に自分の定数を足して送る）。`step(h)` の後の値が手で計算した Jacobi の値と一致し、どの逐次の順とも一致しない
  - **再現性**: 同じ seed で 2 回走らせ、`workers=4` の並列ステップ k 回の結果が bit 一致（10 回程度繰り返し、スレッドのタイミングに依らないこと）
  - **pc**（`tests/mujou/pc/test_chain.py`）: float64 の `pc.Chain` を `step(h)` と `step(h, *chain.from_next())` で同じ回数回し、重み・r・λ が bit 一致
  - **GPU**（CUDA が無ければ skip）: CUDA の 2 ノードの chain で並列ステップと逆順の逐次が一致

テンソルは小さく（intra-op の並列が入らない大きさ）して bit 一致を問う（リスク参照）。

## docs の変更

- **新設** `docs/en/topics/simulation.md` / `docs/ja/topics/simulation.md`（nav は Topics の「Time」の後に `Simulation: topics/simulation.md`、`nav_translations` に `Simulation: シミュレーション`）。教材サイト級（memory `feedback_docs_teaching_quality.md`）で:
  - 2 つの走らせ方の表（`mujou.run`: 実時間・デプロイ / `mujou.run_manually`: 仮想時間・計算能力いっぱい・シミュレーションとテスト）
  - `step(dt)`（Jacobi、並列）と `step(dt, *nodes)`（Gauss-Seidel、逐次）の違いを図（mermaid の sequenceDiagram）と表で。chain で `step(h)` が `step(h, c, b, a)` と一致する例
  - 並列と workers、`dedicated_thread`。**注意**: ワーカーごとに torch の intra-op スレッドが走るので CPU を取り合う。`torch.set_num_threads` で抑える目安を書く（実装はしない）。GIL ありのビルドでは並列にならないが結果は同じ
  - 決定的であるための条件（`self.generator`、時間を動かすのは `step` だけ、`write` はステップの間に、ノードどうしで Python のオブジェクトを共有しない）
  - ステップ境界は手動の実行だけの仕組みで、更新手法の正しさをそれに頼らないこと
  - 保存と再開、例外、周期を無視すること
- **tutorial 3** `docs/{en,ja}/tutorial/3-test.md`: タイトルは「3. Test a Node」のまま（テストの章として残し、仕組みの詳細は topic へ）。`mujou.testing` の節を `mujou.run_manually` に書き直し、「import mujou.testing が要る」の記述を消す。ツール表の `workers` の行を直し、「The order of the updates」を「`step(dt)` は全ノードを同時に、ステップ開始時の値で」「順を決めたいときは `step(dt, *nodes)`」に。末尾から Simulation へリンク
- `examples/test_leaky.py`: `import mujou.testing` → `mujou.run_manually`、module docstring（4 行目の「fixed order」）、chain の節に `step(h)` と `step(h, net.b, net.a)` の一致を足す
- **API**: `docs/{en,ja}/api/testing.md` を削除し、`api/run.md` に `::: mujou.run_manually` と `::: mujou.ManualRun`（`members: [step, timer]`、`inherited_members: false`）を足す。mkdocs.yml の nav から `Testing: api/testing.md` を外す（`nav_translations` の対応する行も）
- `docs/{en,ja}/concepts.md` の 8 節（en 397〜430 行付近）: 見出しを「Simulation and Testing」/「シミュレーションとテスト」にし、表の列を `mujou.run_manually` に、`step` の説明を Jacobi に、mermaid を並列ステップに。目次の表（38 行）も
- `docs/{en,ja}/topics/time.md:11`、`topics/errors.md:25`、`examples.md:31`、`index.md:41,74`（「Deterministic tests」を「Simulation and deterministic tests」に）
- `CONTRIBUTING.md:69,97`
- `CHANGELOG.md` の `[Unreleased]`:
  - Added: `mujou.run_manually` and `mujou.ManualRun` at the top level, for simulations; `workers=`
  - Changed: `ManualRun.step(dt)` without nodes updates every node in parallel from the values at the start of the step (Jacobi), instead of one after another in the order of the tree
  - Removed: `mujou.testing` (use `mujou.run_manually`)

## specs / memory / CLAUDE.md / skill の変更

- `specs/execution-model.md`: 71 行の表を `mujou.run_manually(root)` に。「停止はシステム全体のバリアだが…例外として認める」の段落（195 行付近）に、手動の実行のステップ境界も同じ理由で例外とすることを足す。207 行の `Loop` を「ステップの許可（同時に複数）」に。「## テスト」節を「## 手動の実行（シミュレーション）」に書き直す（2 本立ての位置づけ、Jacobi / Gauss-Seidel、送信の保留と公開、失敗時は公開しない、workers、周期の無視、オラクルとしての使い方）
- `specs/scenarios.md`: S6 を「シミュレーション・テスト | `mujou.run_manually` と `ManualRun.step(dt, *nodes)`」に。53 行、K19、104〜105 行、115 行の `mujou.testing` を直す。並列ステップの例外（K 番号を新設: 並列ステップ中の例外 → `NetworkError`、保留は公開しない）を足す
- `specs/architecture.md:29-31,38,40,126`: `testing.py` の行を消し、公開の入口から `mujou.testing` を外す
- `specs/native-core.md:32`: `step(order)` の `None` の意味、`Staging` と `Outlet` / `start` の `staging` を書く
- `specs/pc.md:371`（`mujou.testing.run` → `mujou.run_manually`）。186・192 行は意味が変わらないので文言だけ確認
- `CLAUDE.md` 52 行: 「例外は状態の取得・復元や観測のための停止だけで、その実行のタイマーごと止める。」を「例外は状態の取得・復元や観測のための停止と、`ManualRun`（`mujou.run_manually`）のステップ境界だけで、どちらもその実行のタイマーが止まっている間に限る。」の趣旨に。ツーリングや索引で `mujou.testing` に触れていないことも確認
- `.claude/skills/locality-and-async/SKILL.md`: 73 行の例外に「手動の実行のステップ境界（タイマーは `step` でしか進まない）。更新手法の正しさをこれに頼らない」を足す。112 行のテストの説明を `mujou.run_manually` と並列の `step(dt)` に
- `.claude/skills/testing-strategy/SKILL.md:46,64,65`: `mujou.testing` を外し、`start_manual` の説明と `step` の意味（ノードなしは Jacobi）を更新
- memory:
  - `memory/project_testing_api.md` を `memory/project_manual_run_simulation.md` に置き換える（旧ファイルは削除）。内容: 2026-11-16 のユーザーの決定 1〜5、Why（シミュレーションを正式な用途に、計算能力をフルに使う）、How to apply（入口・Jacobi / GS・保留・バリアの例外・torch のスレッドは docs だけ）
  - `MEMORY.md` の「テスト用の公開の口」の行を「[手動の実行はシミュレーション用](project_manual_run_simulation.md) — `mujou.run_manually` と `ManualRun` を直下に公開、`mujou.testing` は廃止。`step(dt)` は並列の Jacobi、`step(dt, *nodes)` は逐次。ステップ境界のバリアは ManualRun だけの例外（2026-11-16）」に
  - 現行の規約を述べている箇所だけ直す: `project_public_modules.md:17`、`project_pc_module.md:11`、`project_runtime_api_revision.md:10`（公開モジュールの一覧から `mujou.testing` を外す）、`project_mn_scheduler.md:11`（「`mujou.testing.run` には無い」→ `run_manually` にもある）、`project_async_node_execution.md:25`。`memory/agents/` 配下の過去の記録は直さない

## 想定リスク・トレードオフ

- **torch の intra-op 並列と bit 一致**: 複数のワーカーが同時に大きな CPU 演算を出すと、intra-op のスレッドプールの使われ方が変わる。分割はスレッド数で決まるので結果は変わらない見込みだが、確かめていない。テストは小さいテンソルで bit 一致を問い、大きなものは docs で注意するに留める
- **利用者のノードが Python のオブジェクトを共有している場合**（グローバルなリストへの追記など）: 並列ステップで非決定的になる。従来の逐次ステップでは表に出なかった。docs の決定性の条件に書く
- **ステップの所要時間**: 全ノードを待つので、最も遅いノードがステップを律する（Jacobi の本質）。workers が少ないと逐次と大差ない
- **ホストの読み出しの原子性**: commit はスロットごとに公開するので、ステップ中に別スレッドから 2 つのポートを読むと片方だけ新しいことがある。版番号の案なら防げるが、用途に対して複雑さが見合わない
- **失敗後の保留を捨てる**: 失敗したステップで他のノードが送った値は観測できない。失敗後はホストの呼び出しがすべて `NetworkError` なので、今の契約の範囲では差が出ない
- **benchmarks/pc_dense.py の手動の数値**: 意味が逐次から並列に変わる

## 確認事項

1. **0.1.0 は 2026-10-30 に PyPI へ公開済み**（`CHANGELOG.md`）。memory `feedback_no_compat_before_release.md` は「公開した後は改めて方針を確かめる」としている。依頼文では「公開前なので互換を残さない」とあるが、前提が違う。暫定案: 0.x の破壊的変更として `mujou.testing` を単に消し、CHANGELOG の Removed に書く（deprecation は入れない）。あわせて、この memory の方針を「0.x の間は互換を残さない」に更新してよいか
2. 依頼文の「Gauss-Seidel の順に依存するテスト約 38 か所」は、調べた限り 2 か所（`test_port.py:478` と `test_testing.py:48`）。ほかに想定している箇所（ノード指定の `step` を `step(dt)` に置き換えたい箇所など）があれば教えてほしい。暫定案: 2 か所だけ直し、pc テストの `step(h, *chain.from_next())` は書き換えない

## 参照

- `src/mujou/execution.py:531-592`（ManualRun）、`:673-736`（Kind、start）、`src/mujou/testing.py`、`src/mujou/wiring.py:97-192`
- `crates/mujou-core/src/slot.rs:91-300`（Slot、Outlet、InputGroup）、`runtime/control.rs:83-103`（step）、`runtime/lifecycle.rs:105-370`（grant / end_step）、`runtime/mod.rs:61-72,195-207`、`runtime/shared.rs:66-100`
- `crates/mujou-python/src/runtime.rs:233-367`、`crates/mujou-python/src/slot.rs:397-445`
- `specs/execution-model.md` の「RunClock」「ライフサイクルの遷移」「テスト」、`specs/scenarios.md` S6 / K19
- skill `locality-and-async`、`testing-strategy`、`refactor-conventions`
