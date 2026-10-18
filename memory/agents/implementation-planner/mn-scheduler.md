# M:N スケジューリング

## 概要

ノードを、ノード数 M に対して N 本のワーカースレッドで回す（既定でワーカーに載せる。Python のノードも載せる）。ユーザーの要求は「大量のスレッドを立てると OS のコンテキストスイッチを取り合うので、M:N でなければならない」（2026-10-18）。周期・停止・状態の取得・手動ステップ・例外による終了の意味は変えない。ブロッキングする I/O のノード向けに、専用スレッドを opt-in で残す。

前提: 別 PR「generator を `super().__init__` で作る」が先に入る。`_on_thread_start` の generator の生成は、この計画では扱わない。

## 公開インターフェース案

### Python（利用者向け）

- `mujou.run(component, *components, timer=None, inputs=None, state=None, workers: int | None = None) -> Run`
  - `workers`: この実行のワーカーの数。`None` なら `min(ワーカーに載るノードの数, 論理 CPU 数)`。論理 CPU 数はコアの `std::thread::available_parallelism()` で取る。`int` で 1 以上。`bool` は `TypeError`、0 以下は `ValueError`。プールは実行ごとに作る（確認事項 Q2）
  - ワーカーに載るノードが 0 個ならワーカーは立てない
- `mujou.testing.run(...)`: 引数は増やさない。手動モードは 1 度に 1 ノードしか回さないので、ワーカーは既定の数で足りる
- `Node.__init__(*, device="cpu", seed=None, period=0.01, dedicated_thread: bool = False)`
  - `dedicated_thread=True` のノードは、今までどおり自分の OS スレッドで動く。`bool` 以外は `TypeError`（`period` と同じく `isinstance(x, bool)` で判定する）
  - property `Node.dedicated_thread -> bool` を足す。`dedicated_thread` は予約名にする（`period` と同じく、クラス属性で置き換えたら `mujou.run` で `TypeError`。`checked_period` と同じ形の `checked_dedicated(node) -> bool` を `node.py` に置く）
- `on_thread_start()`: 呼ぶのは `dedicated_thread=True` のノードだけ（自分のスレッドで `on_start` の直前に呼ぶ）。ワーカーに載るノードのクラスが `on_thread_start` を override していたら、`mujou.run` が何も起動しないうちに `TypeError` を出す（メッセージで `dedicated_thread=True` を案内する）。確認事項 Q1
- hook を呼ぶスレッドについて、利用者が知るべき変化（docstring と docs に書く）
  1. ワーカーに載るノードの hook は、呼ばれるたびに別の OS スレッドで走りうる。ただし 1 つのノードの hook が同時に 2 つ走ることはなく、連続する 2 つの hook の間には happens-before がある（前の hook の書き込みは次の hook から見える）
  2. `contextvars` はノードごとに保たれる（ノードごとの `contextvars.Context` の中で hook を呼ぶ）。`threading.local`・`threading.current_thread()`・`threading.get_ident()` は hook ごとに変わりうる
  3. 勾配の記録の停止・CUDA の current device と current stream は、フレームワークが hook の前に毎回そのノードの設定に戻す。それ以外のスレッドごとの torch の設定（`torch.autocast` など）は、hook の中の `with` に閉じる。hook の外に持ち越さない
  4. hook の中でブロッキングする処理（`time.sleep`、I/O の待ち、lock の待ち）はワーカーを占有する。そのあいだ、ほかのノードの $\Delta t$ が伸びる。そういうノードは `dedicated_thread=True` にする
  5. torch の内部スレッド数の目安を「CPU で計算するノードの数 × n」から「ワーカーの数（+ 専用スレッドのノードの数）× n ≤ 物理コア数」に改める

### native（`mujou._core._native`、private）

- `Runtime.start(*, manual=False, states=None, last_update_ns=None, periods=None, timer_state=None, workers: int | None = None, dedicated: list[bool] | None = None) -> None`
  - `dedicated` を省略すると、全ノードがワーカーに載る。長さが違えば `ValueError`
- 内部の hook `Node._on_thread_start()` を `Node._on_enter()` に置き換える。呼ばれるのは、ノードの hook を走らせるスレッドが直前に別のノードを走らせていたとき（またはどのノードも走らせていなかったとき）で、次の hook の直前に呼ぶ。専用スレッドなら起動時の 1 回だけになる。ここで出た例外は、次に呼ぶ公開の hook の名前で報告する

### Rust コア（`mujou_core::runtime`）

- `RuntimeConfig` に 2 つ足す
  - `pub workers: Option<NonZeroUsize>`: `None` なら `min(ワーカーに載るノード数, available_parallelism)`
  - `pub dedicated: Option<Vec<bool>>`: `None` なら全ノードを false とする。長さが違えば `RuntimeError::InvalidArgument`
- `NodeHooks` に 1 つ足す
  - `fn enter(&mut self, next_hook: &'static str) -> Result<(), HookError> { Ok(()) }`: スレッドがこのノードに切り替わったとき、`next_hook` の直前に、同じ `invoke` の中で呼ぶ
- `NodeHooks::on_thread` は残す。ワーカーの本体も `N::on_thread` で包み、Python のスレッド状態はワーカーの寿命の間 1 つにする
- 内部の純関数 `fn next_target_ns(target_ns: u64, now_ns: u64, period_ns: u64) -> u64` を `node.rs` から切り出す（今の node.rs:127-130 の規則）。専用スレッドとワーカーで共有する

## 理論上の対応

- 更新式には触れない。PC-ALM にも、ほかの更新手法にも依存しない（ワーカーが知るのは `NodeHooks` だけ）
- 局所性: ノードが読めるのは自分の状態と自分の Port のスロットだけ、は変えない。Port の所有の判定を「スレッドの ID」から「そのスレッドが今走らせているノード」に変える。ワーカーが同じでも、ほかのノードの Port は読めない
- 非同期性: ワーカーは 1 つのノードの `on_update` を 1 回走らせるごとにキューへ戻る。ノード同士が互いを待つことはない（待つのはワーカーの割り当てだけで、$\Delta t$ が伸びる形で連続時間の定式化に吸収される）。グローバルなバリアは今と同じく、停止・状態の取得・終了だけ
- 同期実行（手動ステップ）は今と同じく、非同期実行の特殊なスケジュールのまま。同期専用の経路は作らない
- 実行の配置（どのワーカーで回すか）を変えても、非同期の境界は変わらない（`specs/nodes.md#非同期の境界と実行の配置`）

## Rust コアの設計

### 構造

- `Runtime<N>` は 1 つの型 `N` のまま、ジェネリックで残す。今あるノードは全部 `PyNode` なので足りる。プールのノードと専用スレッドのノードは同じ `N` で、違いは `RuntimeConfig::dedicated` だけ。ネイティブのノードを混ぜるのは、それが入るときに enum で包む（`NodeHooks` は `on_thread` が generic で `type State` も持つので、`dyn` にはできない）。今は `dyn` 化しない（必要が無い）
- 新しいサブモジュール `runtime/pool.rs` にワーカーとスケジューラを置く。`node.rs` は専用スレッドのループとして残す
- `Shared<N>` に足すもの
  - `sched: Mutex<Sched>` と `sched_changed: Condvar`
  - `sched_epoch: AtomicU64`: タイマーで待つワーカーを起こすための語。キューの先頭が早まったら増やす
  - `dedicated: Vec<bool>`
- lock の順序は control → sched → timer の一方向にする。sched を持ったまま control を取らない。hook は、どの lock の外でも呼ぶ（今の規則と同じ）

### スケジューラの状態（`Sched`）

- ノードごとの `Slot` 状態。プールのノードは、どの時点でも次のどれか 1 つにある
  - `Task(Lifecycle)`: 起動・一時停止・再開・例外・終了の hook の実行待ち（期限 −∞ の扱いで、更新より先に取る）
  - `Queued { target_ns }`: 期限つきの更新待ち。ヒープに入っている
  - `Running`: どれかのワーカーが持っている（ヒープにもタスクの列にも無い）
  - `Parked`: 起動後の go 待ち、一時停止中、手動モードで許可待ち、終了の許可待ち、終了済み
- 期限のヒープ `BinaryHeap<Reverse<(target_ns, index)>>`。期限が同じなら添字の小さい方が先（決定的にする）
- ライフサイクルのタスクの列 `VecDeque<(index, Lifecycle)>`
- 排他の保証: `Queued` / `Task` から `Running` に移るのは、sched の lock の下で取り出したときだけ。`Running` から戻すのは、それを持つワーカーだけ。だから 1 つのノードを 2 つのワーカーが同時に走らせることはない。念のため `hooks[i]` の `Mutex` は残し、debug ビルドではワーカーが `try_lock` に失敗したら panic する（`debug_assert`）

### ワーカーのループ

1. sched の lock を取る。タスクの列が空でなければ先頭を取り、`Running` にして lock を放し、その hook を走らせる
2. 列が空なら、制御語を読む（node.rs の `boundary` と同じ手順。語を読む → 時刻を読む → 語を読み直し、変わっていたら時刻を捨てる）。`Run` で、ヒープの先頭の期限が今の時刻以下なら、取り出して `Running` にし、lock を放して `update(dt)` を走らせる。$\Delta t$ はナノ秒の差から作る（今と同じ）
3. 期限がまだ来ていなければ、待つワーカー（timekeeper）を 1 本だけ置く。timekeeper は lock を放して `timer.wait_until_ns_or(先頭の期限, MAX, || sched_epoch か制御語が変わった)` で待つ。ほかのワーカーは `sched_changed` で待つ。timekeeper は起きたら 1 から繰り返し、仕事を取ったら別のワーカー 1 本を起こして timekeeper を譲る（一斉に起こさない）
4. 更新の後: `last_update_ns` を書く → `next_target_ns` で次の期限を求める → lock の下で `Queued` に戻す。新しい期限が今の先頭より早ければ `sched_epoch` を増やし、`timer.interrupt_waiters()` を呼ぶ
5. スレッドの局所記憶 `LAST_NODE: Cell<(usize, usize)>`（ランタイムの ID とノードの添字）が今から走らせるノードと違えば、hook の前に同じ `invoke` の中で `enter(next_hook)` を呼び、`LAST_NODE` を更新する。専用スレッドは起動時に 1 回呼ぶ
6. ワーカーは `RUNTIME_OF_THREAD` を設定する（ワーカーからの要求は `Reentrant` になる）。プールのノードがすべて `Stopped` になったら抜ける

### 停止・取得・手動ステップ・例外・終了との統合

制御スレッド（control.rs）はほぼそのまま使う。各ノードの `Report`（`started` / `paused_at` / `exited` / `exception`）をプールのノードも同じように報告するので、制御側が待つ条件は変わらない。変わるのは、ノードの側で状態を進める主体が「ノード専用スレッドのブロッキング待ち」から「sched のタスク」になる点。プールのワーカーは、ノードごとに待ってはならない（N < M ならデッドロックする）。

- 起動: プールのノードごとに `Task(Start)` を積む（start = `enter` → binding の起動 hook 群）。制御側が全ノードの `started` を待ち、`go` を立てたら、`set_command` と同じ場所で、全ノードを最初の期限（最初の Δt の起点 + 周期）でヒープに入れる
- 一時停止（capture）: `set_command(Pause)` のとき、sched の lock の下で、`Queued` の全ノードをヒープから出して `Task(Pause(epoch))` に積む。`Running` のノードは、持っているワーカーが更新を終えたところで `Task(Pause)` に回す。`on_pause` の後は `paused_at = Some(epoch)` を報告し、`Parked` にする。凍結より後に始まる `on_update` が無いことは、手順 2 で語を読み直すことで保つ（今の証明と同じ）
- 再開: `set_command(Run)` で `run_epoch > epoch` になったら、`Parked` の各ノードに `Task(Resume)` を積む。`on_resume` の後、停止前の期限でヒープに戻す（時刻は凍結していたので、期限は動かない。`a_capture_does_not_move_the_target` を満たす）。一時停止中に `Stop` が来たら `on_resume` を呼ばない（node.rs:197-203 と同じ規則）
- 手動モード: 制御側が `grant = Some(i)` を立てたら、sched にノード i の `Task(Step)` を積む（control → sched の順で取る）。ワーカーは $\Delta t > 0$ なら 1 回だけ更新し、そうでなければ何もしないで `grant = None` にする。周期は使わない
- 例外: どのワーカーで hook が失敗しても、今の `Shared::fail` を呼ぶ（制御語を `Stop` にして、制御スレッドへ `Failed` を送る）。ワーカーは抜けずに仕事を続ける
- 終了（`Stop`）: sched の lock の下で、`Queued` / `Parked` の全ノードを `exited` にする。`Running` のノードは、持っているワーカーが hook を終えたところで `exited` にする。`failed` なら、`exception == NotCalled` の各ノードに `Task(Exception)` を積む。正常に終了した後に例外が来た場合も、`fail` のときに同じく積む（node.rs:218-232 の規則をタスクで表す）。制御側が `stop_allowed` を立てたら、プールの全ノードに `Task(Stop)` を積む。全ノードが `Stopped` になったらワーカーは抜け、`Controller::finish` がワーカーと専用スレッドを join する（`Controller.threads` に両方入れる）
- `abandon`（スレッドを立てられなかったとき）: ワーカーを 1 本も立てられなければ今と同じく失敗を返す。一部しか立たなかった場合も、プールのノードはタスクとして残るので、立ったワーカーが終了まで進める

### 専用スレッドのノード

`run_node` / `node_loop` / `node_finish` をそのまま使う。変更点は 3 つだけ: 起動時に `enter("on_start")` を呼ぶ、目標の計算を `next_target_ns` にする、`dedicated[i]` が真のノードにだけスレッドを立てる。

## Python のノードをワーカーで動かすときの設定の付け替え

| 設定 | 今 | M:N での扱い | 費用の見込み（ベンチマークで確かめる） |
| --- | --- | --- | --- |
| 勾配の記録 | `_on_thread_start` で `set_grad_enabled(False)` を 1 回 | `_on_enter` で毎回 `torch.set_grad_enabled(False)` | 1 µs 未満（スレッドローカルのフラグ） |
| CUDA の device | `_on_thread_start` で `set_device` を 1 回 | `_on_enter` で `torch.cuda.set_device(self._device)` | 数 µs |
| CUDA の stream | `_on_thread_start` で作り、`set_stream` を 1 回 | 最初の `_on_enter` で作って `self._stream` に保持し、毎回 `torch.cuda.set_stream(self._stream)` | 数 µs |
| 所有スレッドの判定 | `_thread_ident = get_ident()`、Port は ID を比べる | `hooks.py` の `threading.local` に「今走らせているノード」を持つ。`_on_enter` が `enter_node(self)` で設定し、Port は `current_node() is self._owner` で比べる。`thread_ident` / `end_thread` / `_thread_ident` は消す | 数十 ns |
| hook の中の判定 | `enter_node_thread()` でそのスレッドを永久に「中」にする | 同じ。ワーカーは hook しか走らせないので、最初の `_on_enter` から永久に「中」でよい。`in_hook()` は変えない | なし |
| contextvars | スレッドに 1 つ | `PyNode` が `contextvars.Context()` を 1 つ持ち、hook をすべて `ctx.run(method, *args)` で呼ぶ（専用スレッドのノードも同じにして経路を 1 本にする）。`_capture_state` はノードの文脈の外で呼ぶ（今と同じ。終了後にホストが呼ぶ経路と揃えるため） | 呼び出し 1 回ぶん、0.1〜0.3 µs |

- `_on_enter` は、切り替わったときだけ呼ぶ（Rust の `LAST_NODE`）。同じノードが同じワーカーで続けて走るときは呼ばない
- 合計で CPU のノードは 1 回の切り替えにつき 1〜2 µs、CUDA のノードは 5 µs 前後と見込む。OS のコンテキストスイッチ（数 µs + キャッシュの汚れ）と同程度で、torch の `on_update`（数十 µs 以上）より小さい。ベンチマークで超えていたら、勾配モードの設定を「ワーカーの起動時 1 回 + 切り替え時に `is_grad_enabled()` が真のときだけ」に絞る
- `_after_update` のストリームの同期は今のまま。切り替えの前に、そのノードの仕事は自分のストリームで終わっている

## GIL ありのビルドでの振る舞い

- ワーカーは寿命の間 Python に attach したまま、待つ間は detach している（今の `on_thread` と同じ）。hook を呼ぶ間だけ GIL を取る
- Python のコードは GIL の下で 1 本ずつしか走らない。ただし torch の演算は GIL を外すので、ワーカーが複数あれば並列になる。1:1 のとき（M 本のスレッドが GIL を取り合う）より取り合う数が N 本に減るので、GIL の convoy は減る
- デッドロックを避ける規則は今と同じ: sched と control の lock を持ったまま attach しない。ホストの待ちは detach する（`wait_interruptibly`）
- ワーカーの数の既定は両方のビルドで同じにする。GIL ありで Python の比率が高いノードは、`workers` を小さくしても速さがほとんど変わらないことを docs に一文書く

## 実装ステップ

PR を 3 つに分ける。各 PR で `just run` が通ること。

### PR 1: コアのワーカープール（振る舞いは変えない）

1. `next_target_ns` を切り出し、node.rs から使う。Rust の単体テストを足す
2. `RuntimeConfig::{workers, dedicated}` と `check_start` の検査を足す。`NodeHooks::enter` を足し、`Shared::invoke` の中で切り替えのときに呼ぶ
3. `runtime/pool.rs` に `Sched`・ワーカーのループ・ライフサイクルのタスクを書く。`Runtime::start` で、`dedicated[i]` ならスレッドを、そうでなければプールのノードとして積み、ワーカーを `min(workers, プールのノード数)` 本立てる
4. control.rs の `startup` / `capture` / `step` / `finish` と `Shared::{set_command, fail}` から、sched にタスクを積む経路をつなぐ
5. bindings はまだ `dedicated` を全部 true にして渡す（Python の振る舞いは変わらない）。crates/mujou-core/tests/runtime.rs のテストを、プールと専用スレッドの両方で回す
6. `specs/native-core.md` と `specs/execution-model.md` のコアの節を、実装に合わせて直す

### PR 2: Python のノードをワーカーに載せる（既定を切り替える）

1. bindings: `Runtime.start(workers=, dedicated=)`。`PyNode` に `context` と `dedicated` を持たせ、`enter` で `_on_enter` を呼ぶ。`start` は `dedicated` のときだけ `on_thread_start` を呼ぶ。`public()` の対応表を直す（`_on_enter` は次の hook の名前で出す）。`just stub`
2. `node.py`: `_on_thread_start` を `_on_enter` に置き換える（上の表）。`dedicated_thread` 引数と property、`checked_dedicated` を足す。pool のノードの `on_thread_start` の override を検査する関数を足す
3. `hooks.py`: `enter_node(node)` / `current_node()`。`binding.py:101-108` の判定を差し替える。`session.py:268` の `end_thread` を「ノードの束縛を外す」処理に置き換えるか削る（ノードは `current_node` にしか残らず、ワーカーが次のノードで上書きするので、正常終了後に古い束縛は効かない。終了した実行のワーカーはもう居ない）
4. `session.py` の `start`: `workers` と `dedicated` を native に渡す。`run.py` の `run` / `start_run` に `workers` を通す。`workers` の検査
5. docstring（`Node`、`mujou.run`、native の `Node` の各 hook の「on the node's own thread」）と docs（en / ja）、examples のコメントを直す
6. specs の残り、memory、skill `locality-and-async` を直す（下の一覧）

### PR 3: ベンチマークと既定値の確認

1. `benchmarks/mn_scheduler.py`（pytest では回さない。`just` のレシピは足さない）を書く。測る項目は下の「ベンチマーク」
2. 結果を `specs/native-core.md` の M:N の節に表で残す。既定のワーカー数（論理 CPU 数）が物理コア数より明らかに遅ければ、既定を変えてユーザーに報告する
3. PR 1 のレビューで PR 3 に回した 3 点を測る（2026-10-18、orchestrator の裁定）
   - timekeeper を起こしすぎる問題（algorithm-reviewer sf2・code-reviewer nit4）: 周期の違うノードを混ぜたとき（周期 0 と 10 ms など）、先頭の期限が早まるたびの `Timer::interrupt_waiters` が、この実行の timekeeper と同じタイマーで待つ全スレッド（ほかの実行・専用スレッド・ホストの `sleep`）を起こす回数と、非自発的なコンテキストスイッチ。直す案は「置いた期限が今の時刻以下なら割り込まない」「`wake_pool` の割り込みを最後に 1 回にまとめる」
   - 制御の lock の待ち（algorithm-reviewer nit3）: プールの更新は 1 回につき制御の lock を 2 回取り、その下でタイマーの Mutex も取る。M が大きいときの lock の待ちを測る

## テスト観点

### Rust（`crates/mujou-core/tests/runtime.rs`、プール用は `mod pool`）

- 正常系
  - `next_target_ns`: 遅れが周期未満なら前の目標 + P、P 以上なら now + P、ちょうど P なら now + P（今の 3 つの周期のテストの値をそのまま単体テストにする）
  - 今の周期のテスト（node.rs 由来の 7 本）・手動ステップ・capture・例外・復元のテストを、全ノード dedicated と全ノード pool の 2 通りで回す（ヘルパーで `RuntimeConfig` を切り替える）
  - EDF の順序: workers=1、仮想タイマー、周期 1/2/3 ms の 3 ノード。時刻を 1 ms ずつ進め、更新の順を記録する。オラクルは、同じ規則（期限の昇順、同じなら添字の昇順、`next_target_ns`）で手で書いた期待の列
  - 過負荷の公平さ: workers=1、周期 None の 4 ノード、時刻を一度に大きく進める。どのノードも 2 回目の前に全ノードが 1 回ずつ更新される（期限 last+1ns の順）
  - ワーカーの本数: M=64、workers=2 で hook を走らせた `thread::current().id()` の種類が 2 以下
  - `enter` の呼ばれ方: workers=1、手動で a, b, b, a の順に step する。`enter` は a, b, a の 3 回（b の 2 回目では呼ばれない）
  - dedicated と pool の混在: 両方に on_start → update → pause → resume → stop が 1 回ずつ届く
- 異常系
  - `workers` を使い切るデッドロックが無いこと: workers=2、8 ノード、1 つが `update` で失敗する。全ノードが `exception` → `stop` を 1 回ずつ（既存の `assert_exception_then_stop_once`）、`join` が有限時間で戻る
  - pool のノードの `start` が失敗したら、どのノードも `update` しない
  - 正常な停止で抜けた後に例外が来ても、`exception` → `stop` の順
  - `dedicated` の長さの不一致、`workers` が 0 → `InvalidArgument`（Rust 側は `NonZeroUsize` なので 0 は bindings で検査する）
  - ワーカーの中から `Runtime::shutdown` → `Reentrant`
- エッジケース
  - pool のノードが 0 個（全部 dedicated）ならワーカーを立てない（スレッド名 `mujou-worker-*` が無い）
  - workers > ノード数でも、立てるのはノード数ぶんだけ
  - 一時停止中に Stop が来たら `on_resume` を呼ばない
- 排他（確率的な smoke。決定的なテストが通った後の層、`slow` にはしない短いもの）: workers=4、周期 None の 16 ノードを実時間で 200 ms。各ノードの hook の入口で `AtomicBool::swap(true)` が false であることを確かめる。hook ごとの通し番号が単調に増える

### Python（`tests/mujou/_core/`）

- `test_native_runtime.py`
  - contextvars がノードごとに保たれる: workers=1、2 ノード。a が on_start で設定した ContextVar を b は見ず、a の on_update・on_pause・on_resume・on_stop は見る
  - 今の `test_hooks_share_one_python_thread_state` は `dedicated=[True]` に限る（threading.local と current_thread を保つのは専用スレッドだけ）
  - hook を走らせたスレッドの数: 32 ノード、workers=2 で、on_update の `threading.get_ident()` の種類が 2 以下
  - `_on_enter` の例外が次の公開の hook の名前（最初は `on_start`、手動の 2 回目は `on_update`）で出る
- `test_node.py`
  - 勾配の記録: workers=1 で、a の on_update が `torch.set_grad_enabled(True)` を `with` なしで呼んでも、b の on_update では `torch.is_grad_enabled()` が偽
  - CUDA（使えなければ skip）: workers=1 で CUDA の 2 ノード。各 hook で `torch.cuda.current_stream()` が自分の `_stream`、`current_device()` が自分のデバイス
  - `dedicated_thread`: 既定は False。`dedicated_thread=1` は `TypeError`。クラス属性で置き換えたら `mujou.run` で `TypeError`
  - `on_thread_start`: dedicated のノードでは自分のスレッドで on_start の直前に 1 回。pool のノードのクラスが override していたら `mujou.run` が `TypeError`、何も起動せず、同じノードを直して再び run できる
- `test_port.py` / `test_network.py`
  - workers=1 で、a の hook から b の Port の `recv` / `send` は `RuntimeError`（同じワーカーでも所有でない）
  - ホストから `recv` は今までどおり `RuntimeError`
- `test_run.py`
  - `workers` の検査: `True` は `TypeError`、0 と -1 は `ValueError`
  - pool のノードの hook から `mujou.run` は `RuntimeError`（`in_hook`）
- 数値的オラクル: 手動ステップの $\Delta t$ が、仮想タイマーで進めた値とちょうど一致すること（今のテストを pool で回す）。PC-ALM のテスト（BP 一致・α=0 の退化）が pool の既定のまま通ること（既存のまま）
- 全体: GIL ありの 3.14 でも全テストが通る（CI）。`tests/test_examples.py` が通る

## ベンチマーク（PR 3、`benchmarks/mn_scheduler.py`）

- 条件: ノードは CPU の torch の小さな層（n=32 の `addmm_`、`torch.set_num_threads(1)`）。M ∈ {8, 64, 256, 1024, 4096}。周期は 0.0 と 0.01 の 2 通り。比べるのは 1:1（全ノード `dedicated_thread=True`）と M:N（既定、workers = 論理コア数 と 物理コア数）。3.14t と 3.14 の両方で
- 測るもの
  - プロセスのスレッド数: Linux は `/proc/self/status` の `Threads`、macOS は `ps -M -p <pid>` の行数
  - コンテキストスイッチ: `resource.getrusage(RUSAGE_SELF)` の `ru_nvcsw` / `ru_nivcsw` の 10 秒あたりの増分
  - 更新頻度: ノードごとの更新回数/秒の合計・中央値・最小。周期 0.01 では、目標 100 Hz に対する達成率と $\Delta t$ の 99 パーセンタイル
  - 切り替えの費用: workers=1 で、1 ノードを回したときと 2 ノードを交互に回したときの 1 更新あたりの時間の差（CPU と、あれば CUDA）
- 期待: M ≥ 256 で M:N のスレッド数は N + 2 程度で一定、非自発的なスイッチが 1:1 より桁で少なく、合計の更新頻度が 1:1 以上。M が小さい（≤ 論理コア数）ときは 1:1 と同程度

## 想定リスク・トレードオフ

- ブロッキングする hook がワーカーを占有する。対策は `dedicated_thread=True` と docs の警告だけ（ワーカーの自動追加はしない。tokio の `spawn_blocking` に当たる仕組みは要求に無い）
- `threading.local` に頼る利用者のコードやライブラリが、pool では黙って壊れる。docs に書き、`on_thread_start` の override は `TypeError` で止める
- 全ワーカーが 1 つの sched の mutex を取り合う。Python のノード（µs〜ms）では問題にならないが、将来の小さなネイティブのノードでは、ワーカーごとのキューと work stealing が要る。今は作らない
- `timer.interrupt_waiters()` は、そのタイマーで待つものをすべて起こす（専用スレッドのノード、ホストの `sleep`）。先頭の期限が早まったときだけ呼ぶので頻度は低いが、ベンチマークで確かめる
- タイマーの時刻の読み取りがグローバルな Mutex を取る点（open decision）は、読むスレッドが減るぶん軽くなる。直しはしない
- 同時に走る実行ごとにプールができるので、実行が多いとスレッドが増える（Q2）
- 既定のワーカー数を論理 CPU 数にした。memory には「torch のスレッドを論理コア数まで増やすと遅くなった」とある。物理コア数を取るには crate（`num_cpus`）が要り、コアに初めて依存を足すことになるので、PR 3 の測定を見てから決める
- ノードを特定のワーカーに固定する案（pinning）は採らなかった。`threading.local` を一部保てるが、同じワーカーの別のノードと共有してしまい、保証としては中途半端になる。負荷の偏りも直せない（Q3）
- プロセス全体で 1 つのプールにする案は、実行ごとの一時停止・型 `N` の違い・終了の扱いが複雑になるので採らなかった（Q2）

## specs / docs / memory で直す箇所

- specs
  - `specs/native-core.md:28-45`: M:N の節を書き直す（Python のノードも既定でワーカーに載せる、専用スレッドは opt-in、GIL を理由に載せないという記述を消す、EDF・timekeeper・排他・設定の付け替え・`workers`）。経緯（2026-10-18 のユーザーの要求）を 1 段落
  - `specs/nodes.md:42-44`（配置の API: `workers` と `dedicated_thread` は決定、複数ノードを決まった順に同じワーカーで回す指定は未決定のまま）、`:56`、`:58`（Python のノードはワーカーか専用スレッドから hook を呼ばれる）、`:62-63`・`:69`・`:72`（`_on_thread_start` → `_on_enter`、`on_thread_start` は専用スレッドだけ）、`:78`
  - `specs/execution-model.md`: 停止の手順 3（pool のノードは、ワーカーがノードごとに `on_pause` を呼んで止める）、再開の段落（「スレッドが起きる遅れ」→「ワーカーの割り当ての待ち」）、終了の手順 4、例外の手順 1・5、実装の節（「ノードはそれぞれ専用のスレッドで動く」、起動の順の `_on_thread_start`、周期の段落に pool のヒープ）
  - `specs/torch.md:70`、`:72-90`（「実行スレッドの設定」→ 切り替えごとの設定。表の「ノードのスレッド」→「ノードの hook を走らせるスレッド」）、`:88`（所有の判定）、`:98-111`（内部スレッド数の目安）、MPS の経緯の「ノードごとのスレッドで計算する」
  - `specs/network.md:289`（Port を呼べるのは所有ノードの hook を走らせているスレッド）、`:308`
  - `specs/python-environment.md:9-11`（free-threaded を使う理由をワーカーの並列に）
  - `specs/open-decisions.md:15-17`（実行の配置: 決まった部分を消す）、`:41-43`、`:63`
  - `specs/update-rules.md:27`（`torch.compile` を多数スレッドから、は「ワーカーから」に）
- docs（en と ja の両方）
  - `index.md:29,43`、`concepts.md:31`、`getting-started/installation.md:16`、`getting-started/quickstart.md:36`
  - `user-guide/running.md:12` と、`workers` の節を追加
  - `user-guide/writing-a-node.md:130,231` と、`dedicated_thread` の説明（いつ使うか）
  - `user-guide/torch.md:43`（generator は別 PR で直る）、`:49`、`:67-68`、`:73`、`:95`、`:97-109`
  - `user-guide/state.md:100`、`user-guide/time.md:103`
  - examples のコメント: `examples/online_learning.py:94`、`examples/torch_node.py:3,64-68`、`examples/quickstart.py:66`
- docstring とスタブ: `src/mujou/_core/node.py` の `Node` の docstring（hook のスレッド、`dedicated_thread`）、`src/mujou/_core/run.py` の `run`、`crates/mujou-python/src/runtime.rs` の `Node` と `Runtime` の doc comment、`crates/mujou-core/src/runtime/mod.rs:1-24,48-85` のモジュールと trait の doc。`just stub` で `_native.pyi`
- memory
  - 新規 `memory/project_mn_scheduler.md`（2026-10-18 の要求と決定、確認事項の結論）と `MEMORY.md` に 1 行
  - `memory/project_native_core_rust.md`（M:N の項、「Python のノードはプールに載せない」を改める）
  - `memory/project_open_decisions.md`（実行の配置）
  - `memory/project_torch_module.md`（`_on_thread_start` → `_on_enter`、スレッド数の目安）
  - `memory/project_node_lifecycle.md:12-13,20`（`on_thread_start` は専用スレッドだけ、「Python（専用スレッド）」）
  - `memory/project_python_free_threaded.md`（Why）
  - `memory/agents/orchestrator/core-runtime.md` の Q-c に「2026-10-18 に実装」と追記
- skill: `.claude/skills/locality-and-async/SKILL.md` の「実行クラス（M:N スケジューリング）」の表と箇条（skill `edit-dot-claude` の手順で編集する）

## 確認事項

- Q1. 専用スレッドを残すか、どう指定するか、`on_thread_start` をどうするか
  - 暫定案: `Node(dedicated_thread=False)` を既定にして、専用スレッドは opt-in で残す。`on_thread_start` は専用スレッドのノードにだけ呼び、pool のノードのクラスが override していたら `mujou.run` で `TypeError`
  - 理由: ブロッキングする I/O のノードを pool に載せると、ほかのノードの $\Delta t$ が伸びる。逃げ道が要る。pool のノードでは「スレッドごとの設定」が hook の間で残らないので、`on_thread_start` を黙って呼ぶと罠になる
  - 代案: (a) 専用スレッドを無くして全部 pool にする（単純だが I/O のノードが困る）、(b) `on_thread_start` を廃止して、専用スレッドのノードは `on_start` に書く（hook が 1 つ減る。2026-10-17 に分けた判断を戻すことになる）
- Q2. ワーカープールの範囲と `workers` の指定場所
  - 暫定案: プールは実行ごとに作る。`mujou.run(..., workers=None)` で指定し、既定は `min(pool のノード数, 論理 CPU 数)`
  - 理由: 今のランタイムは実行ごとに閉じている（一時停止・終了・型 `N`）。同時に走る実行が少ないうちは、これで足りる
  - 代案: プロセスで 1 つのプールを共有する（同時に多数の実行を走らせてもスレッドが増えない。ただし実行をまたぐスケジューリング、一時停止の分離、プロセス全体の設定 API が要り、PR が大きくなる）
- Q3. ノードがワーカーの間を移ることと、hook の間で保つもの
  - 暫定案: どのワーカーがどのノードを走らせてもよい（移る）。hook の間で保つのは `contextvars`（ノードごとの `Context`）と、フレームワークが戻す勾配モードと CUDA の device と stream だけ。`threading.local` と `threading.current_thread()` は保たない（今のテスト `test_hooks_share_one_python_thread_state` は専用スレッドに限る）
  - 理由: 期限の順に空いたワーカーへ割り当てられ、負荷が偏らない。ワーカーに固定しても、同じワーカーの別のノードと `threading.local` を共有するので、保証にならない
  - 代案: 起動時にノードをワーカーに固定する（キャッシュの局所性は上がるが、負荷の偏りを直せない）

## 参照

- 現状のコード: `crates/mujou-core/src/runtime/mod.rs:56-101,184-250,366-440`、`node.rs:61-235`、`control.rs:50-178`、`crates/mujou-python/src/runtime.rs:278-346,461-534`、`src/mujou/_core/node.py:345-358,501-510`、`hooks.py`、`binding.py:101-108`、`session.py:136-190,268`、`run.py:396-500`
- 既存のテスト: `crates/mujou-core/tests/runtime.rs`（周期 702-830、例外 306-375）、`tests/mujou/_core/test_native_runtime.py:192-252`
- 仕様: `specs/native-core.md#mn-スケジューリング`、`specs/execution-model.md#停止と状態の取得復元`、`specs/nodes.md#実行の周期`、`specs/torch.md#実行スレッドの設定`
- 経緯: `memory/agents/orchestrator/core-runtime.md` の Q-c、`memory/project_native_core_rust.md`、`memory/project_open_decisions.md`
- skill: `locality-and-async`、`testing-strategy`、`refactor-conventions`

## ユーザーの決定（2026-10-18、orchestrator が中継）

- Q1: 専用スレッドは opt-in で残す。`Node(dedicated_thread=False)` が既定で、全ノードがワーカーに載る。`on_thread_start` は専用スレッドのノードだけに呼び、ワーカーに載るノードが override していたら `mujou.run` が何も起動しないうちに `TypeError`（推奨案どおり）
- Q2: プールは実行ごと、`mujou.run(workers=None)` で指定（推奨案どおり）
- Q3: ノードはワーカーの間を移ってよい。hook の間で保つのは contextvars と、フレームワークが戻す勾配モード・CUDA の device と stream だけ（推奨案どおり）
- PR は計画どおり 3 つ: ① コアのワーカープール（振る舞い不変）② Python のノードをワーカーに載せ既定を切り替え（docs・specs・memory・skill も）③ ベンチマーク
