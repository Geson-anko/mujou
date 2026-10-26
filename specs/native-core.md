# ネイティブコア

## コアの役割

mujou のコアには 2 つの責務がある。1 つは各ノードを非同期・分散的に回すこと、もう 1 つは非同期・分散的に学習するためのフレームワーク自体（PC-ALM）を提供すること。ノードの内側は自由にし、ノードの間のやり取りに不変条件を課す、という分担にしている。mujou 固有の価値は、ノード同士を非同期につなぎ、その上で学習させる部分にある。

ここでいうコアは mujou の中核の機能のことで、Rust の拡張モジュールのことではない。ノードを回す基盤は Rust で書き、PC-ALM は PyTorch で書いて `mujou.pc` から公開する（[更新手法](update-rules.md)）。

ノードを回す基盤の設計と実装は、次の順に優先する。

1. プロトコル: 何を、どの形式で、いつ読み書きするか
2. 実行フレーム: スケジューラ・通信・状態の所有

ノードの内側の実装（PC-ALM のノードや BP のノードなど）は、プロトコルの上に載る差し替え可能な部品として扱う。

## 実装言語

コアは Rust で実装し、Python はインターフェースにする。Python からは PyO3 + maturin で作る private な拡張モジュール `mujou._native` として使う。Rust を選んだのは、並行処理の安全性をコンパイラが保証し、PyO3 で Python から使えるからである。コアは torch を import しない。フレームワークの基礎を自分たちで制御するため（[構成](architecture.md#rust-と-python-の境界)）。

## ABI

`_native` の名前は 15 個で、それぞれ 1 つのことをする。

| 名前 | 役割 |
| --- | --- |
| `Timer`、`VirtualTimer`、`ReadOnlyTimer`、`Timestamp`、`global_timer` | 時間（`mujou.time` がそのまま再エクスポートする。pyclass の module は `"mujou.time"`） |
| `Schema`、`Outlet`、`Slot`、`InputSlot`、`InputGroup`、`Message`、`BufferView` | スロットとメッセージ |
| `start`、`Runtime` | 実行 |
| `current` | いま hook を走らせている Driver |

- `start(nodes, clock, *, manual, workers) -> Runtime` の 1 回の呼び出しで起動する。「作ったが始まっていない」状態が無い。`nodes` はレコード `_NodeEntry(driver, name, period, dedicated_thread, last_update_ns, state)`（`execution.py`）の列で、並べたリストの長さを照合する引数を表せなくした
- `Runtime` は `step(order)`、`capture(build, timeout)`、`shutdown(timeout) -> bool`（止める要求を出して待つ。timeout の変換は要求の前に行い、断られた呼び出しは何も変えない）、`join(timeout) -> bool`、`finished`、`check()`（失敗なら `NetworkError`）、`last_update_ns()`、`workers`（実効のワーカー数）を持つ。終了の状態は Rust だけが持ち、Python は聞く
- 正常に終わった後の取得も `Runtime.capture` の同じ口で行う
- bindings は `Runtime` の hook を、`Hook` から Driver のメソッド名への対応表で呼ぶ。Python を呼ぶ直前に thread-local の `CURRENT` に Driver を付け、戻ったら外す（`current.rs`）。core の `RUNTIME_OF_THREAD`（このスレッドがどのランタイムのものか）は、内部の `Reentrant` の検査のためにあり、「hook の中か」とは別の事実
- 例外は Python の `mujou.errors.NetworkError` を `import_exception!` で参照する。停止の timeout は組み込みの `TimeoutError`
- 内部の契約（値の個数、DLPack で読めない値、無いチャネルの番号など）の検査は FFI の健全性のためだけに残す内部エラーで、利用者に届いたら mujou のバグ（[提供シナリオとガード](scenarios.md#ガードではない意味論)）
- コアの `Slot` は 1 つの型で、書き手を `Source`（`Outlet` か `InputGroup`）で区別する。bindings は書き手の違いを型で見せるため `Slot` と `InputSlot` に分ける

## M:N スケジューリング

M 個のノードを N 本のワーカースレッドに載せる。各ノードは次のどちらかの実行クラスを持つ。

| 実行クラス | 実行のされ方 | 向いているノード |
| --- | --- | --- |
| ワーカー（既定） | 実行ごとのワーカープールで協調的に実行する | Python/PyTorch のノードを含むほとんどのノード |
| 専用スレッド（opt-in） | 自分の OS スレッドを持つ（`Node(dedicated_thread=True)`） | hook がブロッキングするノード（I/O など）、スレッドごとの状態に頼るノード |

### M:N にした理由

asyncio や Redis が 1 スレッドで大量のタスクを捌けるのは、タスクの多くが I/O 待ちだからである。mujou のノードは CPU を使い続けるので、1 スレッドでは 1 コアしか使えない。かといってノードごとに OS スレッドを立てると、スレッド数の上限とコンテキストスイッチのコストに縛られ、ノードの間の不公平も大きい。M:N なら、ノードの数によらずスレッドは N 本で済み、ブロッキングするノードだけを専用スレッドに逃がせる。GIL ありのビルドでも、取り合うスレッドが N 本に減る。プロセスで共有するプールやノードの pinning は、実行ごとの停止・終了の分離が複雑になり、負荷の偏りを直せないので採らなかった（比較の数値は [性能の基準値](research/benchmarks.md#mn-スケジューラの比較)）。

### ノードの形

ノードは「状態 + ライフサイクル hook」で、ほかのノードを待たないので async/await のコルーチンにする必要は無い。`on_update` の 1 回が協調スケジューリングの区切りになる。ランタイムが知っているのは `NodeHooks`（hook の呼び出し）だけで、ノードが何を計算し何をやり取りするかは知らない。

### ワーカーの実装

ノードはレーンに分かれる。プール（M 個のノードが N 本のワーカーを共有する）と、専用スレッドのノードごとのレーン（自分専属のワーカー 1 本）。どのワーカーも同じループを回す。

- ワーカーは制御の lock の下で 1 つのノードの仕事を 1 つ取り、lock の外で hook を 1 つ呼び、ノードを戻す。次に何をするかは状態機械（`lifecycle.rs`）が決める（[実行モデル](execution-model.md#ライフサイクルの遷移)）
- ライフサイクルの仕事（起動・一時停止・再開・`on_exception`・`on_stop`）は、積まれた順に更新より先に取る。ただし hook が失敗したノードの `on_exception` は先頭に積む。自由実行の更新は期限の早い順（同じなら添字の小さい順）。期限は [ノード](nodes.md#実行の周期) の規則（`next_target_ns`）で求める
- ノードを取るのは lock の下でキューから取るときだけで、戻すのは取ったワーカーだけ。そのため 1 つのノードを 2 つのスレッドで同時に回すことはない。連続する 2 つの hook の間で lock を通るので、前の hook の書き込みは次の hook から見える
- ワーカーは 1 つのノードを待たない（ワーカーがノードより少ないとデッドロックする）。仕事を待ち、レーンに 1 本だけの timekeeper が先頭の期限まで Timer の上で待つ
- timekeeper を起こすと、同じタイマーで待つすべてのスレッド（ほかの実行の timekeeper、専用スレッド、ホストの `sleep`）が起きる。そこで起こす場面を絞る。ノードを戻したワーカーは自分ですぐキューを見るので、戻したときにはほかのワーカーを起こさない。起こすのは、仕事を取った後に残りがあるときと、まだ来ない先頭の期限が timekeeper の期限より早いときだけ
- 更新を始めるかは、制御の lock の下で制御語と時刻を読んで決める。停止は同じ lock の下でフラグを立ててタイマーを凍結するので、凍結より後に始まる `on_update` は無い
- ノードが 1 つのレーン（専用スレッドと、ノードが 1 つのプール）は、自由実行の間はキューが要らないので、制御語が変わるまで lock を取らずに更新する

### ワーカー数の既定

プールのワーカー数は `mujou.run(..., workers=None)` で選ぶ。

- 既定は bindings だけが決め、コアの `RunConfig.workers` は常に数（`NonZeroUsize`）を受け取る。free-threaded では `available_parallelism`（cgroup の CPU quota と affinity を数える）、GIL が有効なとき（`sys._is_gil_enabled()`）はその半分（少なくとも 1）。Python の `os.process_cpu_count()` は quota を数えないので、コンテナの中で quota を超える
- 実効の数は `min(要求か既定, プールに載るノードの数)` で、プールに載るノードが無ければ 0 本。全ノードが専用スレッドの実行も正しい使い方なので、0 を許す。明示した値が 1 未満なら K20
- GIL ありの既定を半分にしたのは、GIL の取り合いで論理コア数のワーカーが約半分の速さになったためである。物理コア数は使わない。SMT の無い機械でしか測れておらず、free-threaded では論理コア数が最速だった（[性能の基準値](research/benchmarks.md#mn-スケジューラの比較)）

### Python のノードの載せ方

- hook はすべて Driver のメソッドを通り、Driver がノードごとの `contextvars.Context` の中で hook を呼ぶ（[ノード](nodes.md#driver)）
- ノードはワーカーの間を移ってよく、ワーカーに固定しない。hook の間で保つのは `contextvars` と、Driver が設定する CUDA の device と stream だけ
- GIL ありのビルドでは、ワーカーは寿命の間 Python に attach したまま、待つ間は detach する（`NodeHooks::on_thread`）。GIL を取るのは hook を呼ぶ間だけ。Python のコードは 1 本ずつしか走らないが、torch の演算は GIL を外すので、ワーカーが複数あれば並列に進む
- ホストの待ち（`join`、`capture`、`sleep` など）は Python から detach し、シグナル（Ctrl-C）を一定の間隔で確かめる。ノードと制御のスレッドが hook を呼ぶために attach するので、GIL ありのビルドでホストが GIL を持ったまま待つとデッドロックする

## スロットの実装

スロットの読み書きでワーカーを止めない。書き手はバッファプール（[メッセージ](messages.md#メモリ管理-バッファプール)）から取ったバッファを lock の外で埋め、スロットごとの mutex の下では `Arc` の複製と差し替えだけを行う。読み手も書き手も、相手の更新の進み具合は待たない。待つのはこの短い区間だけで、lock を持つスレッドが OS に横取りされると、その間は待つ。

lock-free の `arc-swap` は使わない。arc-swap の読み手は、読んだポインタを「借り」として、プロセスに 1 つの一覧に記録する。書き手は値を外すと、外したポインタと**値が同じ**借りを、どの `ArcSwap` のものかを区別せずに払う。読み手がスロット A からポインタ P を読んだあと、借りを記録する前に、P が解放されて同じアドレスにスロット B の値が置かれることがある。そのあと B の書き手がその値を外すと、読み手の借りを払ってしまい、読み手は B の値を A の値として返す。型が違えば型の取り違えになり、SIGSEGV を起こした。`readers_never_see_a_message_of_another_slot`（`crates/mujou-core/src/slot.rs`）はこれを確率的に捉える smoke test。

メッセージのフィールドは、プールのバッファ（`Value::Bytes`）か、書き手がプールの外に持つ不透明な値（`Value::Held`、`Arc<dyn Any + Send + Sync>`）のどちらか。書き手は `Fill` に渡されるフィールドごとの `FieldOut` に、`bytes()`（初回にプールからバッファを取る）か `hold(value)` で値を置く。保持するフィールドはホストのプールからバッファを取らない。不透明な値の中身は bindings が入れる `Py<PyAny>`（codec が書き手の device の上で作った、detach したコピーの `torch.Tensor`）で、コアは読まず検査もしない。device・stream・event もコアは知らない。shape と dtype の照合は `dlpack::import::check`（`copy_from` から切り出したもの）を bindings が呼ぶ（[メッセージ](messages.md#コアの中の値)）。

却下した案は次のとおり。

- コアのスロットに device ポインタ・stream・event を持たせる: コアに CUDA の意味論が入り、torch を import しない原則も崩れる
- Python の側に別の表を持ち、スロットには鍵だけを置く: 版番号の一斉切り替え・`wait_newer`・寿命の管理を、コアと二重に持つことになる
- `Slot<V>` の総称化: 変更が大きく、得るものが無い

複数のマシンにまたがる実行は想定しない。

## 性能の予算

基準値と測り方は [性能の基準値](research/benchmarks.md) にある。設計の判断に使う予算は次のとおり。

- ポートの `recv` / `send` の 1 回は数 µs（`from_dlpack` とコピーが大半）。境界を細くする変更（`current()` の FFI、Driver）は、悪化が 5% 未満であることを確かめてから入れる
- 1 回の update で FFI をまたぐのは 1 回だけ（Driver の `update`）
- ノードの内側の torch の状態は、その場で更新する。新しいテンソルを確保する書き方は、多数のスレッドでメモリの確保が律速になる
- torch のノードを動かすスレッドの数（ワーカーと専用スレッド）× torch の内部スレッド数は、物理コア数を超えないようにする

## ツールチェーン

- 開発は free-threaded 版の Python 3.14t で行う（`.python-version` は `3.14t`、`requires-python` は `>=3.14`）。GIL ありの通常ビルドもサポートし、CI のテストは 3.14t と 3.14 の両方で回す。free-threaded を使うのは、Python のノードをワーカーと専用スレッドで並列に動かしたいからである。3.13t は実験的な扱いで 1 スレッドの性能低下も大きく、3.14 から公式サポートになった（PEP 779）。free-threaded のビルドで守る規則は [構成](architecture.md#free-threaded-の規則)
- pre-commit のフックも Python 3.14 で動かす（ソースを parse するので、型パラメータの既定値などの構文を読める版が要る）。CI の pre-commit ジョブは、`python3.14` という名前の実行ファイルを確実に用意するために GIL ありの 3.14 を使う
- リポジトリのルートを Cargo workspace にし、crate を 3 つ置く

| crate | 中身 |
| --- | --- |
| `crates/mujou-core` | コアの本体。PyO3 に依存しないので、`cargo test` が libpython をリンクせずに回る |
| `crates/mujou-python` | PyO3 の bindings。拡張モジュール `mujou._native` になる。free-threaded 対応を明示する（`gil_used = false`）。`test = false` で、bindings は pytest で確かめる |
| `crates/mujou-stubgen` | `_native` の型スタブの生成器（`just stub` が使う） |

- Python のパッケージのビルドバックエンドは maturin。`uv sync` が拡張モジュールをビルドして `src/mujou/` に置く（editable install）。`pyproject.toml` の `[tool.uv] cache-keys` に Rust のソースを入れてあるので、`uv run` は Rust の変更を検知して再ビルドする
- 版数は Cargo workspace の `version` だけで決める。maturin が Python パッケージの版にし、`mujou.__version__` は `importlib.metadata` から取る
- Rust の版は `rust-toolchain.toml` で固定する。最低限サポートする版は `Cargo.toml` の `rust-version` で、CI の MSRV ジョブがその版で `cargo check --workspace --all-targets` を通す。`cargo fmt` と `cargo clippy -D warnings` は pre-commit で、`cargo doc`（警告で失敗）は CI で回す
- 実行時の依存は `torch` だけ。examples が使う numpy は dependency group `examples` に置き、`default-groups` で `uv sync` が入れる

### 型スタブ

`src/mujou/_native.pyi` は `just stub` で生成し、生成物をコミットする。手で編集しない。

- pyo3 の `experimental-inspect` が拡張モジュールに埋め込むシグネチャと doc comment を、`crates/mujou-stubgen`（`pyo3-introspection`）が読んで書き出し、ruff で整形する。docstring の正本は Rust の doc comment で、前提条件・例外・スレッドの約束はそこに書く
- pre-commit のフックが毎回生成し直し、コミットしたスタブが古ければ失敗する。CI の pre-commit ジョブは拡張モジュールをビルドしないのでこのフックを飛ばし、代わりに Test のワークフローが `just stub && git diff --exit-code` で確かめる
- introspection は inline module（`#[pymodule] mod`）にしか効かない。モジュールの名前は `lib.rs` の `#[pymodule(module = "mujou", name = "_native")]` で決まり、`pyproject.toml` の `module-name` もそろえる
- 生成器は pyo3 の足りない所を補う。`__new__` の戻りを `typing.Self` にし、注釈の無い `*args` / `**kwargs` を `typing.Any` にし、`object` のメソッドの override に `@typing.override` を付ける
- `#[pyo3(signature = ...)]` の文字列の注釈は import を伴わないので、組み込みとモジュール内の名前だけを書く。`collections.abc.Callable` などが要る型は、小さな型で表す（`crates/mujou-python/src/hint.rs`）
- `pyo3-stub-gen` は pyo3 0.30 に対応していないので使わない
