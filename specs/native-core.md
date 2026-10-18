# ネイティブコア

## コアの役割

mujou のコアには 2 つの責務がある。1 つは、各ノードを非同期・分散的に回すことである。もう 1 つは、非同期・分散的に学習するためのフレームワーク自体を提供することである。ノード同士が非同期に値を交換し、推論と学習を同時に進める仕組みそのものが、フレームワークの本体になる。

ノードの内側は自由にし、ノードの間のやり取りには不変条件を課す、という分担にしている。mujou 固有の価値はノードの内側の計算ではなく、ノード同士を非同期につなぎ、その上で学習させる部分にある。

### ノードを非同期・分散的に回す

この責務の中身は、ノード間で値をやり取りするプロトコルと、その実行フレームである。設計と実装は次の順に優先する。

1. プロトコル: 何を、どの形式で、いつ読み書きするか
2. 実行フレーム: スケジューラ・通信・状態の所有

ノード内部の実装（PyTorch による PC-ALM ノードや BP ノードなど）は、プロトコルの上に載る差し替え可能な部品として扱う。ノードの内側では PyTorch などの既存ライブラリを使ってよく、テンソル演算や autograd（ノード内 BP を含む）をそれらに任せられる。

### 非同期・分散的に学習するためのフレームワーク

mujou は、ノードを回す基盤だけでなく、その上で非同期・分散的に学習するためのフレームワークも、コアの責務として提供する。学習フレームワークとして提供するのは PC-ALM である。ここでいうコアは mujou の中核の機能のことで、`mujou._core` パッケージ（とその中の Rust の拡張モジュール `_native`）のことではない。PC-ALM は PyTorch で実装し、`mujou.pc` から公開する（[更新手法](update-rules.md#実装と公開元)）。

PC-ALM を mujou が提供しても、ノード・スロット・スケジューラは特定の学習則を前提にしない。更新手法を差し替えられるという要件（[更新手法](update-rules.md)）はそのまま保ち、PC-ALM 固有の状態や値は PC-ALM の更新手法の側に閉じ込める。局所性・非同期性の不変条件も、実行の基盤と同じように守る。

## 実装言語

コアは Rust で実装し、Python はインターフェースにする。Python からは private な拡張モジュール `mujou._core._native` として使う。Python の実装も private なパッケージ `mujou._core` に集め、利用者が使う名前は `mujou.*` のトップレベルに公開する。`mujou._core._native` は torch に依存しない（[PyTorch の利用](torch.md#モジュールの構成)）。Python からは PyO3 + maturin で呼ぶ想定で、コアの処理中は GIL を解放する。Rust を選んだのは、並行処理の安全性をコンパイラが保証し、PyO3 で Python から使えるからである。

## M:N スケジューリング

M 個のノードを N 本のワーカースレッド（既定は論理 CPU 数まで）に載せる。各ノードは次のどちらかの実行クラスを持つ。

| 実行クラス   | 実行のされ方                                       | 向いているノード                                       |
| ------------ | -------------------------------------------------- | ------------------------------------------------------ |
| ワーカー     | 実行ごとのワーカープールで協調的に実行する（既定） | Python/PyTorch のノード、ネイティブのノードなど大半    |
| 専用スレッド | 自分の OS スレッドを持つ（opt-in）                 | hook がブロッキングするノード（I/O など）、スレッドごとの状態に頼るノード |

Python のノードも既定でワーカーに載り、`Node(dedicated_thread=True)` を渡したノードだけが専用スレッドで動く（[ノード](nodes.md#非同期の境界と実行の配置)）。以前は GIL を理由に Python のノードをプールへ載せず、全部専用スレッドで動かしていた。2026-10-18 にユーザーから「大量のスレッドを立てると OS のコンテキストスイッチを取り合うので M:N でなければならない」と要求があり、既定をワーカーに切り替えた。

### M:N にした理由

asyncio や Redis が 1 スレッドで大量のタスクを捌けるのは、タスクの多くが I/O 待ちだからである。mujou のノードは CPU を使い続けるので、1 スレッドでは 1 コアしか使えない。かといってノードごとに OS スレッドを立てると、スレッド数の上限とコンテキストスイッチのコストに縛られる。M:N なら、ノードの数によらずスレッドは N 本で済み、ブロッキングするノードだけを専用スレッドに逃がせる。

### ノードの形

ノードは「状態 + ライフサイクル hook」である（[ノード](nodes.md#ライフサイクル-hook)）。ノードは他のノードを待たないので（[実行モデル](execution-model.md)）、async/await のコルーチンにする必要はない。スロットを読み、$\Delta t$ ぶん更新し、書き込むという 1 ステップ（`on_update`）が、協調スケジューリングの区切りになる。ノードに目標周期を持たせる場合、スケジューラは期限の早い順に実行する。Rust のネイティブなノードは hook を trait として実装し、Python のノードはワーカーか専用スレッドから同じ hook を呼ばれる。

### ワーカープールの実装

コアの `Runtime` はノードごとに実行クラスを受け取る（`RuntimeConfig::dedicated`。省略するとすべてプール）。専用スレッドのノードは自分のスレッドでループし、プールのノードは実行ごとに立てるワーカーが回す。ワーカーの数は `RuntimeConfig::workers` で、既定は `std::thread::available_parallelism()`。プールのノードの数より多くは立てず、プールのノードが無ければ 1 本も立てない。プールは実行をまたいで共有しない。

- ノードはほかのノードの更新を待たない。待つのは空いたワーカーだけで、過負荷では各ノードの $\Delta t$ が伸びる。ブロッキングする hook はその間ワーカーを占有する
- プールのノードは専用スレッドのループと同じ位置（起動・起動の完了待ち・ループ・一時停止・終了処理）を状態として持ち、制御の lock の下に置く。ワーカーはこの lock の下で仕事を 1 つ取り、lock を解放して hook を 1 つ呼んだ後、ノードを戻す
- 専用スレッドなら待つ箇所（全ノードの起動、一時停止の終わり、手動ステップの許可、終了の許可）で、ワーカーはノードを保留にして次の仕事に移る。条件を変えた側（制御語の変更・例外・起動の完了・ステップの許可・終了の許可）がノードを評価し直す。ワーカーが 1 つのノードを待つと、ワーカーがノードより少ない場合にデッドロックする
- hook を呼ぶ仕事（起動・一時停止・再開・`on_exception`・`on_stop`、手動モードのステップの更新）は、積まれた順に更新より先に取る。ただし hook が失敗したノードの `on_exception` は先頭に積む。自由実行の更新は期限の早い順（同じなら添字の小さい順）。期限は専用スレッドと同じ関数 `next_target_ns` で求める
- タイマーで待つワーカー（timekeeper）は 1 本だけで、先頭の期限まで `Timer::wait_until_ns_or` で待つ。ほかのワーカーは仕事を待つ。より早い期限やほかの仕事が入ったとき、制御語が変わったときに timekeeper を起こす。仕事を取ったワーカーは、残りの仕事と timekeeper の役を別のワーカー 1 本に渡す
- ノードを取るのは lock の下でキューから取るときだけで、戻すのは取ったワーカーだけ。そのため 1 つのノードを 2 つのスレッドで同時に回すことはない。連続する 2 つの hook の間で lock を通るので、前の hook の書き込みは次の hook から見える
- 更新を始めるかは、制御の lock の下で制御語と時刻を読んで決める。停止は同じ lock の下でフラグを立ててタイマーを凍結するので、凍結より後に始まる `on_update` は無い（専用スレッドの区切りと同じ保証）
- スレッドが別のノードに切り替わるとき（そのスレッドで初めてそのノードを回すときを含む）、コアは次の hook の直前に同じ呼び出しの中で `NodeHooks::enter` を呼ぶ。専用スレッドでは `enter` が成功すれば `on_start` の前の 1 回だけ。制御スレッドの `state_dict` の前には呼ばない
- `enter` の失敗（エラーでも panic でも）は次の hook の失敗として扱い、コアが次の hook の名前で報告する。次の hook は呼ばず、次の hook の前にもう一度 `enter` を呼ぶ。ただし `on_exception` と `on_stop` は `enter` が失敗しても呼び、資源を解放させる（[例外による終了](execution-model.md#例外による終了)の手順 5）。専用スレッドで `on_start` の前の `enter` が失敗したときも、`on_start` の失敗と同じく `on_exception` → `on_stop` と進む。失敗のログはノードの側で出す

### Python のノードの載せ方

bindings はノードごとの `dedicated_thread` を `RuntimeConfig::dedicated` に、`mujou.run(..., workers=None)` の `workers` を `RuntimeConfig::workers` に渡す。`None` なら `min(ワーカーに載るノード数, available_parallelism)`。`workers` が `int` でなければ（`bool` も）`TypeError`、1 未満なら `ValueError`。`mujou.testing.run` に `workers` は無い。

- `on_thread_start` は専用スレッドのノードにだけ、そのスレッドで `on_start` の直前に 1 回呼ぶ。ワーカーではスレッドごとの設定が hook の間で残らないので、黙って呼ぶと罠になる。そのためワーカーに載るノードのクラスが `on_thread_start` を override していたら、`mujou.run` は何も起動しないうちに `TypeError` を投げる
- `NodeHooks::enter` は内部の hook `_on_enter` を呼ぶ。`mujou.Node` はここでまず今走らせているノードとして自分を記録し、CUDA なら `set_device` と、最初の `_on_enter` で作ったノード専用のストリームの `set_stream` を行う。どれもスレッドごとの設定なので、切り替わるたびに設定し直す。勾配の記録は、bindings が `state_dict` 以外の公開の hook の直前に毎回呼ぶ内部の hook `_before_hook` で切る（[PyTorch の利用](torch.md#切り替えごとの設定)）
- hook はすべてノードごとの `contextvars.Context()`（空で作る）の中で `Context.run` で呼ぶ。専用スレッドのノードも同じ。`state_dict`（`_capture_state`）は文脈の外で呼ぶ
- ノードはワーカーの間を移ってよく、ワーカーに固定しない。hook の間で保つのは `contextvars` と、フレームワークが戻す勾配モード（hook の前に毎回）・CUDA の device と stream（切り替えのたび）だけ。`threading.local`・`threading.current_thread()`・`threading.get_ident()` は hook ごとに変わりうる（専用スレッドでは保たれる）
- GIL ありのビルドでは、ワーカーは寿命の間 Python に attach したまま、待つ間は detach する（`NodeHooks::on_thread`）。GIL を取るのは hook を呼ぶ間だけ。Python のコードは 1 本ずつしか走らないが、torch の演算は GIL を外すのでワーカーが複数あれば並列に進む。GIL を取り合うスレッドはノードの数でなく N 本に減る。`workers` の既定は両方のビルドで同じ

### スロットの実装

スロットの読み書きでワーカーを止めないようにする。書き手はバッファプール（[メッセージ](messages.md#メモリ管理-バッファプール)）から取ったバッファを lock の外で埋め、スロットごとの mutex の下では `Arc` の複製と差し替えだけを行う。読み手も書き手も、相手の更新の進み具合は待たない。待つのは、この短い区間だけである。ただし lock を持つスレッドが OS に横取りされると、その間は待つ（2026-10-17 にユーザーが承認。競合が問題になったときの扱いは [未決定事項](open-decisions.md#グローバルタイマーとスケジューラの境界)）。

以前は `arc-swap`（1.9.2）で lock-free に差し替えていたが、別のスロットの値を読み手に渡す不具合があったのでやめた。arc-swap の読み手は、読んだポインタを「借り」として、プロセスに 1 つの一覧に記録する。書き手は値を外すと、外したポインタと**値が同じ**借りを、どの `ArcSwap` のものかを区別せずに参照数を増やして払う。読み手がスロット A からポインタ P を読んだあと、借りを記録する前に、P が解放されて同じアドレスにスロット B の値が置かれることがある。そのあと B の書き手がその値を外すと、読み手の借りを払ってしまう。読み手は A を読み直して P でないことを見ても、借りが払われているので P を自分の値として返す。これは B の値である。型が違えば（`InputSlot` の中身と `Message` など）型の取り違えになり、SIGSEGV を起こした。`readers_never_see_a_message_of_another_slot`（`crates/mujou-core/src/slot.rs`）はこれを確率的に捉える smoke test で、arc-swap に対しては約半分の実行で落ちた。

複数マシンにまたがる実行は、現状は想定しない。

## メッセージの値の型

コアは、スロットの値を不変なバッファとしてホストのメモリに持つ。このバッファは `mujou._core._native` の内部の型で、利用者には見せない。利用者がやり取りする値は `torch.Tensor` だけである。コアは特定のテンソルライブラリに依存せず、torch も import しない。フレームワークの基礎を自分たちで制御するためである。

`torch.Tensor` とコアの間は DLPack で受け渡し、読み書きとも必ずコピーする。コアは独自の dtype を持たず、DLPack の dtype の表現をそのまま使う。コアは GPU をサポートせず、GPU 上の値は受け取らない。値の型とメッセージの型、スロットのメモリ管理は [メッセージ](messages.md) で、ノードの内側で GPU を使う場合の方針は [PyTorch の利用](torch.md) で定める。

## ツールチェーン

リポジトリのルートを Cargo workspace にし、crate を本体と bindings の 2 つに分ける。ほかに、型スタブの生成器を別の crate として持つ。

| crate                  | 中身                                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `crates/mujou-core`    | コアの本体。PyO3 に依存しないので、`cargo test` が libpython をリンクせずに回る           |
| `crates/mujou-python`  | PyO3 の bindings。拡張モジュール `mujou._core._native` になる。free-threaded 対応を明示する（`gil_used = false`） |
| `crates/mujou-stubgen` | `mujou._core._native` の型スタブの生成器（`just stub` が使う）                                    |

- Python のパッケージのビルドバックエンドは maturin にする。`uv sync` が拡張モジュールをビルドし、`src/mujou/_core/` に置く（editable install）。`pyproject.toml` の `[tool.uv] cache-keys` に Rust のソースを入れてあるので、`uv run` は Rust の変更を検知して再ビルドする
- 版数は Cargo workspace の `version` だけで決める。maturin が Python パッケージの版にし、`mujou.__version__` は `mujou._core._native.__version__` を再 export する
- `just test` は `cargo test --workspace` と pytest を回す。`cargo fmt` と `cargo clippy -D warnings` は pre-commit で回す
- 型スタブ `src/mujou/_core/_native.pyi` は `just stub` で生成し、生成物をコミットする。pyo3 の `experimental-inspect` が拡張モジュールに埋め込むシグネチャと doc comment を、`crates/mujou-stubgen`（`pyo3-introspection`）が読んで書き出し、ruff で整形する。docstring は Rust の doc comment から出るので、説明はそこに書く。pre-commit の `mujou-stub` が毎回生成し直し、コミットしたスタブが古ければ「ファイルを変更した」として失敗する（`just run` の `format` でも走る）
  - `pyo3-stub-gen` は pyo3 0.30 に対応していない（0.23.1 時点で `<0.30`）ので使わない
  - introspection は inline module（`#[pymodule] mod`）にしか効かない。モジュールの名前は `lib.rs` の `#[pymodule(module = "mujou._core", name = "_native")]` で決まり、各クラスの `module = "mujou._core._native"` と `pyproject.toml` の `module-name` もそろえる（pyo3 はクラスの `module` に文字列リテラルしか取らない）。`just stub` は `module-name` から拡張モジュールとスタブの置き場所を決める
  - 生成器は pyo3 の足りない所を補う。`create_exception!` の例外（`NetworkError` / `PauseTimeoutError`）は introspection に出ないので、生成器が同じ docstring で足す。`__new__` の戻りを `typing.Self` にし、注釈の無い `*args` / `**kwargs` を `typing.Any` にし、`object` のメソッドの override に `@typing.override` を付ける。`tests/mujou/_core/test_native_stub.py` がスタブと実行時のモジュールの名前・docstring・基底クラスのずれを検出する
  - `PauseTimeoutError` をクラスにできないのは、pyo3 0.30 が `PyTimeoutError` を `BaseException` のレイアウトで宣言しており、継承するとクラスの初期化に失敗するからである
  - `#[pyo3(signature = ...)]` の文字列の注釈は import を伴わないので、組み込みとモジュール内の名前だけを書く。`collections.abc.Callable` などが要る型は、`IntoPyObject::OUTPUT_TYPE` / `FromPyObject::INPUT_TYPE` を持つ小さな型で表す（`crates/mujou-python/src/hint.rs`）
