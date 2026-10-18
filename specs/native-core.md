# ネイティブコア

## コアの役割

mujou のコアには 2 つの責務がある。1 つは、各ノードを非同期・分散的に回すことである。もう 1 つは、非同期・分散的に学習するためのフレームワーク自体を提供することである。ノード同士が非同期に値を交換し、推論と学習を同時に進める仕組みそのものが、フレームワークの本体になる。

ノードの内側は自由にし、ノードの間のやり取りには不変条件を課す、という分担にしている。mujou 固有の価値はノードの内側の計算ではなく、ノード同士を非同期につなぎ、その上で学習させる部分にある。

### ノードを非同期・分散的に回す

この責務の中身は、ノード間で値をやり取りするプロトコルと、その実行フレームである。設計と実装は次の順に優先する。

1. プロトコル: 何を、どの形式で、いつ読み書きするか
2. 実行フレーム: スケジューラ・通信・状態の所有

ノード内部の実装（PyTorch による PC-ALM ノードや BP ノードなど）は、プロトコルの上に載る差し替え可能な部品として扱う。ノードの内側では PyTorch などの既存ライブラリを使ってよく、テンソル演算や autograd（ノード内 BP を含む）をそれらに任せられる。

ホストからの制御（停止・手動のステップ・終了）は、実行ごとに 1 本の制御スレッドが 1 件ずつ処理する。例外を捕まえたノードも制御スレッドに知らせる。制御を 1 本のスレッドに集めるので、ホストの要求と例外による終了が入り混じらない。タイマーもコアの側にあり（`crates/mujou-core/src/time.rs`）、止めておきたい側がそれぞれ `PauseHold` という保持を持つ。保持が 1 つでも残っていればタイマーは止まったままなので、その実行・ホスト・別の実行の停止が互いを解いてしまうことはない。詳細は [実行モデル](execution-model.md#実装-1)。

### 非同期・分散的に学習するためのフレームワーク

mujou は、ノードを回す基盤だけでなく、その上で非同期・分散的に学習するためのフレームワークも、コアの責務として提供する。学習フレームワークとして提供するのは PC-ALM である（未実装）。ここでいうコアは mujou の中核の機能のことで、`mujou._core` パッケージ（とその中の Rust の拡張モジュール `_native`）のことではない。PC-ALM は PyTorch で実装し、`mujou.pc` から公開する予定である。`mujou.pc` もまだ無い（[更新手法](update-rules.md#実装と公開元)）。

PC-ALM を mujou が提供しても、ノード・スロット・スケジューラは特定の学習則を前提にしない。更新手法を差し替えられるという要件（[更新手法](update-rules.md)）はそのまま保ち、PC-ALM 固有の状態や値は PC-ALM の更新手法の側に閉じ込める。局所性・非同期性の不変条件も、実行の基盤と同じように守る。

## 実装言語

コアは Rust で実装し、Python はインターフェースにする。Python からは private な拡張モジュール `mujou._core._native` として使う。Python の実装も private なパッケージ `mujou._core` に集め、利用者が使う名前は `mujou.*` のトップレベルに公開する。`mujou._core._native` は torch に依存しない（[PyTorch の利用](torch.md#モジュールの構成)）。Python からは PyO3 + maturin で呼ぶ。スロットへの書き込みと待機は GIL を外して行う。`Slot.read` と `InputSlot.read` だけは GIL を持ったままスロットの lock を取り、`Arc` を複製する（短い区間）。Rust を選んだのは、並行処理の安全性をコンパイラが保証し、PyO3 で Python から使えるからである。

## M:N スケジューリング

M 個のノードを N 本のワーカースレッド（目安は CPU コア数）に載せる。各ノードは次のどちらかの実行クラスを持つ。

| 実行クラス   | 実行のされ方                         | 向いているノード                                           |
| ------------ | ------------------------------------ | ------------------------------------------------------------ |
| タスク       | 共有ワーカープールで協調的に実行する | ネイティブのニューロンノードなど、小さく大量の軽いノード |
| 専用スレッド | 自分の OS スレッドを持つ             | Python/PyTorch のノード、ノード内 BP、I/O、重いスタック  |

Python のコードを動かすノードは GIL を要する場合があるので、タスクのプールには載せない。

### M:N にした理由

asyncio や Redis が 1 スレッドで大量のタスクを捌けるのは、タスクの多くが I/O 待ちだからである。mujou のノードは CPU を使い続けるので、1 スレッドでは 1 コアしか使えない。かといってノードごとに OS スレッドを立てると、スレッド数の上限とコンテキストスイッチのコストに縛られる。M:N なら、大量の小さなノードと、重いノードや GIL が要るノードを、同じ仕組みで扱える。

### ノードの形

ノードは「状態 + ライフサイクル hook」である（[ノード](nodes.md#ライフサイクル-hook)）。ノードは他のノードを待たないので（[実行モデル](execution-model.md)）、async/await のコルーチンにする必要はない。スロットを読み、$\Delta t$ ぶん更新し、書き込むという 1 ステップ（`on_update`）が、協調スケジューリングの区切りになる。ノードに目標周期を持たせる場合、スケジューラは期限の早い順に実行する。Rust のネイティブなノードは hook を trait として実装し、Python のノードは専用スレッドから同じ hook を呼ばれる。

### スロットの実装

スロットの読み書きでワーカーを止めないようにする。書き手はバッファプール（[メッセージ](messages.md#メモリ管理-バッファプール)）から取ったバッファを lock の外で埋め、スロットごとの mutex の下では `Arc` の複製と差し替えだけを行う。読み手も書き手も、相手の更新の進み具合は待たない。待つのは、この短い区間だけである。ただし lock を持つスレッドが OS に横取りされると、その間は待つ（2026-10-17 にユーザーが承認。競合が問題になったときの扱いは [未決定事項](open-decisions.md#グローバルタイマーとスケジューラの境界)）。

以前は `arc-swap`（1.9.2）で lock-free に差し替えていたが、別のスロットの値を読み手に渡す不具合があったのでやめた。arc-swap の読み手は、読んだポインタを「借り」として、プロセスに 1 つの一覧に記録する。書き手は値を外すと、外したポインタと**値が同じ**借りを、どの `ArcSwap` のものかを区別せずに参照数を増やして払う。読み手がスロット A からポインタ P を読んだあと、借りを記録する前に、P が解放されて同じアドレスにスロット B の値が置かれることがある。そのあと B の書き手がその値を外すと、読み手の借りを払ってしまう。読み手は A を読み直して P でないことを見ても、借りが払われているので P を自分の値として返す。これは B の値である。型が違えば（`InputSlot` の中身と `Message` など）型の取り違えになり、SIGSEGV を起こした。`readers_never_see_a_message_of_another_slot`（`crates/mujou-core/src/slot.rs`）はこれを確率的に捉える smoke test で、arc-swap に対しては約半分の実行で落ちた。

複数マシンにまたがる実行は、現状は想定しない。

## メッセージの値の型

コアは、スロットの値を不変なバッファとしてホストのメモリに持つ。このバッファは `mujou._core._native` の内部の型で、利用者には見せない。利用者がやり取りする値は `torch.Tensor` だけである。コアは特定のテンソルライブラリに依存せず、torch も import しない。フレームワークの基礎を自分たちで制御するためである。

`torch.Tensor` とコアの間は DLPack で受け渡し、読み書きとも必ずコピーする。コアは独自の dtype を持たず、DLPack の dtype の表現をそのまま使う。コアは GPU をサポートせず、GPU 上の値は受け取らない。pinned memory（DLPack のデバイス種別 `kDLCUDAHost`）はホストから読めるので受け付ける。値の型とメッセージの型、スロットのメモリ管理は [メッセージ](messages.md) で、ノードの内側で GPU を使う場合の方針は [PyTorch の利用](torch.md) で定める。

## ツールチェーン

リポジトリのルートを Cargo workspace にし、crate を本体と bindings の 2 つに分ける。ほかに、型スタブの生成器を別の crate として持つ。

| crate                  | 中身                                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `crates/mujou-core`    | コアの本体。PyO3 に依存しないので、`cargo test` が libpython をリンクせずに回る           |
| `crates/mujou-python`  | PyO3 の bindings。拡張モジュール `mujou._core._native` になる。free-threaded 対応を明示する（`gil_used = false`） |
| `crates/mujou-stubgen` | `mujou._core._native` の型スタブの生成器（`just stub` が使う）                                    |

- Python のパッケージのビルドバックエンドは maturin にする。`uv sync` が拡張モジュールをビルドし、`src/mujou/_core/` に置く（editable install）。`pyproject.toml` の `[tool.uv] cache-keys` に Rust のソースを入れてあるので、`uv run` は Rust の変更を検知して再ビルドする
- 版数は Cargo workspace の `version` だけで決める。maturin が Python パッケージの版にし、`mujou.__version__` は `mujou._core._native.__version__` を再 export する
- Rust の版は `rust-toolchain.toml` で固定する。最低限サポートする版は `Cargo.toml` の `rust-version` で、CI の MSRV ジョブがその版で `cargo check --workspace --all-targets` を通す
- `just test` は `cargo test --workspace` と pytest を回す。`crates/mujou-python` は `test = false` で、テストバイナリは作らない。bindings は pytest で確かめる。`cargo fmt` と `cargo clippy -D warnings` は pre-commit で回す。CI は `RUSTDOCFLAGS="-D warnings"` で `cargo doc --workspace --no-deps` を回し、doc comment の壊れたリンクで失敗させる
- 型スタブ `src/mujou/_core/_native.pyi` は `just stub` で生成し、生成物をコミットする。pyo3 の `experimental-inspect` が拡張モジュールに埋め込むシグネチャと doc comment を、`crates/mujou-stubgen`（`pyo3-introspection`）が読んで書き出し、ruff で整形する。docstring は Rust の doc comment から出るので、説明はそこに書く。pre-commit の `mujou-stub` が毎回生成し直し、コミットしたスタブが古ければ「ファイルを変更した」として失敗する（`just run` の `format` でも走る）。CI の pre-commit ジョブは拡張モジュールをビルドしないので `mujou-stub` を SKIP し、代わりに test ワークフローが `just stub && git diff --exit-code` でスタブが最新かを確かめる
  - `pyo3-stub-gen` は pyo3 0.30 に対応していない（0.23.1 時点で `<0.30`）ので使わない
  - introspection は inline module（`#[pymodule] mod`）にしか効かない。モジュールの名前は `lib.rs` の `#[pymodule(module = "mujou._core", name = "_native")]` で決まり、各クラスの `module = "mujou._core._native"` と `pyproject.toml` の `module-name` もそろえる（pyo3 はクラスの `module` に文字列リテラルしか取らない）。`just stub` は `module-name` から拡張モジュールとスタブの置き場所を決める
  - 生成器は pyo3 の足りない所を補う。`create_exception!` の例外（`NetworkError` / `PauseTimeoutError`）は introspection に出ないので、生成器が同じ docstring で足す。`__new__` の戻りを `typing.Self` にし、注釈の無い `*args` / `**kwargs` を `typing.Any` にし、`object` のメソッドの override に `@typing.override` を付ける。`tests/mujou/_core/test_native_stub.py` がスタブと実行時のモジュールの名前・docstring・基底クラスのずれを検出する
  - `PauseTimeoutError` をクラスにできないのは、pyo3 0.30 が `PyTimeoutError` を `BaseException` のレイアウトで宣言しており、継承するとクラスの初期化に失敗するからである
  - `#[pyo3(signature = ...)]` の文字列の注釈は import を伴わないので、組み込みとモジュール内の名前だけを書く。`collections.abc.Callable` などが要る型は、`IntoPyObject::OUTPUT_TYPE` / `FromPyObject::INPUT_TYPE` を持つ小さな型で表す（`crates/mujou-python/src/hint.rs`）
