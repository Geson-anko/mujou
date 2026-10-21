# 構成

モジュールの木と、各モジュールの責務、境界の決め方を定める。層の順序と `_native` を import してよいモジュールの一覧の正本は `tests/test_layers.py` で、この文書は理由を書く。

## 方針

- 依存は DAG（有向非巡回グラフ）にする。上の層は下の層だけを import し、同じ層どうしも import しない。循環、関数の中の import、`TYPE_CHECKING` で逃がした依存は持たない。`tests/test_layers.py` が AST で全モジュールを読んで強制する（ruff の禁止規則などと併用しない。仕組みを 2 つ持つと正本も 2 つになる）
- 1 つの事実は 1 か所に置く。利用者のオブジェクト（Node、Port、Network）は宣言だけを持ち、実行時の事実は実行の側（Driver、Wire、Run）が持つ。そのため凍結・解除・差し直し・巻き戻しが要らない
- 特別扱いの分岐はデータの形を変えて消す。たとえば「まだ届いていない」は時刻 `None`、終わった後に止まる時計は `min(timer, end)`、Tensor は名前 `""` のフィールド 1 つのメッセージ、「学習しない」は学習の間隔を無限大で表す
- 抽象と間接層は、実在する使い手が 2 人以上いるときだけ作る

## Python の層

下から順に並べる（`tests/test_layers.py` の `LAYERS`）。

| 層 | モジュール | 責務 |
| --- | --- | --- |
| 0 | `_core/errors.py`、`_core/_native`（Rust） | `NetworkError`（何も import しない葉。Rust も import して raise に使う）と、Rust の拡張モジュール |
| 1 | `_core/spec.py` | `MessageSpec`（抽象）、`TensorSpec`、`DictSpec` と、dtype の対応表 `DLPACK` |
| 2 | `_core/codec.py` | torch とコアの唯一の変換（[メッセージ](messages.md#codec)） |
| 3 | `_core/port.py` | `Port`、`MultiPort`、`Channel`（宣言）と `Wire`（実行中の読み書き）、`running()` |
| 4 | `_core/component.py` | `Node` と `Network` の共通の基底。クラス定義の時点の検査（K1）だけを持つ |
| 5 | `_core/node.py` | 利用者が継承する `Node`。設定、状態の規則、既定の `state_dict` / `load_state_dict` |
| 6 | `_core/network.py` | `Network`、`Group(**named)` |
| 7 | `_core/driver.py` | `Driver`: Rust が hook のたびに呼ぶ、ノード 1 つ分の入口 |
| 8 | `_core/tree.py` | `walk(root)`: `__dict__` を代入の順に読んで木を作り、検査する純関数 |
| 9 | `_core/wiring.py` | `wire(tree, clock)`: 木から native のスロットと、各ノードの Driver を作る |
| 10 | `_core/state.py` | 状態の木と実行の間の変換（`capture` / `restore`） |
| 11 | `_core/run.py` | `Run`、`ManualRun`、`run()`。起動の 3 段、走った印、終了、atexit |
| 12 | `mujou/__init__.py`、`mujou/time.py`、`mujou/testing.py` | 公開の入口（再エクスポートだけ） |
| 13〜17 | `mujou/pc/*` | PC-ALM。公開名だけを import する（[更新手法](update-rules.md)） |

`network.py` が `node.py` の上にあるのは、`Group` の注釈が `Node` を使うから。`tree.py` が `driver.py` の上にあるのは、木の検査がノードの状態の規則（`node.is_state`）を共有するからで、Driver は木を知らない。`state.py` は `wiring.py` が作ったスロットを読み書きするので、その上に置く。

実行時に `_native` を import してよいのは、`codec.py`（Schema）、`port.py`（`current`）、`node.py`（`ReadOnlyTimer` の型）、`wiring.py`（Outlet、Slot、InputGroup）、`run.py`（`start` とタイマー）、`mujou/time.py`（再エクスポート）の 6 つだけにする。native の型が `_core` 全体に散ると、ABI を変えるたびに触る場所が増えるからである。

サブモジュールはすべて明示的に import する（`import mujou.time`、`import mujou.testing`、`import mujou.pc as pc`）。`import mujou` はサブモジュールを読み込まない。規則を 1 文で言えるようにし、`__init__` が pc を読むことによる循環を防ぐ。

## Rust の木

```
crates/mujou-core/src/
  time.rs              Timer、Timestamp、PauseHold、RunClock、global()
  dlpack/ffi.rs        dlpack.h を写した構造体（葉）
  schema.rs            FieldSpec、Schema、FieldError
  buffer.rs            BufferPool、Buffer、Message（値と time_ns: Option<u64>）
  dlpack/import.rs     copy_from: Schema と照合し、stride を扱いながら必ずコピーする
  dlpack/export.rs     DLPack での貸し出し
  slot.rs              Slot（Source で書き手を区別する）、Outlet、InputGroup
  runtime/mod.rs       Runtime、RunConfig、NodeSpec、NodeHooks、Reply と、共有の状態の操作
  runtime/lifecycle.rs ノードのライフサイクルの唯一の状態機械
  runtime/sched.rs     ワーカー（プールと専用スレッド）
  runtime/control.rs   制御スレッド（起動・ステップ・取得・停止）
  runtime/error.rs     Hook、HookError、NodeFailure、RuntimeError
crates/mujou-python/src/
  lib.rs               #[pymodule] の宣言だけ
  wait.rs              シグナルを見ながら待つ処理と、秒の変換の唯一の実装
  errors.rs            mujou._core.errors.NetworkError の参照
  current.rs           thread-local の CURRENT と current()
  time.rs              Timer、VirtualTimer、ReadOnlyTimer、Timestamp、global_timer
  slot.rs              Schema、Outlet、Slot、InputSlot、InputGroup、Message、BufferView
  dlpack.rs            capsule の借用と貸し出し
  runtime.rs           start と Runtime、Driver を呼ぶ PyDriver、Hook から Driver のメソッド名への対応表、
                       ワーカー数の既定
  hint.rs              スタブ用の型注釈
crates/mujou-stubgen/src/main.rs   _native.pyi を生成する
```

依存の向きは次のとおり。`runtime` はメッセージもスロットも知らない。

```
time（葉）          dlpack::ffi（葉）
                    schema ← ffi
                    buffer ← schema
                    dlpack::{import,export} ← schema, buffer
slot ← buffer, schema, time
runtime ← time
runtime/mod → {lifecycle, sched, control, error}
mujou-python: wait ← errors ← time ← slot(dlpack) ← runtime ← lib
```

`runtime` の子（`sched` と `control`）は親の `Shared` と `Control` を読み、そのメソッドを呼ぶ。使い手が 1 人の trait を挟まないためで、子どうしは互いを知らない。

## Rust と Python の境界

Rust は仕組みを、Python は方針を持つ。

| Rust（仕組み） | Python（方針） |
| --- | --- |
| スロット、バッファプール、shape と dtype の照合（`copy_from`） | spec、メッセージの構造の検査、torch との変換（`codec.py`） |
| タイマー、RunClock、停止の保持 | 実行のタイマーの選び方（既定、渡したもの、復元した時刻から） |
| ライフサイクルの状態機械、ワーカー、周期の待ち | hook で何をするか（Driver）、CUDA の device と stream |
| hook の呼び出しと例外の捕捉、終了の流れ | 木の読み方、パスと label、ガード、状態の木の形 |
| 「いま hook を走らせている Driver」（bindings の `CURRENT`） | Port の所有者の検査（Driver の `wires` の lookup） |

- コアは学習則を知らない。`NodeHooks` が hook の契約のすべてで、どの更新手法も載る
- コアは torch を import しない。利用者がやり取りする値は `torch.Tensor` だけで、コアの値の型と dtype は公開しない
- 例外は Python で定義し、Rust はそれを import して raise する（`NetworkError`）。timeout は組み込みの `TimeoutError`
- 境界の約束は 1 か所ずつにする。torch とコアの変換は `codec.py`、shape と dtype の検査は Rust の `copy_from`、「いま hook を走らせている Driver」は bindings の thread-local
- Rust は Python の `Driver` のメソッドを、hook 1 回につき 1 回だけ呼ぶ（[ノード](nodes.md#driver)）

## torch への依存

torch は必須の依存で、ノードの内側の計算に全面的に採用する。torch に依存させないと決めているのは `_native` と `mujou/time.py` だけで、ほかのモジュールは要るなら torch を使ってよい。バックエンドを分けた `mujou.torch` などは持たない。

GPU はコアでサポートしない。コアの値はホストのメモリにだけ置き、GPU を使うノードは境界で device → host → device とコピーする。ノードの内側での GPU の利用は自由（[ノード](nodes.md#cuda)）。

## free-threaded の規則

開発は free-threaded 版の Python 3.14t で行い、GIL ありの 3.14 もサポートする（[ネイティブコア](native-core.md#ツールチェーン)）。どちらのビルドでも正しく動くよう、次を守る。

- GIL による暗黙の排他に頼らない。共有するものは lock で守るか、1 つのスレッドだけが書く
- Rust の拡張は free-threaded に対応していることを明示する（`#[pymodule(gil_used = false)]`）。待つ間は Python から detach し、シグナル（Ctrl-C）を見る
- 依存（特に C 拡張）を足すときは free-threaded への対応を確かめる。未対応の C 拡張を import すると GIL が有効に戻る（`torch.optim` も GIL を有効にするので、例では勧めない）
- ノードの hook はワーカーの間を移る。hook の間で保たれるのは、ノードごとの `contextvars.Context` と、Driver が設定する CUDA の device と stream だけ（[ノード](nodes.md#スレッド)）

## 内部の名前の規約

- private なパッケージ `mujou._core` の中のモジュールの、`_` の無いモジュールレベルの名前は、パッケージ内部の API である（`spec.DLPACK`、`port.Wire`、`port.running`、`tree.walk`、`node.is_state`、`node.initialized` など）。利用者の公開名は、`mujou`、`mujou.time`、`mujou.testing`、`mujou.pc` が再エクスポートしたものだけ（`tests/mujou/test_init.py`）
- `mujou.pc` の private なモジュール（`_layer.py` など）の、`_` の無い名前も同じくパッケージ内部の API（`_messages.rows` など）
- 公開クラス（Node、Port、MultiPort、Network）の `_` 属性には、そのクラスの外から触らない。実行時の事実は実行の側に置いたので、触る必要がある箇所は無い。src の `# pyright: ignore[reportPrivateUsage]` は 0 件にする
- Node の設定は、名前修飾した 1 属性（`__config`）に置く。利用者の `self._device` などと衝突しない
- フレームワークは、ノードの設定を公開の property（`period`、`device`、`dedicated_thread`、`generator`）だけから読む。サブクラスが property を置き換えても、フレームワークと食い違わない
- 走った印は `run.py` の `WeakSet` が持つ。印を付けるのも見るのも `run.py` だけ。そのため Node のサブクラスで `__eq__` を定義する（hash できなくなる）ことはサポートしない
