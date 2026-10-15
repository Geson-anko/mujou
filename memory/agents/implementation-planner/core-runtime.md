# コアのランタイム（`mujou.run(network)` まで）

## 概要

スロット（Rust）、ノードの実行と制御（Rust）、ネットワークの宣言と build（Python）、それらの結び付けとホスト入出力（Python）を実装し、`mujou.run(network)` でネットワークをバックグラウンドで動かせるようにする。WP-A/B/C を別 worktree で並列に実装し、そのあと WP-D で結合する。

前提にした docs・memory は依頼文の一覧のとおり。docs と食い違う点や docs で未決定の点は「確認事項」に推奨案を添えて挙げた。

## 公開インターフェース案

### 全体の層

| 層 | 場所 | torch | WP |
| --- | --- | --- | --- |
| スロット・バッファ・DLPack | `crates/mujou-core/src/{slot,dlpack}.rs` → `_core.{Schema,Outlet,Slot,InputGroup,InputSlot,Message,BufferView}` | 非依存 | A |
| ノードの実行と制御 | `crates/mujou-core/src/runtime.rs`（＋`time.rs` に 2 関数） → `_core.{Node,Runtime,NetworkError,PauseTimeoutError}` | 非依存 | B |
| spec・Port・Network の構造 | `src/mujou/{_spec,_port,_network,_node}.py` | 依存 | C |
| 結び付け・ホスト入出力・`mujou.Node` の torch 設定・`run` | `src/mujou/{_session,_port,_network,_node}.py` | 依存 | D |

Python の実装モジュールは `_` 付き（利用者は `mujou.*` から使う）。テストは `_core` ↔ `test_core.py` の前例に倣い、先頭の `_` を落とした名前でミラーする（`_spec.py` ↔ `tests/mujou/test_spec.py`）。`_core` のテストは衝突を避けて `test_core_slot.py`（A）・`test_core_runtime.py`（B）に分ける。

### WP-A: Rust（`mujou_core::slot` / `mujou_core::dlpack`）

- `pub struct DType { pub code: u8, pub bits: u8, pub lanes: u16 }` — DLPack の `DLDataType` そのもの。独自の dtype は持たない
- `pub struct FieldSpec { pub name: String, pub shape: Vec<i64>, pub dtype: DType }` / `fn nbytes(&self) -> usize`
- `pub struct Schema { fields: Vec<FieldSpec> }` / `pub fn new(fields: Vec<FieldSpec>) -> Result<Self, SlotError>`（フィールド 1 つ以上、各次元 ≥1、`()` 可、lanes=1、名前の重複なし）
- `pub struct Buffer` — プールから借りた不変のバイト列。`fn bytes(&self) -> &[u8]`、`Drop` でプールへ戻る
- `pub struct Message { buffers: Vec<Buffer>, time_ns: u64 }` — 不変。`fn buffers(&self) -> &[Buffer]`、`fn time_ns(&self) -> u64`。常に `Arc<Message>` で共有する
- `pub struct Slot` — `ArcSwap<Message>` の最新値。`fn name(&self) -> &str`、`fn load(&self) -> Arc<Message>`、`fn wait_newer(&self, last: Option<&Arc<Message>>, timeout: Duration) -> Option<Arc<Message>>`（`last` と `Arc::ptr_eq` でない値になるまで待つ）、`fn restore(&self, msg: Arc<Message>)`
- `pub struct Outlet` — 1 つの Port の送り側の唯一の書き手。`!Clone`。
  - `pub fn new(schema: Arc<Schema>, channels: usize, name: &str, timer: &'static GlobalTimer) -> Self`（外向きスロット `"{name}/out"` と Channel のスロット `"{name}/ch{k}"` をゼロ・作成時刻で作る）
  - `pub fn outbox(&self) -> &Arc<Slot>`、`pub fn channel(&self, k: usize) -> &Arc<Slot>`
  - `pub fn send_with<E>(&self, target: Target, fill: impl FnOnce(&mut [&mut [u8]]) -> Result<(), E>) -> Result<(), E>`（`Target::All` は外向き＋全 Channel に同じ `Arc<Message>`、`Target::Channel(k)` はそのスロットだけ。プールのバッファへ `fill` で書き、時刻を付けて差し替える）
- `pub struct InputGroup` — 版番号 V（`AtomicU64`）と書き込みの直列化 lock。`pub fn new(timer: &'static GlobalTimer) -> Arc<Self>`、`pub fn input(self: &Arc<Self>, schema: Arc<Schema>, name: &str) -> Arc<InputSlot>`（名前は `"{name}/in"`）、`pub fn write<E>(&self, items: Vec<(&InputSlot, Box<dyn FnOnce(&mut [&mut [u8]]) -> Result<(), E>>)>) -> Result<(), E>`（全部 fill してから全部置き、最後に V を進める。fill が 1 つでも失敗したら何も置かない）
- `pub struct InputSlot` — `ArcSwap<Pair { current: Arc<Message>, next: Option<(u64, Arc<Message>)> }>`。`fn load(&self) -> Arc<Message>`（Pair を読んでから V を Acquire で読み、`next.version <= V` なら next）、`fn name`、`fn restore(&self, msg)`（current に置き next を消す）
- `pub enum SlotError` — スキーマ不正・shape/dtype 不一致・デバイス不正・DLPack の不正。`Display` に「スロット名・フィールド名・期待値・実際の値」を含める
- `mujou_core::dlpack`: `#[repr(C)]` の `DLDevice` / `DLDataType` / `DLTensor` / `DLManagedTensor` / `DLManagedTensorVersioned` / `DLPackVersion`、`pub unsafe fn copy_from(t: &DLTensor, field: &FieldSpec, dst: &mut [u8]) -> Result<(), SlotError>`（デバイスは `kDLCPU`=1 と `kDLCUDAHost`=3 だけ、strides・byte_offset を見て row-major へ詰めてコピー、strides が null なら連続）、`pub fn export_versioned(msg: Arc<Message>, i: usize) -> *mut DLManagedTensorVersioned` と `pub fn export_legacy(...) -> *mut DLManagedTensor`（manager_ctx が `Arc<Message>` を持ち、deleter で手放す）

### WP-A: Python（`_core`、`crates/mujou-python/src/slot.rs`）

- `type DLDataType = tuple[int, int, int]`（code, bits, lanes）
- `class Schema: def __init__(self, fields: Sequence[tuple[str, Sequence[int], DLDataType]]) -> None` — Tensor のメッセージは名前 `""` のフィールド 1 つ、TypedDict はキー順
- `class BufferView: def __dlpack__(self, *, stream: object = None, max_version: tuple[int, int] | None = None, dl_device: tuple[int, int] | None = None, copy: bool | None = None) -> CapsuleType` / `def __dlpack_device__(self) -> tuple[int, int]`（`(1, 0)`）。`max_version` が `(1, *)` 以上なら `"dltensor_versioned"`、そうでなければ `"dltensor"` の capsule
- `class Message: buffers: tuple[BufferView, ...]`（property、キー順）/ `time_ns: int`（property）
- `class Slot: name: str` / `def read(self) -> Message` / `def wait_newer(self, last: Message | None, timeout: float) -> Message | None`（GIL を外して待つ）/ `def restore(self, values: Sequence[SupportsDLPack], time_ns: int) -> None`
- `class Outlet: def __init__(self, schema: Schema, channels: int, name: str) -> None` / `outbox: Slot` / `def channel(self, index: int) -> Slot` / `def send(self, values: Sequence[SupportsDLPack]) -> None` / `def send_to(self, index: int, values: Sequence[SupportsDLPack]) -> None`
- `class InputGroup: def __init__(self) -> None` / `def input(self, schema: Schema, name: str) -> InputSlot` / `def write(self, items: Sequence[tuple[InputSlot, Sequence[SupportsDLPack]]]) -> None`
- `class InputSlot: name: str` / `def read(self) -> Message` / `def restore(self, values: Sequence[SupportsDLPack], time_ns: int) -> None`
- `SupportsDLPack` は stub 内の Protocol（`__dlpack__` を持つ任意のオブジェクト）
- 例外: shape・dtype・フィールド数の不一致とスキーマ不正は `ValueError`、`__dlpack__` を持たない・デバイスが CPU でないは `TypeError`、`index` の範囲外は `IndexError`
- 時刻は常に `mujou_core::time::global()` から付ける（bindings は timer を渡さない）

### WP-B: Rust（`mujou_core::runtime`、`time.rs` の追加）

- `time.rs` に追加: `pub fn wait_until_ns_or(&self, target_ns: u64, timeout: Duration, interrupted: impl Fn() -> bool) -> Wait`（`Wait::{Reached, Interrupted, TimedOut}`。`interrupted` はタイマーの lock の下で確かめる）と `pub fn interrupt_waiters(&self)`（lock を取って `notify_all`）
- `pub struct HookError { pub hook: &'static str, pub source: Arc<dyn Error + Send + Sync> }`
- `pub trait NodeHooks: Send + 'static { type State: Send + 'static; fn start(&mut self) -> Result<(), HookError>; fn update(&mut self, dt: f64) -> Result<(), HookError>; fn pause(&mut self) -> Result<(), HookError>; fn resume(&mut self) -> Result<(), HookError>; fn exception(&mut self) -> Result<(), HookError>; fn stop(&mut self) -> Result<(), HookError>; fn state_dict(&mut self) -> Result<Self::State, HookError>; }` — `start` が「内部の設定 hook → `on_start` → 復元時は `load_state_dict`」の 3 つを含む
- `pub struct NodeFailure { pub node: usize, pub hook: &'static str, pub error: Arc<dyn Error + Send + Sync> }`
- `pub struct RuntimeConfig { pub manual: bool, pub pause_timeout: Duration, pub last_update_ns: Option<Vec<u64>>, pub timer_state: Option<TimerState> }`
- `pub type ExceptionCapture<S> = Box<dyn FnOnce(BTreeMap<usize, S>, &NodeFailure) + Send>` — 例外のときの状態の取得。`state_dict` が例外を投げたノードは map に入らない
- `pub struct Runtime<N: NodeHooks>`
  - `pub fn start(nodes: Vec<N>, names: Vec<String>, timer: &'static GlobalTimer, config: RuntimeConfig, capture: Option<ExceptionCapture<N::State>>) -> Result<Self, RuntimeError>` — 全ノードの `start` が終わり、タイマーを再開してから戻る。起動中の例外は終了処理を最後まで済ませてから `Err(Failed)`
  - `pub fn step(&self, order: Option<&[usize]>) -> Result<(), RuntimeError>` — 手動モードだけ
  - `pub fn capture<R: Send + 'static>(&self, build: Box<dyn FnOnce(Vec<N::State>) -> R + Send>) -> Result<R, RuntimeError>` — 停止 → 全ノードの `state_dict` → `build` → 再開。`build` は停止中に制御スレッドで呼ぶ
  - `pub fn shutdown(&self) -> Result<(), RuntimeError>` / `pub fn join(&self, timeout: Option<Duration>) -> Result<bool, RuntimeError>`（終わっていれば true）
  - `pub fn failure(&self) -> Option<&NodeFailure>` / `pub fn is_finished(&self) -> bool` / `pub fn last_update_ns(&self) -> Vec<u64>`（atomic を読むだけ）
- `pub enum RuntimeError { Failed(NodeFailure), PauseTimeout { nodes: Vec<String> }, NotManual, Finished }`
- タイマーは引数で受け取る（`cargo test` が並列にスレッドで走るので、テストごとに `Box::leak` した `GlobalTimer` を使うため）。bindings は常に `time::global()` を渡す

### WP-B: Python（`_core`、`crates/mujou-python/src/runtime.rs`）

- `class Node`（`#[pyclass(subclass)]`、`__new__` は任意の引数を受けて捨てる）: `def _on_thread_start(self) -> None` / `def on_start(self) -> None` / `def on_update(self, dt: float) -> None` / `def on_pause(self) -> None` / `def on_resume(self) -> None` / `def on_exception(self) -> None` / `def on_stop(self) -> None` / `def state_dict(self) -> object`（既定 `None`）/ `def load_state_dict(self, state: object) -> None` — すべて既定は何もしない
- `class NetworkError(RuntimeError)` — 例外で終了したことを表す。`__cause__` に最初の例外。メッセージに「ノードの名前と hook 名」
- `class PauseTimeoutError(TimeoutError)` — 停止が時間内に終わらなかった。メッセージに止まらなかったノードの名前
- `class Runtime`
  - `def __init__(self, nodes: Sequence[Node], names: Sequence[str]) -> None`
  - `def start(self, *, manual: bool = False, states: Sequence[object] | None = None, last_update_ns: Sequence[int] | None = None, timer_state: Mapping[str, int | float] | None = None, pause_timeout: float = 10.0, on_exception_capture: Callable[[dict[int, object], int, str, BaseException], object] | None = None) -> None` — 起動が終わるまで GIL を外して待つ。起動中の例外は終了処理の後に `NetworkError`。2 回目は `RuntimeError`
  - `def step(self, order: Sequence[int] | None = None) -> None`
  - `def capture[T](self, build: Callable[[list[object]], T], *, timeout: float | None = None) -> T` — `timeout` 省略時は `start` の `pause_timeout`。`build` が投げた例外はそのままホストへ（システムは止めない）。ノードの `state_dict` の例外はシステムを終了させ `NetworkError`
  - `def shutdown(self) -> None` / `def join(self, timeout: float | None = None) -> bool` / `def check(self) -> None`（例外で終了していれば `NetworkError`）
  - `is_running: bool` / `is_finished: bool`（property）/ `def last_update_ns(self) -> list[int]` / `def exception_state(self) -> object | None`（`on_exception_capture` の戻り値）
  - 待つメソッドはすべて GIL を外し、50 ms ごとに `check_signals`（`sleep` と同じ）
- hook の例外は `logging.getLogger("mujou")` に `exc_info` 付きで出す（2 つ目以降の例外、`on_exception`・`on_stop` の例外、例外時の取得での `state_dict` の例外も同じ）

### WP-C: Python（構造）

- `src/mujou/_spec.py`
  - `DLPACK_DTYPES: Final[Mapping[torch.dtype, tuple[int, int, int]]]` — 10 種（float16 `(2,16,1)`、bfloat16 `(4,16,1)`、float32 `(2,32,1)`、float64 `(2,64,1)`、int8/16/32/64 `(0,b,1)`、uint8 `(1,8,1)`、bool `(6,8,1)`）
  - `class MessageSpec[M]`（抽象基底。`M` はメッセージの型）
  - `class TensorSpec(MessageSpec[torch.Tensor])`: `def __init__(self, shape: Sequence[int], dtype: torch.dtype | None = None) -> None` / `shape: tuple[int, ...]` / `dtype: torch.dtype` / `def zeros(self) -> torch.Tensor` / `def rand(self, seed: int | None = None, *, generator: torch.Generator | None = None) -> torch.Tensor` / `__eq__`・`__hash__`・`__repr__`
  - `class DictSpec[TD: Mapping[str, object]](MessageSpec[TD])`: `def __init__(self, typed_dict: type[TD], /, **specs: TensorSpec) -> None` / `typed_dict: type[TD]` / `specs: Mapping[str, TensorSpec]`（TypedDict の定義順）/ `def zeros(self) -> TD`
  - `type Spec = TensorSpec | DictSpec[Any]`
  - `def fields(spec: Spec) -> list[tuple[str, tuple[int, ...], tuple[int, int, int]]]` — `_core.Schema` に渡す形
  - `def split(spec: Spec, msg: object, *, where: str) -> list[torch.Tensor]` — 型・キー・shape・dtype を検査してキー順の列にする（`detach` や CPU への移動はしない）
  - `def join(spec: Spec, tensors: Sequence[torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]`
- `src/mujou/_port.py`
  - `class Single` / `class Multi` — 相手の数を表す型の印（中身なし）
  - `class Port[S = Never, R = Never, M: (Single, Multi) = Single]`: `def __init__(self, send: MessageSpec[S] | None = None, recv: MessageSpec[R] | None = None, *, multi: bool = False) -> None`（`multi: Literal[False]` / `Literal[True]` の overload で `M` を決める）
    - `send_spec: MessageSpec[S] | None` / `recv_spec: MessageSpec[R] | None` / `multi: bool` / `channels: tuple[Channel, ...]`（接続順）
    - `def connect(self, other: Port[Any, Any, Any]) -> Channel`
    - `def recv(self: Port[Any, R2, Single]) -> R2` / `def recv(self: Port[Any, R2, Multi]) -> dict[Channel, R2]`（self 型の overload）
    - `def recv_with_time(self: Port[Any, R2, Single]) -> tuple[R2, int]` / `(self: Port[Any, R2, Multi]) -> dict[Channel, tuple[R2, int]]`（時刻は ns）
    - `def send(self, msg: S) -> None` / `def send_to(self, channel: Channel, msg: S) -> None`
    - C の段階では recv/send 系は「未 build」の `RuntimeError` を投げる本体にし、D が中身を入れる
  - `class PrevPort[S, R, M = Single](Port[S, R, M])`、`class NextPort[S, R, M = Multi](Port[S, R, M])`（既定 `multi=True`）
  - `@final class Channel`: `ends: tuple[Port, Port]` / `def other(self, port: Port[Any, Any, Any]) -> Port[Any, Any, Any]`。`connect` だけが作る。`__eq__` は定義しない（同一性で dict のキーになる）
- `src/mujou/_network.py`
  - `class Component`（Node と Network の共通基底。`__init__` で状態を持たない）: `def connect(self, other: Component) -> Channel`（自分の `NextPort` と相手の `PrevPort`）
  - `class Network(Component)`: `def build(self) -> None`
  - `@dataclass(frozen=True) class Graph`: `nodes: tuple[tuple[str, Node], ...]`（パスと node。深さ優先・属性の定義順）/ `ports: Mapping[Port[Any, Any, Any], tuple[int, str]]`（所有ノードの番号と属性名）/ `def port_path(self, port) -> str`（`"body.l1.next"`）。`Network.build()` が作り、`self._graph` に置く
- `src/mujou/_node.py`: `class Node(Component)` の骨組み（C の段階では `_core.Node` を継承しない。D が基底に足す）

### WP-D: Python（結合とホスト入出力）

- `class Node(_core.Node, Component)`: `def __init__(self, *, device: torch.device | str = "cpu", seed: int | None = None) -> None` / `device: torch.device`（property）/ `generator: torch.Generator`（property。スレッドの起動前は `RuntimeError`）/ `_on_thread_start` を override。`super().__init__()` を呼ばないサブクラスでも動くよう、`_device` と `_seed` はクラス属性に既定値を置く
- `Network` に追加: `def run(self, *, capture_state_on_exception: bool = False, pause_timeout: float = 10.0, _manual: bool = False) -> None` / `def write(self, values: Mapping[Port[Any, Any, Any], torch.Tensor | Mapping[str, torch.Tensor]]) -> None` / `def read[S](self, port: Port[S, Any, Any]) -> tuple[S, float]` / `def subscribe[S](self, port: Port[S, Any, Any]) -> Iterator[tuple[S, float]]` / `def state_dict(self) -> dict[str, Any]` / `def load_state_dict(self, state: Mapping[str, Any]) -> None` / `def exception_state_dict(self) -> dict[str, Any] | None` / `def shutdown(self) -> None` / `def join(self) -> None` / `def _step(self, order: Sequence[Node] | None = None) -> None`
- `def run[N: Network](network: N, *, capture_state_on_exception: bool = False, pause_timeout: float = 10.0, _manual: bool = False) -> N` — 未 build なら build してから `network.run(...)` を呼び、同じ `network` を返す
- `NetworkError = _core.NetworkError`、`PauseTimeoutError = _core.PauseTimeoutError` を再公開
- `src/mujou/__init__.py` の公開名: `Node, Network, Port, PrevPort, NextPort, Channel, TensorSpec, DictSpec, NetworkError, PauseTimeoutError, run, __version__`
- 状態の木: `{"nodes": {path: state}, "slots": {slot_name: (Tensor | dict[str, Tensor], time_ns)}, "last_update_ns": {path: int}, "timer": {"time_ns": int, "scale": float}}`。例外のときの木はさらに `"exception": {"node": path, "hook": str, "error": str}`（トレースバックの文字列。例外オブジェクトは pickle できないので入れない）
- スロット名: `"{node のパス}.{Port の属性名}/out"`・`"/ch{k}"`（k は書く側の Port での接続順）・`"/in"`

## 理論上の対応

学習則には触れない。実行モデル（`docs/execution-model.md`）をそのまま実装する。

- 局所性: ノードが読めるのは自分の Port に結び付いたスロットだけ。Port の send は所有ノードのスレッドからだけ受け付ける（D が `_on_thread_start` で記録したスレッドと照合）。ノードが他のノードを属性に持つ構成は build でエラーにする。受け取った値は毎回コピーした新しい Tensor で、勾配はつながらない（送るときに `detach`）
- 非同期性: スロットは書き手 1 つ・読み手 N の最新値。ノードは互いを待たない。バリアは制御の操作（停止・状態の取得・終了・手動ステップ）だけで、停止中はタイマーが凍結している
- 更新手法の非依存: `NodeHooks`・`Slot`・`Runtime` はメッセージの中身も集約も知らない。複数の相手の値は接続順の dict で渡すだけ
- 連続時間: Δt は ns の差（`(now_ns - last_ns) / 1e9`）。停止 → 凍結 → 区切りで停止、の順を守るので、停止中の最後の更新時刻とスロットの時刻は凍結時刻 T 以下
- 手動モードは非同期実行の特殊ケースのスケジュールとして同じループで実装する（同期専用の経路を作らない）。違いは「区切りで、ステップの許可が来るまで待つ」ことだけ

## 実装ステップ

### WP-0（orchestrator、並列化の前に main から切った統合ブランチで）

1. `uv add torch`（確認事項 3）。`uv.lock` もコミットする
2. 衝突を避ける骨組みを先に置く: `crates/mujou-core/src/lib.rs` に `pub mod slot; pub mod dlpack; pub mod runtime;`、空の `slot.rs`・`dlpack.rs`・`runtime.rs`、`crates/mujou-python/src/{slot,runtime}.rs` に空の `pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()>`、`lib.rs` に `mod slot; mod runtime;` と `slot::register(m)?; runtime::register(m)?;`、`_core.pyi` に `# --- slot (WP-A)` と `# --- runtime (WP-B)` の区切り、`src/mujou/{_spec,_port,_network,_node,_session}.py` の空ファイル
3. A/B/C をこのブランチから切る。マージ順は A → B → C、そのあと D を切る

### WP-A（Rust のスロット）

1. `crates/mujou-core/Cargo.toml` に `arc-swap` を足す（A だけが触る）
2. `dlpack.rs`: `repr(C)` の構造体、`copy_from`（N 次元の strided コピー。要素サイズ単位で、負の stride も符号付きオフセットで扱う）、`export_versioned` / `export_legacy`。READ_ONLY フラグは立てない（受け手が必ずコピーするので不要で、torch が読み取り専用の DLPack を拒む可能性がある）
3. `slot.rs`: `BufferPool`（`nbytes` 固定、空きリストは短い臨界区間の `Mutex<Vec<Box<[u8]>>>`。競合が測定で問題になったら lock-free キューへ）、フィールドごとのプールの組、`Buffer` の `Drop` での返却、`Slot`（外向きスロットだけ `wait_newer` 用の通知を持つ。書き手は待ち手が 0 なら lock を取らない）、`Outlet`、`InputGroup` / `InputSlot`
4. bindings `crates/mujou-python/src/slot.rs`: capsule の受け取り（`__dlpack__(max_version=(1, 0))` を試し、`TypeError` なら引数なしで呼び直す → capsule 名で版を判別 → `copy_from` → deleter を呼ぶ → capsule の名前を `used_dltensor(_versioned)` に変える）、`BufferView.__dlpack__`、各 pyclass。fill のクロージャの中で Python を呼ばないよう、`__dlpack__` の呼び出しはフィールドごとに先に済ませてから `send_with` に渡す
5. `_core.pyi` の A の区切りに stub を書く
6. テスト: Rust の単体テストと `tests/mujou/test_core_slot.py`（torch を使う）

### WP-B（Rust のランタイム）

1. `time.rs` に `wait_until_ns_or` と `interrupt_waiters`
2. `runtime.rs` の共有状態: 指示（`Run` / `Pause` / `Stop`）と手動ステップの許可を `Mutex` + `Condvar`、ノードごとの報告（起動済み・停止済み・例外処理済み・終了可・終了済み）、`last_update_ns: Vec<AtomicU64>`、最初の例外 `OnceLock<NodeFailure>`、タイマーの凍結の理由（停止要求・例外）。指示を変えたら `Condvar::notify_all` と `timer.interrupt_waiters()` の両方を呼ぶ
3. 制御スレッド（1 本）: ホストからの要求（Start・Step・Capture・Shutdown）を `mpsc` で受け、1 つずつ処理して返信する。ノードの例外の通知でも起き、終了処理を進める。タイマーの pause/resume を呼ぶのは制御スレッドと「例外を捕まえたノードのスレッド（凍結だけ）」に限り、凍結の理由が全部消えたときだけ resume する（状態の取得の再開が例外の凍結を解かない）
4. ノードのスレッド（ノードごとに 1 本）:
   1. `start()` → 報告。例外なら例外の処理（下記 6）へ
   2. 開始の合図を待つ。`last_ns` は `config.last_update_ns` か、合図の時点の時刻
   3. ループの区切り: 指示を見る → `Stop` なら抜ける / `Pause` なら `pause()` → 停止済みを報告 → 指示が変わるまで待つ → `Stop` なら `resume()` を呼ばずに抜ける、そうでなければ `resume()` / 手動モードなら許可か指示の変化を待つ
   4. `now = timer.time_ns()`。`now == last_ns` なら、手動モードでは更新せずにステップ完了を報告し、自由実行では `wait_until_ns_or(last_ns + 1, ..., 指示が変わった)` で待って区切りに戻る（ビジーループしない）
   5. `update((now - last_ns) as f64 / 1e9)` → `last_ns = now` → `last_update_ns[i].store(now)`
   6. 抜けた後: 例外で終わるときは自分の `exception()`（失敗はログだけ）→ 報告 → 制御スレッドの「on_stop してよい」を待つ（待つ間に例外の指示が来たら `exception()` を先に呼ぶ）→ `stop()`（失敗はログだけ）→ 終了済みを報告
5. 停止（Capture の中）: 指示を `Pause` → すぐ凍結 → 全ノードの停止済みをタイムアウト付きで待つ → 時間切れなら再開して `PauseTimeout`（停止済みのノードは `on_resume` を呼ばれる）→ 停止中に `hooks.state_dict()` を各ノードで呼ぶ（hooks は `Arc<Mutex<N>>` に入れ、ノードのスレッドは停止中は lock を持たない）→ `build` → 再開（タイマーを再開してから指示を `Run` に戻す）
6. 例外: hook の例外を捕まえたスレッド（ノードでも制御スレッドでも）が、最初の例外なら記録し、ただちに指示を `Stop`（例外つき）にし、例外のときの取得が設定されていればすぐ凍結する。制御スレッドは全ノードの例外処理済みを待ち → 取得の設定があれば各ノードの `state_dict`（失敗したノードは除いてログ）と `capture` を呼ぶ → 全ノードに on_stop を許可 → join → 自分が凍結したタイマーを再開（確認事項の一括確認 4）
7. 正常な終了: 指示を `Stop` → 全ノードの停止済みを待つ（タイムアウトなし）→ on_stop を許可 → join
8. 手動モード: `step(order)` は順にノード i へ許可を出し、i の完了を待ってから次へ。途中の例外は終了処理を済ませてから `Failed`
9. bindings `crates/mujou-python/src/runtime.rs`: `PyNodeHooks`（`Py<Node>` と復元用の状態を持ち、各 hook で `Python::attach` してメソッドを名前で呼ぶ。`start` は `_on_thread_start` → `on_start` → 状態があれば `load_state_dict`。どれが投げたかを `hook` に入れる）、`create_exception!` で `NetworkError` / `PauseTimeoutError`、`RuntimeError` から Python 例外への変換（`Failed` は `NetworkError` に `__cause__` を付ける）、ログ出力
10. `_core.pyi` の B の区切りに stub
11. テスト: Rust の単体テスト（Rust のモックのノードで hook の呼び出し順を記録）と `tests/mujou/test_core_runtime.py`（torch を使わない `_core.Node` のサブクラス）

### WP-C（Python の構造）

1. `_spec.py`: `DLPACK_DTYPES`、`TensorSpec`（dtype の省略時は作った時点の `torch.get_default_dtype()`、表にない dtype は `TypeError`、次元は 1 以上の int）、`DictSpec`（`typing.get_type_hints` でキーと値の型を取り、値は `torch.Tensor` だけ、`NotRequired` と入れ子は拒否、キーの過不足は `ValueError`）、`fields` / `split` / `join`
2. `_port.py`: Port 3 種と `Channel`。`connect` は両方の Port の send/recv が逆向きに一致するか（`None` 同士も一致とみなし、両向きとも `None` は拒否）、相手 1 つの Port がすでに接続済みでないか、同じ Port 同士でないか、どちらも build 済みでないかを確かめ、`Channel` を作って両端の `channels` に足す。spec の不一致は `TypeError`、本数超過は `ValueError`
3. `_network.py`: `Component.connect` は自分と相手の直接の属性から `NextPort` / `PrevPort` を探す（無ければ・2 つ以上なら `ValueError`）。`Network.build()` は木を `vars()` の定義順で深さ優先にたどって `Graph` を作り、次を検査する: ルートは `Network`、build は 1 回、同じノードが 2 つのパスに現れない、Port の所有ノードは 1 つ、Network が公開する Port は木の中のノードが所有する、ノードの属性に Component が無い、Channel の両端が木の中にある、各 Component の直接の `PrevPort`・`NextPort` は 1 つまで、ノードが 1 つ以上。構造の誤りは `ValueError`、順序の誤り（2 回目の build、build 後の connect）は `RuntimeError`。build の最後に全 Port を凍結する
4. `_node.py`: `class Node(Component)` の骨組み
5. テスト: `tests/mujou/test_spec.py`・`test_port.py`・`test_network.py`
6. Port の型付けを pyright strict で確かめる（`src/` 側の小さな利用箇所か、C のテストを一時的に pyright にかけて `reveal_type` を見る）。TypeVar の既定値と self 型の overload で解けなければ、確認事項に上げずに「相手 1 つは `recv()`、複数は `recv_all()`」の 2 メソッドに分ける代案へ切り替え、D に通知する

### WP-D（結合、A/B/C のマージ後）

1. `_node.py`: 基底を `(_core.Node, Component)` にし、`_on_thread_start` で `torch.set_grad_enabled(False)`、CUDA なら自分の `torch.cuda.Stream` を作って `torch.cuda.set_stream`、`torch.Generator(device)` を seed で作る（seed が無ければ `generator.seed()`）、このスレッドの ident を記録する
2. `_session.py`: build の後に 1 つ作る `Session`。`Graph` から、送る spec を持つ Port ごとに `Outlet(Schema(fields(spec)), len(port.channels), port_path)`、受ける spec を持つ Port ごとに読む元のスロット列（接続順。相手の Port の `Outlet.channel(相手側での接続番号)`）、未接続で相手 1 つの受ける Port には `InputGroup.input(...)`。Port に結び付け情報を置く
3. `_port.py` の recv/send の本体: recv はスロットを読み、`torch.from_dlpack(view)` をノードのデバイスへコピー（CPU なら `.clone()`、CUDA なら `.to(device)`）して `join`。相手 1 つで未接続かつホストが書いていなければ初期値（ゼロ）、相手が複数で未接続なら `{}`。send は所有ノードのスレッドか確かめ、`split` → `detach` → CUDA なら自分のストリームで `.cpu()` して自分のストリームだけ同期 → `Outlet.send` / `send_to(channels.index(channel), ...)`
4. `_network.py` のホスト入出力: 各メソッドの最初に `runtime.check()`（例外で終了していれば `NetworkError`）。`write` は全 Port が「木の中・未接続・相手 1 つ・受ける spec あり」かを確かめ、`split` → `detach().cpu()` → `InputGroup.write`。`read` は外向きスロット（送る spec が無い Port は `ValueError`）。`subscribe` は `wait_newer(last, 0.05)` を回し、合間に `check_signals` 相当（Rust 側で済む）と終了の確認をする（例外なら `NetworkError`、正常終了ならイテレータを終える）。`state_dict` は `runtime.capture(build)`、`build` の中で `copy.deepcopy`・スロットの読み出し（`.clone()`）・`last_update_ns`・`_core.timer.state_dict()` をまとめる。`load_state_dict` は run の前だけ。パス・スロット名の集合が一致するか確かめ、スロットはその場で `restore`、ノードの状態・最後の更新時刻・タイマーは run まで保持する
5. `run`: `Runtime(nodes, paths).start(manual=_manual, states=..., last_update_ns=..., timer_state=..., pause_timeout=..., on_exception_capture=...)`。`atexit` に、実行中の Session（`weakref.WeakSet`）を `shutdown` する関数を 1 回だけ登録する
6. `mujou.run` と `__init__.py` の公開名
7. テスト: `tests/mujou/test_node.py`・`test_port.py`（recv/send）・`test_network.py`（ホスト入出力・状態・例外・E2E）。走らせたネットワークは fixture で必ず `shutdown` する
8. ドキュメント: `docs/torch.md`（`_on_thread_start`）、`docs/network.md`（確定した名前、run の戻るタイミング）、`docs/execution-model.md`（実装の節）、`docs/open-decisions.md` と `memory/project_open_decisions.md`（決まった項目を消す）、`memory/project_host_io.md`

## テスト観点

観点ごとに「どの WP のどのファイル」を添えた。Rust の単体テストは `#[cfg(test)] mod tests`。

### WP-A スロット

- 正常系（`test_core_slot.py`）: 10 種の dtype すべてで、torch の値を書いて `torch.from_dlpack(...).clone()` で読むとビット単位で一致する（`torch.equal`。bf16・bool を含む）。0 次元 `()`、TypedDict 相当の複数フィールドもキー順で往復する
- 正常系: 転置・スライスした非連続の Tensor、`storage_offset` が 0 でない Tensor を書いても、`contiguous()` したものと一致する
- コピーの独立性: 書いた後に元の Tensor をその場で書き換えても、読んだ値は変わらない。読んで得た Tensor を書き換えても、次の read の値も別の読み手の値も変わらない
- 不変性: Message を 1 つ保持したまま同じ Outlet に 100 回別の値を書いても、保持した Message から作った Tensor は最初の値のまま（プールに戻らない）
- 全員に送る: 外向きと全 Channel のスロットが同じ値・同じ時刻を返す（Rust では `Arc::ptr_eq`）。`send_to(k)` は Channel k だけを変え、外向きも他の Channel も変えない
- 時刻: 仮想時計で `advance` してから書くと、時刻はその時点の `time_ns` に等しい。`InputGroup.write` で書いた全スロットの時刻は等しい
- 版番号の一斉切り替え（Rust、スレッド 2 本、10^5 回）: 書き手が k 回目に 2 つの InputSlot へ値 k を書き、読み手が A → B の順に読むと常に `b >= a`、B → A の順なら常に `a >= b`（どの瞬間にも新旧が混ざらないことの、観測できる形）
- 書き手 1 つ（Rust、スレッド 4 本で読み 1 本で書く）: 読み手が見る値の列は単調で、書きかけの値（フィールド間で k が食い違う）を一度も見ない
- プールの再利用（Rust）: 読み手が Message を手放す速さで書き続けると、確保の回数が一定で頭打ちになる
- `wait_newer`: 書かれるまで戻らず（タイムアウトで `None`）、書かれたら戻る。`last` と同じ値なら待つ。待ち手が無いときの書き込みは lock を取らない（Rust で確認）
- 異常系: shape・dtype・フィールド数の不一致は `ValueError` で、メッセージにスロット名とフィールド名を含む（substring）。`InputGroup.write` の 1 つが不正なら、どのスロットも変わらない。`__dlpack__` を持たないオブジェクトは `TypeError`。デバイスが `kDLCUDA` の DLTensor は拒否（Rust で DLTensor を手で組んで確かめる）
- エッジケース: `channels=0` の Outlet、要素数 1 の Tensor、`restore` した値と時刻がそのまま読める

### WP-B ランタイム

- 起動の順序（`test_core_runtime.py`）: 各ノードで `_on_thread_start` → `on_start` →（states があれば）`load_state_dict` → `on_update` の順。`load_state_dict` は渡した状態を受け取る。hook はすべて同じノードでは同じスレッド（`threading.get_ident()` が一致、`state_dict` を除く）
- on_start の時間が最初の Δt に入らない: 実時計・スケール 1 で、`on_start` が実時間で 0.3 秒眠っても、最初の `dt` は 0.25 未満（上界だけを見るので負荷で落ちない）
- Δt の和: 仮想時計で起動時刻 T0 から自由実行し、ホストが `advance` を何回か行ってから shutdown すると、各ノードの `dt` の和は `(last_update_ns - T0) / 1e9` と一致し（相対 1e-12）、`on_update` の回数は `advance` の回数以下（Δt=0 で回らない＝ビジーループしない）
- Δt=0 の待ち: 仮想時計を進めないまま 0.1 秒（実時間）待っても `on_update` は 0 回。そのまま `shutdown` が 1 秒以内に戻り、`capture` も戻る（待ちが制御のイベントで起きる）
- 停止: `capture` の `build` の中で読んだ全ノードの `last_update_ns` は、そのときのタイマーの時刻 T 以下。`build` の中ではタイマーが停止中。`build` の後はタイマーが動いている。各ノードで `on_pause` → `state_dict` → `on_resume` の順
- 停止のタイムアウト: `on_update` で実時間 1 秒眠るノードがあると、`capture(timeout=0.1)` は `PauseTimeoutError`（メッセージにそのノードの名前）。その後もシステムは動き続け、タイマーは停止していない
- 手動モード: `step()` を n 回呼び、毎回その前に `advance(h)` すると、各ノードは n 回、`dt == h` で呼ばれる。`order` で与えた順に呼ばれる（共有のリストに記録して確かめる）。`advance` しないで `step()` すると `on_update` は呼ばれずに戻る。自由実行で `step()` は `RuntimeError`
- 例外の順序: 手動モードでノード X の `on_update` が 3 ステップ目に投げると、`step()` が `NetworkError` を送出し、`__cause__` は X の投げた例外そのもの（`is`）。全ノードで `on_exception` がちょうど 1 回、`on_stop` がちょうど 1 回、`on_exception` が `on_stop` より前、`on_exception` の後に `on_update` は無い。以後の `step`・`capture`・`shutdown`・`join`・`check` も `NetworkError`
- 例外の種類: `on_start`・`on_pause`・`on_resume`・`load_state_dict`・（通常の取得での）`state_dict` が投げても同じく全体が終了する。`on_start` の例外では `start()` が終了処理を済ませてから `NetworkError` を送出し、`on_update` は一度も呼ばれない
- 例外のときの取得: `on_exception_capture` を渡すと、例外を投げたノードを含む全ノードの状態と、失敗したノードの番号・hook 名・例外を受け取り、その戻り値を `exception_state()` で受け取れる。取得中に `state_dict` が投げたノードは dict から欠け、ログに出て、終了は続く。取得したときのタイマーの時刻は全ノードの `last_update_ns` 以上
- 2 つ目の例外: 2 つのノードがほぼ同時に投げても、`__cause__` は最初に記録した 1 つだけで、もう 1 つはログ（`caplog`）に出る
- 終了: `on_stop` が投げても他のノードの `on_stop` は呼ばれ、`shutdown()` は例外を送出しない（ログに出る）。停止中に終了を求めても `on_resume` を呼ばずに `on_stop` まで進む
- GIL を持たずに待つ: ホストが `join` で待つ間も、別の Python スレッドが進める（GIL ありのビルドでも `advance` できる）
- Rust 単体: 凍結の理由が 2 つ重なったとき、片方を下ろしてもタイマーは止まったまま。`wait_until_ns_or` は `interrupt_waiters` で起き、目標の時刻でも起きる

### WP-C 構造

- `TensorSpec`: dtype を省略すると作った時点の既定の dtype になり、後で `torch.set_default_dtype` を変えても変わらない。`zeros` は全 dtype で CPU の 0。`rand` は浮動小数点だけで、同じ seed・同じ generator の状態なら同じ値、値は [0, 1)。表にない dtype（`complex64` など）は `TypeError`
- `DictSpec`: キーの過不足は `ValueError`。値の型が `torch.Tensor` でない、入れ子、`NotRequired` は `TypeError`。キーの順は TypedDict の定義順
- `split` / `join`: 往復で元に戻る。shape・dtype の不一致はメッセージに `where` とキー名を含む
- `connect`: send/recv が逆向きに一致すれば Channel が両端の `channels` に接続順で入る。一致しない（shape・dtype・メッセージの型・TypedDict の型の違い）と `TypeError`。片方向（send だけ ↔ recv だけ）は接続できる。相手 1 つの Port の 2 本目は `ValueError`。`NextPort` は既定で複数を取れる
- `a.connect(b)` は `a` の NextPort と `b` の PrevPort をつなぐ。入れ子の Network が公開した Port でも同じ。PrevPort/NextPort が無い・2 つあると `ValueError`
- `build`: パスが `body.l1` の形で、深さ優先・定義順に並ぶ。検査の各項目（同じノードが 2 か所、木の外へ接続、ノードが Component を属性に持つ、ルートが Node、2 回目の build、build 後の connect）がそれぞれのエラーになる

### WP-D 結合と E2E

- 離散アルゴリズムとの一致（数値的オラクル）: ノード A（`x_A ← x_A + dt·(−x_A + u)`、u はホストの入力）と B（`x_B ← x_B + dt·(−x_B + x_A)`、A から受け取る）を float64 でつなぎ、手動モード・仮想時計・固定の h で n ステップ回すと、同じ式を Python のループで書いた参照とビット単位で一致する。順を `[A, B]` にすると B は同じステップの A を読み（Gauss–Seidel 型）、`[B, A]` にすると 1 ステップ前の A を読む（Jacobi 型）参照と一致する
- 停止中の一貫性: 自由実行中に `state_dict()` を取ると、木の全スロットの時刻と全 `last_update_ns` が `timer.time_ns` 以下
- 復元: 手動モードで k ステップ → `state_dict()` → 同じ構造の新しいインスタンスに `load_state_dict` → `run(_manual=True)` → m ステップ、の結果が、続けて k+m ステップ回した結果とビット単位で一致する（ノードの状態・スロット・タイマーの時刻のどれを木から落としても一致しなくなるので、3 つとも戻っていることを確かめられる）。木は `pickle` で往復できる
- コピーの独立性: `state_dict()` の後にノードが状態をその場で更新し続けても、受け取った木は変わらない
- 書き込み: run の前の `write` がそのスロットの初期値になり、最初の `on_update` が読む。`load_state_dict` の後の `write` が木の値より優先される。1 回の `write` で書いた複数の Port の値を、ノードは同じ時刻で読む。接続済みの Port・相手が複数の Port・受ける spec の無い Port・木の外の Port への `write` は `ValueError`。shape・dtype の不一致は `ValueError` / `TypeError`
- 読み出し: `read` は外向きスロットの値と秒の時刻（`time_ns / 1e9`）を返す。接続済みの Port からも読める（中間層の観測）。`send_to` だけを使う Port の `read` は初期値のまま（確認事項 2）。`subscribe` は書かれた値を新しい順に返し、途中の値は飛んでよい（返る値の時刻は単調非減少）。正常終了でイテレータが終わる
- recv: 未接続で相手 1 つなら初期値（ゼロ）、相手が複数なら `{}`、複数なら dict のキーは接続順の Channel。受け取った Tensor はノードのデバイスにあり、`requires_grad` は False
- `mujou.Node` の設定: `on_start` を override したサブクラスでも、`on_start` の中で `torch.is_grad_enabled()` が False、`generator` が seed から作られ（同じ seed なら同じ乱数）、`super().__init__()` を呼ばないサブクラスでも動く
- 単一の書き手: 所有ノードのスレッド以外（ホスト）から `port.send` すると `RuntimeError`
- 例外: ノードの例外の後、`write`・`read`・`state_dict`・`shutdown`・`join` と `subscribe` のイテレータが `NetworkError`（`__cause__` が元の例外）。`run(capture_state_on_exception=True)` なら `exception_state_dict()` が木を返し、`"exception"` に失敗したノードのパスが入る。既定では `None`
- `mujou.run`: 未 build のネットワークを build して走らせ、同じオブジェクトを返す。build 済みでも走る。2 回目の `run` は `RuntimeError`
- CUDA（`torch.cuda.is_available()` でなければ skip）: ノードのデバイスが CUDA のとき、recv がそのデバイスの Tensor を返し、send した値をホストが読める

## 想定リスク・トレードオフ

- GIL のあるビルド: ホストが Runtime の完了を待つ間に GIL を持っていると、ノードのスレッドと制御スレッドが Python を呼べずデッドロックする。待つ処理は必ず `py.detach` で包む（B のテストで GIL ありのビルドでも確かめる）
- インタプリタの終了: Rust のスレッドは `threading` の管理外なので、走ったまま終わると終了中に `Python::attach` する。D の `atexit` で全 Session を `shutdown` する。テストでも fixture で止める
- ノードの `on_update` の中で `mujou.time.sleep` を呼ぶと、停止中は戻らないので停止がタイムアウトする。目標周期と Δt の上限を今回は入れないので、自由実行の軽いノードはコアを 1 つ使い切る。どちらも docstring に書き、future work にする
- torch の DLPack: READ_ONLY を立てない（受け手は必ずコピーする約束で補う）。`__dlpack__(max_version=...)` を受け付けない producer には引数なしで呼び直す。bool の kDLBool を torch が受け付けるかは A の最初のテストで確かめる
- torch を 3.14t で import すると GIL が有効に戻る可能性がある。D の E2E で `sys._is_gil_enabled()` を記録し、戻るなら memory に残す（今回は止めない）
- `Arc<Message>` と `Vec<Buffer>` は書き込みごとに小さく確保する（大きなバイト列だけをプールする）。問題になったら Message もプールする
- Port の型付け（TypeVar の既定値と self 型の overload）が pyright で解けない可能性。代案（`recv()` / `recv_all()` に分ける）を C のステップ 6 に置いた
- `Network` のホスト入出力のメソッド名（`read`・`write` など）を利用者のサブクラスが属性に使うと壊れる。docs どおりの名前を採り、衝突は利用者の責任とする
- 状態の木で `state_dict` を呼ぶのは制御スレッド（ノードのスレッドではない）。停止中なので競合はなく、CUDA のノードは各ステップの終わりに自分のストリームを同期している。ノードのスレッドに呼ばせる案は、停止中のスレッドへ仕事を渡す仕組みが増えるので採らなかった
- 正常な終了と例外の終了処理は、止まらないノードがあると無期限に待つ（停止のタイムアウトは状態の取得にだけ効く）。ホストは Ctrl-C で抜けられる
- 採らなかった案: 制御スレッドを持たずホストのスレッドが lock の下で制御する案（例外の終了処理を誰が進めるかが決まらない）、`run` の前の `state_dict()` で hook を直接呼ぶ案（`on_start` 前の状態は未確保）

## future work（今回入れない）

PC-ALM（`mujou.pc`）、タスク実行クラスのワーカープールとネイティブのノード、ニューロン単位のノード、全値の記録、目標周期と Δt の上限、実行の配置の公開 API、list/dict に入れた子ノードの走査（今回は直接の属性だけ）、CUDA の pinned memory、タイマーの読み取りの Mutex の置き換え。

## 確認事項

### 要判断（成果物が変わる 3 件）

1. `run()` が戻るタイミング: docs は「すぐに戻る」。**推奨: 全ノードの `on_start`（と復元）が終わり、タイマーを再開してから戻る。** `on_start` の例外を `run()` 自身が `NetworkError` で送出でき、仮想時計のテストで「起動中はタイマーが停止していて `advance` できない」問題も起きない。代案（すぐ戻る）なら、起動の完了を待つ private な API が別に要る
2. `state_dict()` を呼べるとき: docs は「いつでも」。**推奨: 実行中だけ。** `run` の前は `on_start` が状態を確保していないので `RuntimeError`、正常終了の後も `RuntimeError`（取りたければ終了前に取る。memory の方針どおり）、例外で終了した後は `NetworkError`（木は `exception_state_dict()`）
3. torch の入れ方: `uv add torch` は Linux では CUDA 版の wheel（数 GB）を引く。**推奨: `[[tool.uv.index]]` に PyTorch の CPU 版の index を `explicit = true` で置き、Linux だけそこから取る**（CI と dev container が軽くなる。GPU のテストはローカルで CUDA 版を入れて回す）。CUDA 版のままにするかはユーザーの判断

### docs の未決定事項に置いた推奨案（一括で可否を確認したい）

4. 例外で終了した後のタイマー: 終了処理の最後に、ランタイムが凍結したタイマーを再開する（ホストの `mujou.time.sleep` が永久に戻らなくなるのを避ける）
5. 停止のタイムアウトの既定値: 実時間 10 秒。`run(pause_timeout=...)` で変える
6. スロットの識別名: `"{ノードのパス}.{Port の属性名}/out"`・`"/ch{k}"`・`"/in"`
7. 例外のときの木: 設定は `run(capture_state_on_exception=False)`、受け取る API は `Network.exception_state_dict()`
8. 相手を指定して送る Port を外から読んだとき: 外向きスロットは「全員に送る」ときだけ更新するので、`send_to` しか使わない Port の `read` は初期値を返す
9. `on_start` の override で設定が漏れない仕組み: フレームワークが `on_start` の前に `_on_thread_start()` を同じスレッドで呼ぶ。`docs/torch.md` の「`on_start` で設定する」を書き換える
10. トップレベルの公開名: `Node, Network, Port, PrevPort, NextPort, Channel, TensorSpec, DictSpec, NetworkError, PauseTimeoutError, run`（`mujou.time` は既存のまま）
11. torch の内部スレッド数: mujou は設定しない（docs どおりプロセス全体の設定なので、ホストが実行前に決める）。docs に推奨値の目安だけ書く
12. 相手 1 つの Port への 2 本目の接続: docs は「構築時にエラー」だが、`connect` の時点でエラーにする（型の検査と同じく早く失敗させるため。意味は変わらない）

## 参照

- docs: `docs/execution-model.md`（停止・復元・例外の流れ）、`docs/messages.md`（スロット・バッファプール・DLPack）、`docs/network.md`（Port・ホスト入出力・版番号）、`docs/nodes.md`（hook）、`docs/torch.md`（`mujou.Node`）、`docs/open-decisions.md`（グローバルタイマーとスケジューラの境界）
- 既存実装: `crates/mujou-core/src/time.rs:245`（`wait_until_ns`。Δt=0 の待ちの元）、`crates/mujou-python/src/lib.rs:317`（GIL を外して 50 ms ごとにシグナルを見る `sleep`。待つ処理はすべてこれに倣う）、`tests/mujou/conftest.py`（`virtual_timer`）
- skill: `locality-and-async`、`testing-strategy`、`refactor-conventions`
- pamiq-core: `thread/thread_control.py`（制御は 1 本、再開を先に set してから終了を set）
