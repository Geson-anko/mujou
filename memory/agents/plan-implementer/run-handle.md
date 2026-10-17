# run-handle: 実行の API の作り直し

ブランチ `feature/2026-10-16/run-handle`（`refactor/2026-10-16/core-package` の上）。skill `solo-dev-cycle` で 1 人で回す。

決定の出典: `memory/project_runtime_api_revision.md`、`memory/project_testing_api.md`、`memory/project_timer_per_run.md`。

## 段階 1: 計画

### 公開 API（シグネチャ）

`mujou`（トップレベル）:

```python
class Network(Component):
    def connect(self, other: Component) -> Channel: ...          # 変更なし
    def build(self) -> Graph: ...                                  # 何度呼んでも同じ Graph
    # run / write / read / subscribe / state_dict / load_state_dict / shutdown / join / _step は削除

class Compose(Network):
    def __init__(self, *networks: Network, **named: Network) -> None: ...

@final
class Graph:                     # build が作る。直接は作らない（__init__ は TypeError）
    @property
    def nodes(self) -> tuple[tuple[str, Node], ...]: ...
    def port_path(self, port: Port[Any, Any, Any]) -> str: ...

def run[P: Port[Any, Any, Any]](
    *networks: Network,
    timer: Timer | None = None,
    inputs: Mapping[P, torch.Tensor | Mapping[str, torch.Tensor]] | None = None,
    state: Mapping[str, Any] | None = None,
    pause_timeout: float = 30.0,
) -> Run: ...

class Run:                       # mujou.run が作る。直接は作らない（__init__ は TypeError）
    def write[P: Port[Any, Any, Any]](
        self, values: Mapping[P, torch.Tensor | Mapping[str, torch.Tensor]]
    ) -> None: ...
    def read[S](self, port: Port[S, Any, Any]) -> tuple[S, Timestamp]: ...
    def subscribe[S](self, port: Port[S, Any, Any]) -> Iterator[tuple[S, Timestamp]]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def shutdown(self, timeout: float = math.inf) -> None: ...
    def join(self, timeout: float = math.inf) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc_info) -> None: ...                     # shutdown()

class Node:
    @property
    def timer(self) -> Timer: ...   # run の間だけ。それ以外は RuntimeError

class Port:
    def recv_with_time(self) -> tuple[R, Timestamp] | dict[Channel, tuple[R, Timestamp]]: ...
```

`mujou.time`（モジュール関数は変更なし）:

```python
class Timer:
    def __init__(self) -> None: ...                # 時刻 0、実時計、スケール 1
    def now(self) -> Timestamp: ...
    def sleep(self, seconds: float) -> None: ...
    def get_time_scale(self) -> float: ...
    def set_time_scale(self, scale: float) -> None: ...

Timestamp = mujou._core._native.Timestamp       # float()/int() は秒、.ns はナノ秒
```

`mujou.testing`:

```python
class VirtualTimer(Timer):
    def __init__(self) -> None: ...                # 時刻 0 の仮想時計
    def advance(self, seconds: float) -> None: ...

def run[P](*networks, timer=None, inputs=None, state=None, pause_timeout=30.0) -> ManualRun: ...

class ManualRun(Run):
    def step(self, *nodes: Node) -> None: ...      # 引数なしで木の順に全ノード
```

`mujou.__all__` に `Compose`・`Graph`・`Run` を足す。`Timer`・`Timestamp` は `mujou.time` に置く（時間の名前は `mujou.time` に集める。トップレベルの `mujou.time` は star import の対象外なので `time` の衝突も起きない）。`mujou.testing` は `import mujou` で読み込まない（テストのときだけ `import mujou.testing`）。

### 名前の判断

- **ハンドルは `Run`。** 仕様書・docstring・利用者向け文書がすでに「1 つの実行（a run）」「実行ごとのタイマー（each run its own timer）」と呼んでいるので、その語をそのまま型の名前にした。`Handle` / `RunHandle` は何のハンドルかを言わず、`Session` は内部の型の名前で、`Execution` は文書の語と合わない。関数 `mujou.run` と大文字小文字だけ違うが、`threading.Thread` / `thread` のように「動詞の関数が名詞の型を返す」形として読める
- **テストの実行は `mujou.testing.run` と `ManualRun`。** 引数は `mujou.run` と同じで、違いは手でステップすることだけなので、モジュールで区別して同じ名前にした。仮想時計と手動のステップは独立に選べる（実時計の手動実行は、何も更新しないネットワークを作るのに使っている）ので、`VirtualTimer` を渡したら手動になる、といった結び付けはしない
- **`step()` は時計を進めない。** 時計は `timer.advance(dt)` で進め、`step()` は進んだ分で各ノードを 1 回更新する。$\Delta t$ を明示するとオラクルが読みやすい
- **`Timer` は native の `Timer` を包む Python のクラス。** native の `Timer` をそのまま公開すると、`load_state_dict`（走っているタイマーの時刻を飛ばす）・`use_virtual_clock`・`pause` / `resume` / `is_paused`（テストとフレームワーク用）まで見える。利用者が使うものだけ（`now` / `sleep` / スケール）を見せる
- **`Graph` は `nodes` と `port_path` だけ公開。** port → (ノードの番号, 属性名) の表は内部用なので private にする
- **`Node.timer` は on_start の前から、ホストが終了を見るまで（`shutdown` / `join` などが戻るまで）読める。** それ以外（run の前、正常な終了の後の `state_dict` の中を含む）は `RuntimeError`
- **`with` を抜けるとき:** 本体が例外なく終わったら `shutdown()`（`NetworkError` はそのまま送出）。本体が例外で抜けたら `shutdown()` の `NetworkError` は握りつぶし、本体の例外を優先する（ノードの例外はログに出ている）

### 振る舞い

- `run` は `networks` が 1 つならそのネットワーク、複数なら `Compose(*networks)` を `build()` して走らせる
- スロット（`Outlet` / `InputGroup`）とランタイムは、`run` の時点でその実行のタイマー（`timer` の native、`None` ならグローバル）で作る。`Session` が native の `Timer` を 1 つ持ち、全部に同じものを渡す。停止の時刻の記録も同じタイマーから読む
- 順序: スロットを作る → `state` を戻す → `inputs` を書く（`state` があれば木の時刻 T、無ければ書いた時刻）→ ノードに timer を付ける → 起動
- 1 つの Graph は 1 回だけ走る（2 回目は `RuntimeError`）。何も起動せずに `ValueError` で断られた run は数えない
- `Run` は起動が終わってから作られるので、起動中に別スレッドから `write` する経路は無くなる（`run_lock` で守っていた分岐を消す）
- `Compose` の名前: 位置は `"0"`, `"1"`, ...、キーワードはその名前。`_` で始まる名前と `Network` の属性名（`build` / `connect`）は `ValueError`、`Network` でない値は `TypeError`

### テスト観点

- `build()` が同じ Graph を返す／Graph は直接作れない
- `Compose` のパス（位置・キーワード・入れ子）、名前と型の検査、`mujou.run(a, b)` が `Compose(a, b)` と同じパス、1 つなら包まない
- 実行ごとのタイマー: スロットとランタイムがその timer で時刻を付ける（`recv_with_time` / `read` の時刻 = その VirtualTimer の時刻、グローバルタイマーは動かない）、2 つの実行が別のタイマーで独立に止まる（片方の state_dict がもう片方の時間を止めない）
- `inputs` が初期値になる／`state` の後の `inputs` は木の時刻 T
- `Node.timer`: run の前は RuntimeError、on_start と on_update で実行のタイマー、終了後は RuntimeError
- `Timestamp` を返す（`read` / `subscribe` / `recv_with_time`）
- `with` 文: 抜けると止まる、本体の例外が優先される
- `ManualRun.step` が離散アルゴリズムと一致（既存のオラクル）、`VirtualTimer.advance`
- 既存のテスト（書き込み・読み出し・購読・状態・例外・終了・インタプリタの終了・CUDA）を新しい口に書き換える

### リスク

- 並列の worktree が `node.py` と Rust の hook 名を変えている。`node.py` は `timer` の追加だけ
- `slot.rs` は触らない

## 段階 2 以降の記録

（下に追記）

## 段階 2: テスト実装

- `tests/mujou/conftest.py`: fixture を `virtual`（`VirtualTimer`）・`start_manual`（`mujou.testing.run`、既定の timer は `virtual`）・`start`（`mujou.run`）・`started`（終了時に shutdown）に置き換えた。グローバルタイマーを private な `use_virtual_clock` で切り替える `virtual_timer` は、グローバルタイマーそのもののテスト（`test_time.py`・`test_native.py`）にだけ残した。公開の口では切り替えられないため
- `test_network.py` は構築と `Compose` だけにし、実行のテストは `test_run.py`（`_core/run.py` のミラー）へ移した。`test_timer.py`（`Timer` / `VirtualTimer`）、`test_testing.py`（公開名と、公開のシグネチャに `_` 付きの引数が無いこと）を足した
- 新しい API で表せなくなったので消したテスト: 起動中に別スレッドから `write` する（ハンドルは起動の後に返る）、`run` の前の `read` / `state_dict` / `load_state_dict`、`load_state_dict` の前に書いた値が木で置き換わる（順序が state → inputs に固定）、壊れた木で何も変わらない（失敗した run のスロットは捨てられる）
- red の確認: `mujou.testing` が無く conftest の import で失敗

## 段階 3: 機能実装

- `src/mujou/_core/timer.py`（`Timer` / `VirtualTimer` / `GLOBAL` / `native`）、`src/mujou/_core/run.py`（`Run` / `ManualRun` / `run` / `start`）、`src/mujou/testing.py`、`Compose`・`Graph`・`start_once` を `network.py` に
- `Session(graph, timer)` は `mujou.run` の中で、Graph がまだ走っていないことを確かめてから作る（Port を結び付け直すので、走っている実行の Port を奪わないため）
- `VirtualTimer()` は `use_virtual_clock(True)` の後に `load_state_dict({"time_ns": 0, ...})` で 0 にそろえる（切り替えまでに実時計が数百 ns 進むため）
- 計画外: `Graph` は frozen dataclass をやめた（MappingProxyType を含むので hash できず、走ったかどうかのフラグと lock も持つため）
- 発見: native の `Timestamp` は pickle も deepcopy もできない。ノードの `state_dict` に入れると状態の取得が `TypeError` になる。テストでは ns の int に直した。確認事項に上げる

## 段階 4: 自己レビュー

- diff を通読。`Session` の `_run_lock` と、復元後・起動前の `write` の分岐は、ハンドルが起動後にしか返らないので不要になり消した
- `mujou.run` が node の hook から呼ばれたら `RuntimeError`（`start` の先頭で確かめる）
- `with` を本体の例外で抜けるときは `shutdown()` の `NetworkError` を捨てる。本体が `NetworkError` を受けて抜けた場合に二重に出さないため
- 整理点: `Session.start` の `**(restored or {})` は `Runtime.start` の引数名と `_restore` の返り値のキーを一致させている。読み直したが、これ以上の整理点なし

## 段階 5: ドキュメント

- `specs/network.md`（実行・書き込み・複数のネットワーク・実装）、`specs/execution-model.md`（タイマーの公開、複数のネットワーク、復元、実装、テスト）、`specs/nodes.md`（`timer` の属性）、`specs/torch.md`（`run()` の参照）、`examples/README.md`
- `docs/en/api/network.md`（`Network` / `Compose` / `Graph` / `run` / `Run`）、`docs/en/api/time.md`（`Timer` / `Timestamp`）、`docs/en/api/testing.md`（`mujou.testing`）。本文は書いていない
- トップの README には API の記述が無いので変えていない
- `memory/project_timer_per_run.md` と `memory/project_testing_api.md` に実装済みを追記

## 確認事項（ユーザーへ）

1. ハンドルのクラス名 `Run`（推奨）。代案 `Execution` / `RunHandle`
2. `Timestamp` を pickle / deepcopy できるようにするか（推奨: Rust の bindings に `__reduce__` を足す。別タスク）
3. `Run.timer`（実行のタイマーをハンドルから読む property）を足すか（推奨: 今は足さない。`timer=` に渡したものを持っていれば足りる）
4. `mujou.testing.run` の `timer` の既定をグローバルタイマーのままにするか（推奨: そのまま。`mujou.run` と同じ意味に保つ）

## レビュー対応（2026-10-17）

レビュー: `memory/agents/code-reviewer/run-handle.md`・`memory/agents/algorithm-reviewer/run-handle.md`（どちらも request-changes）。ユーザーの決定（ハンドル名 `Run` で確定、`Node.timer` は読み取り専用の view、hook から時間を動かす操作は `RuntimeError`）と orchestrator の裁定に従って直した。

- **algorithm must-fix（`Node.timer` がタイマーの制御を渡していた）:** `Node.timer` の型を `mujou.time.Clock`（`now()` だけの読み取り専用の view。直接は作れない）にした。名前は「時刻を読むもの」の意で、ホストが動かす `Timer` と分けた。`Timer.sleep` / `set_time_scale`、`VirtualTimer.advance`、`mujou.time.sleep` / `set_time_scale` は hook の中で `RuntimeError`（`timer.host_only`。`node.in_hook` は循環 import を避けて関数の中で import）。再現 (a) スケールの変更、(b) 仮想時計の前進（後のノードの dt が H のまま）、(c) タイマーで待つ（`step` も `state_dict` も戻る）を `tests/mujou/_core/test_timer.py` の `TestOnlyTheHostControlsTheTime` にした
- **M1:** `mujou.run(a, b)` が断られたら、`mujou.run` が作った `Compose` の構築を取り消す（`network.unbuild`、`port.thaw`）。1 つのネットワークの build は取り消さない（同じ Graph を返す約束のため）
- **M2:** `build` を module の lock の下で 1 つずつ行う。並行の `mujou.run` は同じ Graph を得て、`start_once` の Graph の lock で 1 回だけ走る
- **algorithm should-fix 2:** 構築した木のノードとネットワークを `WeakSet`（`_BUILT`）で印付けし、`build` で検査する。Port を持たないノードも 2 つの実行に入らない
- **S1 / nit:** `mujou.run()` と `Network` でない値は `TypeError`。`Compose` は空の名前と `.` を含む名前も `ValueError`
- **S4:** `Timestamp` に `__reduce__`（`copyreg.__newobj_ex__(Timestamp, (), {"ns": ...})`）を足し、`just stub` で stub を更新。テストの `.ns` の回避を外した
- **S5:** 正常に終了した後の `state_dict` の中では、ノードの `timer` が状態の時刻で止まった `Clock` になる（`stopped_clock`）。終わったら外す
- **S2 / S3 / S6 / nit:** docstring（`ManualRun.step` の `RuntimeError`、`mujou.run` の Raises、`timer` が予約名、`Graph` は `mujou.run` に渡せないこと、`Run.read` の `Timestamp` の軸、`mujou.testing.run` で `VirtualTimer` を強く勧める）
- **KeyboardInterrupt:** `start_once` は `Exception` 以外の `BaseException` でも Graph を走ったものとして数える（起動はバックグラウンドで続き、インタプリタの終了時に止まる）
- **algorithm should-fix 1:** `test_manual_steps_match_the_discrete_algorithm` を `Compose` でもパラメタライズ
- **S8:** node-state 側が書き直すので対応しない。`Node.timer` の docstring と `specs/nodes.md` に「公開の `on_thread_start` でも読める」を書いた
- specs: `nodes.md`（view・予約名・終了後の時刻）、`network.md`（構築の lock と印、断られた Compose、`TypeError`、名前の検査、`Timestamp` を状態に入れてよいこと、「グローバル時間」を「その実行のタイマー」に。見出しを「入力は実行の時間上の信号」に変え、リンクを直した）、`execution-model.md`（`Clock`、ホストだけが時間を動かす理由、`Timestamp` の pickle）
