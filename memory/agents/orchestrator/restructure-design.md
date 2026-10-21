# mujou の理想像

この文書は 2 段階でできています。まず、8 つの視点の提案とその相互批評を今回の基準で裁定し、1 つの設計にまとめました。その設計に 3 方向から反証が届いたので、それを検証して改訂しました。反証の 3 方向は、初めて使う研究者の目、不変の原則と PC-ALM の数理、実装の完全性です。改訂では判断基準に 1 つの視点を加えました。Linux カーネルのメンテナーが書くコードの「良いセンス」です。反証に答えるために特別扱いの分岐を足すことはせず、データの持ち方を変えて分岐ごと消す方を選んでいます。主な論点はコードを読んで確かめました（10 節の冒頭に、確かめた事実を挙げます）。コードは変更していません。各指摘への対応は、末尾の付録に一覧にしました。

---

## 1. 理念と判断基準（今回の基準をどう解釈したか）

### 1.1 判断基準

ユーザーが出した基準を、次の 8 つの判断規則として読みました。規則どうしが食い違うときは、上にある規則を優先します。

1. **不変の原則は動かさない。** 次のものは前提として扱い、裁定の対象にしません。
   - ノード間の局所性と非同期性（ノードをまたぐ `.backward()` をしない、グローバルなバリアを置かない、学習モードと推論モードを固定しない、エポックを前提にしない）
   - 1 つの実行に時間軸は 1 本
   - コアは Rust
   - 利用者が扱う値は `torch.Tensor` だけで、DLPack では必ずコピーする
   - 更新手法を差し替えられること
   - 理論から導いたオラクルで検証すること
2. **1 つの事実の置き場所は 1 つにする。** 次のものを、木が壊れている印とみなします。
   - 同じ事実を複数の場所で持ち、分岐で照らし合わせていること。たとえば「build 済み」を 4 か所で持ち、属性の登録を `__dict__` と別のレジストリの 2 か所で持っている
   - 循環、遅延 import、`TYPE_CHECKING` で逃がした依存
   - 他のモジュールのグローバルを書き換えること
   - 他のクラスの `_` 属性に触ること
3. **特別扱いの分岐は、データ構造を変えて消す（good taste）。** `if` を足す前に、データの形を変えれば通常の場合に吸収できないかを考えます。この設計で使った例を挙げます。
   - 「まだ届いていない」は、時刻 `None` で表す
   - 終わった後に止まる時計は、`min(timer, end)` で頭打ちにする
   - 並べた複数の `Vec` は、レコードの `Vec` 1 つにする
   - 文字列のフック名は `enum` にする
   - 利用者のオブジェクト（Node、Port、Network）は宣言だけを持つ。実行時の事実は実行側（Driver、Wire、Run）が持つ。凍結、解除、差し直し、巻き戻しが要らなくなる
   - インスタンスの `__dict__` を唯一のレジストリにする
   - 接続は「相手 → Channel」の dict で持つ。2 回目の connect も自己接続も特別扱いしない
   - 復元した実行には、保存した時刻から始まる専用のタイマーを与える。ほかの実行の時計を巻き戻す場合が、そもそも起きない
   - Tensor は、名前 `""` のフィールドを 1 つだけ持つメッセージとして扱う
   - 「学習しない」は、学習の間隔を無限大にすることで表す
4. **抽象と間接層は、実在する使い手が 2 人以上いるときだけ作る。** 使い手が 1 人しかいないので消すもの: `Run`→`Session` の委譲、`pc.functional` の shim、`Learner` の基底、`RunGroup`、`GlobalTimer`、コアの carry（使い手は PC だけ）、`Latent`、enter のキャッシュ。
5. **利用者の道のりを最短にする。** 「やりたいことに、最も少ない名前で着けるか」を問います。1 つの性質（相手の数、時間の操作、テストの時計）を表す手段は 1 つにし、名前を見ればふるまいが分かるようにします。
6. **ガードは提供シナリオの中に限る。** 残すガードは「どの提供シナリオで、利用者のどんな誤りを、どこで知らせるか」を説明できるものだけにし、5b 節の台帳（K 番号）に載せます。台帳に無い検査は書きません。1 つの誤りは、最も早い 1 か所で 1 回だけ知らせます。
7. **明白なコードにする。**
   - 関数は 1 つのことだけをし、インデントは浅く、名前は短く正確にする
   - 境界（公開 API と `_native` の ABI）は少なくし、意味をはっきりさせる
   - 動かないものと中途半端なもの（後で消す暫定コード、片方の層だけの対応）は入れない。移行の各ステップも、それだけで完結させる
8. **後方互換は考えない。** 古い名前、移行の案内、deprecation は残しません。文書も同じで、古い状態や変更の履歴を残しません（履歴は git が持ちます）。

### 1.2 ガードの判定規則

1.1 の規則 6 は、次の 6 つの判定規則で運用します。

| 記号 | 規則 |
|---|---|
| G1 残す | 黙って誤った結果になる誤り、または原因から遠い場所で壊れる誤り。最も早い 1 か所で、利用者が書いた名前（`Scale.next`、`body.l1`）を使って知らせる |
| G2 消す | Python、torch、Rust が、利用者の書いたその行で分かる例外を自然に出すもの |
| G3 消す | 型注釈がすでに表している誤りを、実行時にもう一度確かめているもの（`reportUnnecessaryIsInstance` や `The annotation is not checked at runtime` の印が付いた検査） |
| G4 データ構造で消す | 内部どうしの呼び出し契約（並べたリストの長さ、二重の start など）は、表せない形にする。FFI の健全性に要るものだけを、内部エラーとして残す |
| G5 消す | 提供シナリオの外への備え |
| G6 状態ごと消す | 特別扱いの状態（build 済みで未実行、巻き戻し、`RunGroup`、ルートのパスの推測、凍結したポートなど） |

### 1.3 設計の柱（11 個）

1. 依存は DAG（有向非巡回グラフ）にし、`tests/test_layers.py` の 1 本で強制する。`mujou.pc` は公開 API だけに依存する。
2. 利用者のオブジェクト（Node、Port、Network）は宣言だけを持つ。インスタンスの `__dict__` が唯一のレジストリで、実行時の事実は Driver、Wire、Run が持つ。
3. 構築は、状態を変えない純関数にする。起動は 3 段（検査 → native の準備 → 印と start）。段 2 までに断られた run は、何も変えていない。
4. 相手の数はクラスで表す（`Port` と `MultiPort`）。接続は「相手 → Channel」の dict で持つ。
5. コアの carry を廃止する。PC の目標は `Activity` の普通のフィールドとして運ぶ（ユーザーの確認が要る）。
6. 時間軸は `RunClock` の 1 本にし、ランタイムの唯一の時刻源にする。公開名は `ReadOnlyTimer`。ホストが時間を操作する窓口は `Timer` のメソッドだけにする。復元した実行は、専用のタイマーで走る。
7. 1 回の実行の実体は 1 つにする（`Run` が `Session` を吸収する）。配線、状態、変換は、それぞれ純粋なモジュールに置く。
8. `mujou.Node` は純 Python にする。Rust は Python 側の `Driver` を、hook 1 回につき 1 回だけ呼ぶ。
9. 境界は 1 か所ずつにする。
   - torch とコアの変換は `codec.py` だけ
   - shape と dtype の検査は Rust の `copy_from` だけ
   - 例外は Python で定義する
   - 「いま hook を走らせている Driver」は、bindings の thread-local が持つ
10. Rust のライフサイクルの状態機械は 1 本にする。
11. ガード台帳と、知識の置き場所の表を、それぞれ 1 つの正本にする。

---

## 2. 利用者が覚える概念（少数。学ぶ順に）

| 順 | 概念 | 1 行説明 |
|---|---|---|
| 1 | **ノード（`Node`）** | 計算の単位。`mujou.Node` を継承し、`on_update(dt)` に 1 ステップを書く。内側は同期で、通常の PyTorch と同じように計算する（勾配も記録する）。外とはポートだけで話す。状態は自分の属性に持つ |
| 2 | **ポート（`Port` / `MultiPort`）と `Channel`** | 接続点。`Port` は `recv()` が 1 つの値を返すので、受け取る相手は高々 1 つ（送るだけなら何本でも繋げる）。`MultiPort` は同じ種類の相手を束ね、`recv()` は `dict[Channel, 値]` を connect の順で返す。`connect` 1 回で `Channel` が 1 本できる。循環して繋いでもよい |
| 3 | **メッセージの型（spec）** | `TensorSpec` と `DictSpec` が shape と dtype を決める。値はいつも `torch.Tensor`（またはその dict）で、`send` はコピーする |
| 4 | **ネットワークとパス** | `Network` は、ノードを属性に並べて繋ぐ、組み立て専用の入れ物。`Group(**named)` は、名前を付けて並べるだけの入れ物。パスは、ルートから属性名をドットでつないだもの（`"body.l1"`）で、状態の木のキーになる |
| 5 | **実行（`mujou.run` / `Run`）** | 1 つのルートを 1 回だけ走らせる。ホストは `write`、`read`、`wait_for` と `run.timer` で操作する。終わった後は、ノードの属性を読んでよい |
| 6 | **時間（`Timer`）** | 1 つの実行に時間軸は 1 本。ホストは `run.timer` を操作し、ノードは hook の中で `self.timer` を読むだけ。`dt` は、前回の更新からの秒数（その実行のタイマーで測る）。timeout だけは実時間の秒 |
| 7 | **状態** | テンソルの属性と、`state_dict` と `load_state_dict` を持つ属性は、自動で状態に入る。乱数生成器の状態も入る。それ以外で実行中に変わる値（カウンタ、最後に見た時刻）は、`register_state` で加える。`run.state_dict()` で状態を取り、`mujou.run(..., state=)` で新しいインスタンスに戻す。戻した実行は、保存した時刻から始まる専用のタイマーで続く |
| 8 | **テスト** | `mujou.testing.run` は、仮想時計の上で決定的に回す。`run.step(dt, *nodes)` は、時間を `dt` だけ進めてから更新する |
| 9 | **学習** | 学習則はノードの中に書く。ノードの内側の BP（誤差逆伝播）、PC-ALM（`mujou.pc`）、自作の則のどれでもよい。ノードをまたぐ計算グラフは作れない（Port はコピーを送る） |

今は 25 個ほどある概念を、9 個に減らします。消える概念は次のとおりです。

- carry（`send_carry`、`recv_carry`、`carried`、`carry_source`、relay）
- `PrevPort` / `NextPort` / `Single` / `Multi` / `multi=`
- build と `Graph`
- `StateDict`
- `PauseTimeoutError`
- `mujou.time` のモジュール関数の系統
- `has_received`
- `RunGroup`
- 位置引数で並べた `Group`
- 代入時の登録（`super().__init__()` の前の代入が誤りになる規則）

---

## 3. 公開 API

### 3.1 `mujou`（13 名）

`import mujou` はサブモジュールを読み込みません。`mujou.time`、`mujou.testing`、`mujou.pc` は、どれも明示的に import します。

| 名前 | シグネチャ | 1 行説明 |
|---|---|---|
| `Node` | `Node(*, period: float = 0.01, seed: int \| None = None, device: torch.device \| str = "cpu", dedicated_thread: bool = False)` | ノードの基底。`__init__` で `super().__init__(...)` を呼ぶ（必須。呼ばないと run の段 1 で K2）。hook は、通常の PyTorch と同じく勾配を記録する |
| ├ hook | `on_start() -> None` / `on_update(dt: float) -> None` / `on_pause() -> None` / `on_resume() -> None` / `on_exception() -> None` / `on_stop() -> None` | 基底は何もしない。override したら最初に `super()` を呼ぶ（慣習）。呼ぶ順は `on_start` →（復元するなら）`load_state_dict` → `on_update` の繰り返し →（例外で終わるなら `on_exception`）→ `on_stop`。`on_start` では、状態に依存する出力をしない（復元はその後に入るため）。hook は呼ぶたびに属性から引くので、いつ差し替えても反映される |
| ├ 状態 | `state_dict() -> dict[str, Any]` / `load_state_dict(state: Mapping[str, Any]) -> None` | 既定の `state_dict` は、集めた状態と乱数生成器の状態を返す。既定の `load_state_dict` は、それを strict に戻す（K22）。override するときは、`super()` の戻り値を拡張する。`super().load_state_dict` には、自分で足したキーを除いて渡す |
| ├ `register_state` | `register_state(name: str) -> None` | 代入済みの属性を状態に加える（まだ代入していない名前は K30）。テンソルでも、`state_dict` を持つ物でもない値のうち、実行中に変わるもの（カウンタ、最後に見た時刻）に使う。値は `torch.load(weights_only=True)` で読める型にする（int、float、bool、str、None、それらの list / tuple / dict、テンソル） |
| └ 読み取り専用 property | `device: torch.device` / `period: float` / `dedicated_thread: bool` / `generator: torch.Generator` / `timer: mujou.time.ReadOnlyTimer` | `generator` は `seed` で初期化した、ノード専用の生成器。`seed` が効くのはこれだけなので、module の初期化にも `generator=self.generator` を渡す。`timer` は hook の中で読む（外で読むと K12） |
| `Port` | `Port[S = Never, R = Never](*, send: MessageSpec[S] \| None = None, recv: MessageSpec[R] \| None = None, initial: S \| None = None)` | 受け取る相手が高々 1 つの接続点。送るだけの Port は何本でも繋げる。`initial` は、この Port が最初に send するまで、相手に見える値 |
| ├ | `connect(other: Port[R, S] \| MultiPort[R, S]) -> Channel` | 繋いで、その 1 本を返す。同じ相手にもう一度繋ぐと、同じ Channel を返す |
| ├ | `recv() -> R` / `recv_with_time() -> tuple[R, Timestamp \| None]` | 時刻が `None` なら、まだ誰も送っていない（値は initial か zeros） |
| ├ | `send(msg: S) -> None` | 繋いだ全員に送る。値はコピーされるので、送った後に変更してよい |
| └ | `channels: tuple[Channel, ...]`（connect の順）/ `send_spec: MessageSpec[S] \| None` / `recv_spec: MessageSpec[R] \| None` | 読み取り専用 |
| `MultiPort` | `MultiPort[S = Never, R = Never](*, send=None, recv=None, initial=None)` | 同じ種類の相手を束ねる接続点。役割の違う入力（予測と誤差など）は、Port を分けて受ける |
| ├ | `recv() -> dict[Channel, R]` / `recv_with_time() -> dict[Channel, tuple[R, Timestamp \| None]]` | connect の順に並ぶ |
| └ | `send(msg: S) -> None` / `send_to(channel: Channel, msg: S) -> None` / `connect` / `channels` / `send_spec` / `recv_spec` | `send` は全員に送り、outbox（ホストが `read` で読む値）にも書く。`send_to` は 1 本だけに送り、outbox には書かない |
| `Channel` | `ends: tuple[Port \| MultiPort, Port \| MultiPort]` / `other(port) -> Port \| MultiPort` | `connect` が作る接続 1 本 |
| `MessageSpec` | `MessageSpec[T]`（抽象）: `zeros(*, device="cpu") -> T` / `empty(*, device="cpu") -> T` | spec の総称型。注釈と、総称的な関数に使う |
| `TensorSpec` | `TensorSpec(shape: Sequence[int], dtype: torch.dtype = torch.float32)`、`shape`、`dtype`、`rand(*, generator: torch.Generator \| None = None, device="cpu") -> Tensor`、クラス属性 `dtypes: frozenset[torch.dtype]` | 1 つのテンソルの shape と dtype |
| `DictSpec` | `DictSpec[TD](typed_dict: type[TD], /, **specs: TensorSpec)`、`specs` | TypedDict と、キーごとの `TensorSpec` |
| `Network` | `Network()` | 組み立ての基底。属性に代入した `Node` / `Network` を、代入した順に子とする（run が `__dict__` をそのまま読む）。`Port` を代入しても、ただの参照になる |
| `Group` | `Group(**children: Node \| Network)` | 名前を付けて並べるだけの `Network` |
| `run` | `run(root: Node \| Network, *, timer: Timer \| None = None, inputs: Mapping[Port[Any, Any], object] \| None = None, state: Mapping[str, Any] \| None = None, workers: int \| None = None) -> Run` | ルートを 1 つ受け取り、1 回だけ走らせる。`state` を渡すと、保存した時刻と time scale から始まる新しい `Timer` で走る（`timer` との同時指定は K23）。`inputs` は、起動の時刻に書いた値になる |
| `Run` | 下表 | ホストの窓口。`run()` が返す |
| `NetworkError` | `class NetworkError(RuntimeError)` | ノードの例外で実行が止まった。`__cause__` にノードの例外が入る。Python で定義する |
| `__version__` | `str` | `importlib.metadata.version("mujou")` から取る |

`Run` のメソッドは次のとおりです。timeout はどれも実時間の秒で、既定は `math.inf` にそろえます。

| メソッド | シグネチャ | 説明 |
|---|---|---|
| `timer` | `Run.timer -> Timer`（property） | その実行のタイマー。ホストはこれで待ち、止める |
| `write` | `write(values: Mapping[Port[Any, Any], object]) -> Timestamp` | 未接続で受け取る `Port` に書く。1 回の write の値は一緒に切り替わる |
| `read` | `read(port: Port[S, Any] \| MultiPort[S, Any]) -> S` | 送るポートの最新値（outbox。`MultiPort` なら最後の `send` の値） |
| `read_with_time` | `read_with_time(port) -> tuple[S, Timestamp \| None]` | 時刻付きで読む |
| `subscribe` | `subscribe(port, *, timeout: float = math.inf) -> Iterator[tuple[S, Timestamp \| None]]` | 新しい値を追いかける。実行が終わると止まる |
| `wait_for` | `wait_for(port, until: Callable[[S], bool], *, timeout: float = math.inf) -> tuple[S, Timestamp \| None]` | 条件を満たす値を待つ。満たす前に実行が終わると `RuntimeError` |
| `state_dict` | `state_dict(*, timeout: float = math.inf) -> dict[str, Any]` | 全ノードを止めて状態を取る。終了後は、終了時刻の状態を取る。テンソルは CPU のコピー。木の形の正本は、この docstring |
| `shutdown` / `join` | `shutdown(*, timeout: float = math.inf) -> None` / `join(*, timeout: float = math.inf) -> None` | 期限を過ぎたら `TimeoutError`（実行は止まり続ける） |
| `with` | `__enter__` / `__exit__` | 抜けるときに `shutdown` する |

あるノードが例外を出した後は、ホストの呼び出し（`write`、`read`、`read_with_time`、`subscribe`、`wait_for`、`state_dict`、`shutdown`、`join`、`step`）が `NetworkError` を出します。`with` を別の例外で抜けるときは、その例外の方を優先します。

### 3.2 `mujou.time`（5 名）

| 名前 | シグネチャ | 説明 |
|---|---|---|
| `Timer` | `Timer(*, start: Timestamp = Timestamp(0), time_scale: float = 1.0)`、`now() -> Timestamp` / `sleep(seconds: float) -> None` / `pause()` / `resume()` / `is_paused() -> bool` / `time_scale() -> float` / `set_time_scale(scale: float) -> None` | 時間軸。`start` から、実時間 × time scale の速さで進む。制御するメソッドは、hook の中から呼ぶと `RuntimeError`（K16） |
| `VirtualTimer` | `VirtualTimer(*, start: Timestamp = Timestamp(0))`、`advance(seconds: float) -> None` | `advance` でだけ進む `Timer` のサブクラス。テストとシミュレーションに使う。time scale は効かない。pause 中に `advance` すると `RuntimeError` |
| `ReadOnlyTimer` | `ReadOnlyTimer(timer: Timer)`、`now() -> Timestamp` / `time_scale() -> float` | timer を読むだけの見方。実行がノードに渡すもの（`Node.timer`）は、その実行の終わりで止まる |
| `Timestamp` | `Timestamp(ns: int)`、`ns: int`、`float(ts)`、`int(ts)`、`a - b -> float`（秒）、比較、hash、pickle | タイマー上の時刻。どのタイマーの時刻かは持たない |
| `global_timer` | `Timer` のインスタンス（プロセスの開始で 0） | 既定のタイマーそのもの。復元しない複数の実行を、同じ時刻で止めるときに使う |

時間の単位は次のとおりです。`topics/time.md` にも 1 つの表で載せます。

- 実時間の秒: timeout
- タイマーの秒: `sleep`、`advance`、`dt`、`period`、`pc.show(seconds=)`

### 3.3 `mujou.testing`（2 名）

| 名前 | シグネチャ | 説明 |
|---|---|---|
| `run` | `run(root: Node \| Network, *, timer: VirtualTimer \| None = None, inputs=None, state=None) -> ManualRun` | 手動で進める、決定的な実行。`timer` を省くと、新しい `VirtualTimer` を使う（`state` を渡したなら、保存した時刻から始まる）。`timer` を共有すると、`step` は共有しているほかの実行の時間も進める |
| `ManualRun` | `ManualRun(Run)`、`timer -> VirtualTimer`、`step(dt: float, *nodes: Node) -> None` | 下の箇条を参照 |

`ManualRun.step(dt, *nodes)` のふるまいは次のとおりです。

- timer を `dt` だけ進めてから、`nodes` を渡した順に 1 回ずつ更新する。`nodes` を省くと、全ノードを木の順（代入の順）に更新する
- period は無視する
- 各ノードの `dt` は、前回の更新から測る。最初は、開始の時刻か、復元した最後の更新の時刻から
- 同じ `step` に 2 回渡したノードは、2 回目の `dt` が 0 なので更新しない
- 木の順の更新は Gauss-Seidel になる。Jacobi の順が要るテストは、出力側から逆順に渡す
- ノードが例外を出すと、その場で `NetworkError` を出す

### 3.4 `mujou.pc`（15 名）

| 名前 | シグネチャ | 説明 |
|---|---|---|
| `Chain` | `Chain(*modules: nn.Module, x: TensorSpec, y: TensorSpec, loss: Callable[[Tensor, Tensor], Tensor], tau: TimeConstants, learn_interval: float \| None = None, max_dt: float \| None = None, device: torch.device \| str = "cpu")` | `Network`。`modules[:-1]` を `Layer`、`modules[-1]` を `Output` にして、`Input` の後ろに繋ぐ。子のパスは `"input"`、`"layer0"`、…、`"output"`（`modules[i]` が `layer{i}`）。period はすべて `tau.h / 4`。`prev`（`Sample` を受ける）と `next`（`Prediction` を送る）を公開する。`learn_interval` は、credit の往復（2 × ホップ数 × period）より長くすることを勧める（docstring に書く。検査はしない） |
| `show` | `show(run: Run, chain: Chain, sample: Sample, *, seconds: float) -> Prediction` | `sample` を書き、その書き込みに対する最初の予測（学習する前の、prequential な予測）を返す。返すのは、書き込みから `seconds` の間サンプルを提示し続けた後。評価（mask がすべて False）では、伝播に足りる短い `seconds` でよい。誰も時計を進めないので、`ManualRun` の上では返らない |
| `Input` | `Input(*, x: TensorSpec, y: TensorSpec, period: float, device="cpu")`、`prev: Port[Never, Sample]`、`next: MultiPort[Activity, Tensor]`、`act: DictSpec[Activity]` | ホストの `Sample` を `Activity` にする。書き込みの前は、`time` が 0 で `mask` がすべて False の Activity を送る |
| `Layer` | `Layer(module, *, act: DictSpec[Activity], tau, learn_interval=None, period=None, max_dt=None, device="cpu")`、`prev: Port[Tensor, Activity]`、`next: MultiPort[Activity, Tensor]`、`act` | 制約 h = f(h_in)。引数 `act` には前のノードの `act` を渡す。period の既定は `tau.h / 4`、max_dt の既定は `min(tau.h / 2, 3 * tau.lam)` |
| `Output` | `Output(module, *, act, loss, tau, learn_interval=None, period=None, max_dt=None, device="cpu")`、`prev: Port[Tensor, Activity]`、`next: Port[Prediction, Never]` | ŷ = f(h_in) と、サンプルごとの損失 |
| `TimeConstants` | `TimeConstants(*, h: float, lam: float, theta: float)` | 秒で表した時定数。`lam=inf` で PC に、`theta=inf` で学習なしになる。1 回の学習は、平均の勾配に対する学習率 Σdt/θ の SGD に当たる |
| `LearnClock` | `LearnClock(*, period: float, learn_interval: float \| None, max_dt: float)`、`tick(dt: float, time_ns: int, now_ns: int) -> Tick`、`discrete: bool`、`state_dict()` / `load_state_dict()` | 学習の時刻を決める部品。PC 以外の則でも使える。`learn_interval` は、`None` で毎回学習し、`inf` で学習しない。状態を持つので、属性に置くだけで状態に入る |
| `Tick` | NamedTuple `(dt: float, new: bool, learn: bool, elapsed: float)` | `dt` は max_dt で打ち切った dt。`new` は新しい入力か。`learn` が True なら、`elapsed`（前の学習からの Σdt）で重みを付けて学習する。`LearnClock` は、`learn` を返した時点で学習したものとして数え直す |
| `Sample` | TypedDict `{x, y, mask}` | ホストが書く |
| `Activity` | TypedDict `{h, jumps, y, mask, time}` | ノード間を流れる。`time` は `write` が返した時刻の ns（shape ()、int64）。`jumps` は、h が離散的に跳んだ回数（旧 `epoch`）。`y`、`mask`、`time` は経路に沿って中継する |
| `Prediction` | TypedDict `{y_hat, y, mask, time}` | `Output.next` がホストへ送る |
| `activity_spec` | `activity_spec(h: TensorSpec, y: TensorSpec) -> DictSpec[Activity]` | 自作のノードが pc のノードと繋がるための spec。h と y の先頭の次元（B）が違えば `ValueError`（K28） |
| `mse` / `cross_entropy` | `(y_hat: Tensor, y: Tensor) -> Tensor`（(B,) を返す） | サンプルごとの損失 |
| `functional` | `vjp(f, h_in, delta: Callable[[Tensor], Tensor], params) -> (mu, g_in, g_theta)` / `loss_vjp(f, h_in, y, mask, loss, params) -> (y_hat, g_in, g_theta)` / `relax_(h, lam, mu, g, dt, tau)` / `learn_(params, g_theta, elapsed, tau, batch)` | 状態を持たない数式。自作ノードの部品 |

### 3.5 現在からの差分

| 区分 | 現在 | 理想 | 理由（詳しくは 5 節） |
|---|---|---|---|
| 削除 | `PrevPort`、`NextPort`、`Port(multi=)`、`mujou.typing.Single` / `Multi`、型引数 M | `Port` / `MultiPort` | 相手の数を表す軸が 3 つあったのを、1 つにする（D6） |
| 削除 | `Graph`、`Network.build()` | — | `run` が検査を兼ねる。build は、凍結を伴う罠だった（D3） |
| 削除 | `StateDict` | 素の dict（CPU） | `.cpu()` と safe globals の細工が要らなくなる（D23） |
| 削除 | `PauseTimeoutError` | 組み込みの `TimeoutError` | 利用者が区別する場面が無い（D27） |
| 削除 | `mujou.typing` | `MessageSpec` をトップレベルへ移す | 中身が 1 名だけになる（D45） |
| 削除 | `mujou.time.now` / `sleep` / `pause` / `resume` / `is_paused` / `get_time_scale` / `set_time_scale` | `run.timer.*` と `global_timer.*` | 2 本目の道と、その落とし穴を消す（D12） |
| 削除 | `Port.has_received`、`Port.carried`、`send_carry=`、`recv_carry=`、`send(carry=)`、`carry_source`、`send_carry_spec`、`recv_carry_spec`、`Port.multi` | — | 時刻 `None`（D8）と、carry の廃止（D7） |
| 削除 | `mujou.run(a, b, ...)`、`Group(a, b)` の位置引数、`mujou.testing.run(*components)` | `run(root)`、`Group(**named)` | パスが、実行のしかたで変わらないようにする（D4） |
| 削除 | `pc.Carried`、`pc.carry_spec` | `Activity` のフィールド | carry の廃止（D7） |
| 削除 | `Timestamp.float()` | `float(ts)` | 読み方を 1 つにする（D47） |
| 削除 | `DictSpec.typed_dict` | — | 使い手は carry だけだった |
| 削除 | `TensorSpec.rand(seed=)`、`TensorSpec(dtype=None)` | `rand(*, generator=)`、`dtype=torch.float32` | 入口を 1 つにし、既定値を明示する（D53） |
| 削除 | 予約名の検査（クラス属性で `period` などを置き換えると `TypeError`） | — | フレームワークは公開の property だけを読むので、置き換えても食い違わない（D51） |
| 移動 | `mujou.testing.VirtualTimer` | `mujou.time.VirtualTimer` | 時間の名前を 1 か所に集める（D13） |
| 移動 | `mujou.typing.MessageSpec` | `mujou.MessageSpec` | |
| 改名 | `Timer.get_time_scale` | `Timer.time_scale()` | 全層の名前をそろえる（D31） |
| 改名 | `Activity.epoch` | `Activity.jumps` | データセットのエポックと衝突する（D37） |
| 変更 | `ManualRun.step(*nodes)` | `step(dt, *nodes)` | advance を忘れると、黙って何も起きなかった（D15） |
| 変更 | `recv_with_time` / `read_with_time` / `subscribe` / `wait_for` の時刻 `Timestamp` | `Timestamp \| None` | 「送られたか」と時刻を、1 回の読み出しから得る（D8） |
| 変更 | `Run.state_dict(timeout=30)` | `timeout=math.inf` | timeout の既定をそろえる（D28） |
| 変更 | `Node.state_dict() -> Any` | `-> dict[str, Any]`（乱数生成器の状態を含む） | 基底のキーとエラーの文言（パス.キー）のために、木の項目を dict にそろえる（D21、D23） |
| 変更 | 既定の `load_state_dict` が、未登録のテンソルから属性を作る | strict（K22） | 救済をやめ、規則を 1 つにする（D22） |
| 変更 | 代入時に登録する（`super().__init__()` の前の代入は `AttributeError`） | run の段 1 で `__dict__` を読む（super の前の代入も通る） | 1 つの事実を 2 か所で持たない（D49。確認が要る） |
| 変更 | `Port(send, recv)` を位置引数でも渡せる | キーワード専用 | 送りと受けの取り違えを、書いた行で見えるようにする |
| 変更 | `connect`: 自己接続は `ValueError`、`MultiPort` どうしを 2 回繋ぐと 2 本になる | 同じ相手にもう一度繋ぐと、同じ Channel を返す | 接続を「相手 → Channel」の dict で持つ（D6） |
| 変更 | `mujou.run(state=)` は、渡した timer か global_timer に時刻を読み込む | 保存した時刻から始まる新しい `Timer` で走る。`timer` との同時指定は K23 | 共有の時計を巻き戻す場合を無くす（D50。確認が要る） |
| 変更 | `Timer()`（0 から始まる） | `Timer(*, start=, time_scale=)`。`ReadOnlyTimer(timer)` も作れる | 復元の時刻を、生成のときに与える（D50、D11） |
| 変更 | `Node.timer` は、終わった後もホストから読める | hook の中で読む（外では K12） | ホストは `run.timer` を使う。読むのは、いま走っている Driver の時計（D11） |
| 変更 | `pc.Input(shape, carry=, period=)`、`Layer` / `Output(input_shape=, carry=, dtype=)` | `Input(*, x, y, period)`（period は必須）、`Layer` / `Output(module, *, act=, ...)`、`act` を公開 | B を 1 回だけ書けばよくなり、h と y の spec を 1 つのデータで運べる（D38） |
| 変更 | `functional.vjp(delta: Tensor \| Callable, wrt_input=)` | `delta: Callable` だけ | 使い手のいない柔軟性を消す（D40） |
| 変更 | `learn_interval` の docstring の「安定の条件」 | 推奨値（2 × ホップ数 × period より長く） | 実測と食い違う（D39） |
| 追加 | — | `MultiPort`、`TensorSpec.dtypes`、`mujou.time.global_timer`、`Timer(start=, time_scale=)`、`VirtualTimer(start=)`、`ReadOnlyTimer(timer)`、`Timestamp(ns)`（シグネチャを文書化する）、`pc.Chain`、`pc.show`、`pc.LearnClock`、`pc.Tick`、`pc.activity_spec`、`Input.act` / `Layer.act` | 利用者が今は手で写しているか、private な名前でしか取れないもの |

---

## 4. ファイルツリーと層

### 4.1 Python の層図（上の層は、下の層だけを import する）

```
L12  mujou/pc/*                 mujou の公開名だけ（Node, Port, MultiPort, Network, Group, TensorSpec, DictSpec, Run）
L11  mujou/__init__.py  mujou/time.py  mujou/testing.py      公開の入口（再エクスポートだけ）
L10  _core/run.py               Run, ManualRun, run(), testing の run の本体、走った印（K14）、atexit
L9   _core/state.py             状態の木 ⇄ 実行（capture / restore）
L8   _core/wiring.py            Tree → native のスロット、Wire、Driver
L7   _core/tree.py              walk: __dict__ を代入順に読んで木を作り、検査する（純関数）
L6   _core/driver.py            Driver: Rust が呼ぶ、ノード 1 つ分の入口
L5   _core/node.py  _core/network.py    利用者が継承する基底
L4   _core/component.py         Component: クラス定義時の検査（K1）だけを持つ共通の基底
L3   _core/port.py              Port, MultiPort, Channel, Wire、running()（いま hook を走らせている Driver を引く）
L2   _core/codec.py             torch ⇄ コアの唯一の変換
L1   _core/spec.py              MessageSpec, TensorSpec, DictSpec, DLPACK（dtype の対応表）
L0   _core/errors.py  _core/_native (Rust)    errors.py は葉。Rust が import して raise に使う
```

- 実行時に `_native` を import してよいのは、次の 6 つだけです。今は 12 モジュールが import しています。
  - `codec.py`（Schema、BufferView）
  - `port.py`（`current`）
  - `node.py`（`ReadOnlyTimer` の型）
  - `wiring.py`（Outlet、Slot、InputGroup、ReadOnlyTimer）
  - `run.py`（`start`、Timer、VirtualTimer、`global_timer`）
  - `mujou/time.py`（再エクスポート）
- 循環、遅延 import、`TYPE_CHECKING` で逃がした依存、クラスの外からの `_` 属性の読み書きは、どれも 0 にします（D51）。
- `tests/test_layers.py` が、この表（層の順序と `_native` の許可リスト）を AST で検査します。このテストが、層の規則の正本です。

### 4.2 `src/mujou/`

```
src/mujou/
  __init__.py          公開名 13 個の再エクスポートだけ。__version__ は importlib.metadata から取る
  time.py              _native の Timer / VirtualTimer / ReadOnlyTimer / Timestamp / global_timer を再エクスポートする
  testing.py           _core.run の run（手動版）と ManualRun を再エクスポートする
  py.typed
  pc/
    __init__.py        公開名 15 個の再エクスポート（docs と同じ順に並べる）
    functional.py      vjp / loss_vjp / relax_ / learn_ の実装そのもの（共通の骨格は _grad 1 つ。_rows は private）
    _messages.py       Sample / Activity / Prediction と activity_spec（B の照合 K28 はここだけ）
    _time_constants.py TimeConstants
    _losses.py         mse / cross_entropy
    _clock.py          LearnClock と Tick（learn_interval の格子、Σdt、max_dt での打ち切り。now_ns は引数で受け取る）
    _input.py          Input
    _layer.py          Layer（LearnClock を合成で持つ。on_update は段ごとの private メソッドに分ける）
    _output.py         Output（教師の有無の分岐は、loss_vjp の mask の処理に一本化する）
    _chain.py          Chain（act の受け渡し、connect、period の既定）と show
  _core/
    __init__.py        docstring だけ
    _native.*.so / _native.pyi   Rust 拡張と、生成したスタブ（手で編集しない）
    errors.py          NetworkError（何も import しない葉）
    spec.py            MessageSpec（抽象）, TensorSpec, DictSpec と、パッケージ内部の名前 DLPACK（dtype → DLPack の 3 つ組）
    codec.py           fields(spec)（Tensor を名前 "" のフィールド 1 つに正規化する唯一の isinstance）、
                       schema(spec)、to_core(spec, value, label)（構造の検査、detach、CPU へ移す）、
                       from_core(spec, views, device)（必ずコピーする）。spec の公開の属性だけを読む
    port.py            Port, MultiPort（宣言: spec、initial、相手 → Channel の dict）、Channel、
                       Wire（label、outlet、sources、device のただのデータと、recv / send の本体）、
                       running()（current() の Driver を返す。無ければ K12）
    component.py       Component: __init_subclass__ でクラス属性の Port / Node / Network を拒む（K1）
    node.py            Node: 利用者の API、設定（名前修飾した 1 属性）、register_state、既定の state_dict / load_state_dict、
                       is_state(value)（状態の値かの判定。tree と共有）、initialized(node)（K2 のため）
    network.py         Network, Group(**named)
    driver.py          Driver: ノード 1 つ分の入口（start / restore / update / pause / resume / exception / stop / capture）。
                       contextvars の Context、CUDA の device と stream、hook の後の同期。wires と clock を持つ
    tree.py            walk(root) -> Tree: __dict__ を代入順に走査し、パス・ノード・ポートの所有者・channel を作る。
                       K2, K3, K6, K13, K15, K24 を検査する。何も変えない。name(path) と label の規則
    wiring.py          wire(tree, clock, state, inputs) -> Wiring: Outlet / Slot / InputGroup を作り、initial・復元するスロット・
                       inputs を書き、ノードごとに Wire の dict と Driver を作る
    state.py           capture(...) / restore(...): 状態の木を作って CPU へ移す。K21 を検査して復元の準備をする
    run.py             Run / ManualRun / run() / _start(manual)。ホストの API、起動の 3 段、走った印（WeakSet）、終了、atexit
```

消すファイルは次のとおりです。

- `_core/binding.py`: `port.py` の `Wire` と `codec.py` に分ける
- `_core/graph.py`: `tree.py` にする
- `_core/hooks.py`: bindings の thread-local（`_native.current()`）に置き換える
- `_core/session.py`: `run.py`、`wiring.py`、`state.py`、`driver.py` に分ける
- `_core/timer.py`: native の型をそのまま公開する
- `_core/lifetime.py`: `run.py` に入れる
- `_core/pc/`: `mujou/pc/` へ移す
- `mujou/typing.py`

### 4.3 `crates/`

```
crates/mujou-core/src/
  lib.rs               pub mod time, dlpack, schema, buffer, slot, runtime
  time.rs              Timer（Source { Real, Virtual }、start と scale を受け取る生成、凍結は hold の 1 系統で、ホストの pause も hold 1 つ）、
                       Timestamp、PauseHold、RunClock（now = min(timer, end)、end() はランタイムだけが呼ぶ）、global()
  dlpack/ffi.rs        dlpack.h を写した DL* 構造体だけ（葉）
  schema.rs            FieldSpec, Schema, FieldError, 表示の整形（ffi に依存）
  buffer.rs            BufferPool, Buffer, Message（values, time_ns: Option<u64>）
  dlpack/import.rs     copy_from: Schema と照合し、stride を扱いながら必ずコピーする（エラーは FieldError）
  dlpack/export.rs     export_versioned / export_legacy
  slot.rs              Slot（enum Source { Own, Group } で InputSlot を吸収）, Outlet, Target, InputGroup。
                       時刻はすべて RunClock から取る。名前（label）は呼び出し側が渡す
  runtime/mod.rs       公開: Runtime, RunConfig { nodes: Vec<NodeSpec>, manual, workers: Option<NonZeroUsize> }, NodeSpec,
                       NodeHooks（既定実装あり）, enum Hook, Reply。Shared の impl はここにだけ置く。
                       RUNTIME_OF_THREAD（このスレッドがどのランタイムのものか。内部の Reentrant 検査に使う）
  runtime/lifecycle.rs ライフサイクルの唯一の状態機械（Command, advance, record, dt()）。時刻は RunClock だけから取る
  runtime/sched.rs     ワーカープール（ready キュー、target のヒープ、timekeeper）。専用スレッドは「1 ノード専属のワーカー」。
                       実効のワーカー数は min(要求か既定, プールに載るノードの数) で、0 もありうる
  runtime/control.rs   制御スレッド（startup / step / capture / stop）。stop で clock.end()。
                       終了後の capture は、呼び出し側のスレッドで lock を取って行う
  runtime/error.rs     HookError { hook: Hook }, NodeFailure { name, hook }, RuntimeError
crates/mujou-core/tests/
  time.rs              実時計、仮想時計、start と scale からの生成、hold、RunClock の終了
  runtime/main.rs      Probe と Mode（Dedicated / Pool / OneWorker / TwoWorkers）
  runtime/{lifecycle,period,sched,capture,failure,manual,restore}.rs
crates/mujou-python/src/
  lib.rs               #[pymodule] の宣言だけ
  wait.rs              wait_interruptibly と、秒の変換（to_ns / to_duration）の唯一の実装
  errors.rs            import_exception!(mujou._core.errors, NetworkError)
  time.rs              Timer, VirtualTimer（extends Timer）, ReadOnlyTimer（RunClock）, Timestamp, global_timer。
                       pyclass の module は "mujou.time"。制御するメソッドは current() を見て、hook の中なら拒む（K16）
  slot.rs              Schema, Outlet, Slot, InputGroup, Message, BufferView
  dlpack.rs            capsule の借用と貸し出し
  runtime.rs           start(nodes, clock, *, manual, workers) -> Runtime、PyDriver（NodeHooks。Python を呼ぶ前後で CURRENT を付け外しする）、
                       Hook から Driver のメソッド名への対応表、thread-local の CURRENT と current()、
                       ワーカー数の既定（available_parallelism。GIL があれば半分）
  hint.rs              スタブ用の型注釈（signature の文字列注釈で書けない箇所だけを残す）
crates/mujou-stubgen/src/main.rs   _native.pyi を生成する（例外の補完 EXCEPTIONS は消す）
```

Rust の依存の向きは次のとおりです。

```
time（葉）          dlpack::ffi（葉）
                    schema ← ffi
                    buffer ← schema
                    dlpack::{import,export} ← schema, buffer
slot ← buffer, schema, time
runtime ← time      （runtime はメッセージもスロットも知らない）
runtime/mod → {lifecycle, sched, control, error}   親から子への一方向。子は自分の型のメソッドだけを持つ
mujou-python: wait ← errors ← time ← slot(dlpack) ← runtime ← lib
```

### 4.4 `tests/`（`src/` をミラーする。中身は、できるだけ公開名を通して書く）

```
tests/
  test_layers.py              import の層（DAG）と _native の許可リストを、AST で検査する（層の規則の正本）
  test_claude_hooks.py        .claude/hooks のテスト（今のまま。coverage の対象は src/mujou だけなので分けない）
  mujou/
    conftest.py               virtual（新しい VirtualTimer）, start, start_manual, cuda マーカー
    nodes.py                  共通のテスト用ノードと比較の helper（Leaky, Chain, Calls, Probe, Pair, Hidden, assert_same_tree, close）
    test_init.py              mujou / mujou.time / mujou.testing / mujou.pc の __all__ を、1 か所で固定する
    test_pyright.py           利用者のコードを pyright strict で検査する（横断テスト。ミラー元を持たない）
    test_time.py              Timer / VirtualTimer / ReadOnlyTimer / Timestamp。global_timer は同一性と、実時間の smoke 1 本だけ（TestPause は 1 つ）
    test_testing.py           testing.run と ManualRun.step(dt, *nodes)（K19、順序、最初の dt、period の無視）
    _core/
      test_spec.py            TensorSpec / DictSpec（K7, K8、zeros / empty / rand、dtypes と DLPACK のキーの一致）
      test_codec.py           全 dtype × Tensor / dict の往復のビット一致、必ずコピーすること、K9 の構造の誤り
      test_port.py            宣言、connect（K10, K11 の両端、同じ相手への 2 回目、自己接続）、recv / recv_with_time / send / send_to、
                              送るだけの Port の fan-out、K12（他のノード、ホスト、__init__、run の外、on_start で作った Port）
      test_component.py       K1
      test_node.py            設定（K4, K5）、状態の規則（テンソル、state_dict を持つ物、クラスは除く、register_state、K30、乱数生成器）、
                              既定の state_dict / load_state_dict（K22）、timer（hook の中だけ）
      test_driver.py          hook を呼ぶ順、hook ごとの FFI が 1 回、CUDA の stream と同期、contextvars、
                              super を呼ばない override でも枠組みの処理が飛ばないこと
      test_network.py         Network / Group、パス、dataclass の Network
      test_tree.py            walk の検査（K2, K3, K6, K13, K15, K24）、パスの順序、label の規則
      test_wiring.py          スロットの配置、label、fan-out の channel、initial（K9 をパス付きで）
      test_state.py           capture → restore の往復、木の形、K21、CPU のコピーであること、"to" のキー
      test_run.py             ホストの I/O（K17, K18）、K14, K16, K20, K23、起動の 3 段（段 1・段 2 で断られたら同じインスタンスで再実行できる）、
                              shutdown / join、例外による終了、workers（全ノードが専用スレッドの実行を含む）、atexit（サブプロセス）
      native/test_module.py   モジュールの属性、GIL を有効にしないこと、stub と実行時の一致
      native/test_slot.py     Schema / Outlet / Slot / InputGroup / Message の契約、Rust の shape / dtype のエラー文（K9）
      native/test_runtime.py  start / Runtime、hook ごとの FFI の回数、current()、capture の timeout と排他、RunClock での end
    pc/
      _reference.py           純 PyTorch の Jacobi 参照実装（仕様から書く）
      test_functional.py      vjp / loss_vjp の有限差分、relax_ / learn_ の式、K29
      test_clock.py           LearnClock を、仕様の式から書いた期待値で検査する（K26、learn_interval=inf）
      test_messages.py        activity_spec（K28）と各メッセージ
      test_time_constants.py  K25
      test_losses.py          mse / cross_entropy（K29 のうち mse）
      test_input.py           送る内容、jumps が書き込みの回数を数えること、時刻 0 に書いた最初の入力を学ぶこと
      test_layer.py           Jacobi との一致、不動点での BP 一致、α=0 での PC への退化、更新順への非依存、learn_interval、max_dt、跳び、mask、
                              復元、ノードの分け方を変えても同じ結果（test_two_constraints_in_one_node_learn_as_in_two）、
                              mask がすべて False の間は全ノードの θ がビット一致で動かないこと
      test_output.py          K27、教師なしのとき Σdt を捨てること
      test_chain.py           Chain が手組みと torch.equal で一致すること、act の受け渡し、パス
      test_show.py            show が書き込みに対する最初の予測を返すこと（実時間の run で書く）
```

### 4.5 `docs/`（`docs/en/` と `docs/ja/` は同じ構成。API のページは英語だけで、日本語のサイトからは英語にフォールバックする）

```
index.md                     思想と、学び続けるノードを 20 行ほどで見せる
install.md
concepts.md                  9 つの概念と「ノードが守る規則」の節。注意書きの正本はここ
tutorial/1-run.md            examples/quickstart.py。後半で fan-out と MultiPort の合流（examples/branching.py）
tutorial/2-node.md           examples/leaky.py（dt、状態、period）
tutorial/3-test.md           examples/test_leaky.py（同じ Leaky をテストする）
tutorial/4-learn-in-a-node.md examples/online_learning.py（ノードの内側の BP。ストリームから学び続ける）。最後に topics/update-rules.md へのリンク
tutorial/5-learn-with-pc.md  examples/pc_regression.py（pc.Chain と pc.show。書くのは tau と seconds だけ）。最後に「次は examples/mnist」
topics/ports.md              Port / MultiPort / Channel、fan-out、循環、initial、時刻 None、役割の違う入力は Port を分けること（examples/branching.py）
topics/groups.md             Group とパス（examples/group.py）
topics/host-io.md            write / read / subscribe / wait_for、初期値の 3 つの道の表（initial / inputs= / write とそれぞれの時刻）（examples/subscribe.py）
topics/time.md               Timer, global_timer, VirtualTimer, time scale, 時間の単位の表、複数の実行を同時に止める、復元した実行のタイマー（examples/timers.py）
topics/threads-and-devices.md workers, dedicated_thread, GIL, CUDA, generator と seed, torch のスレッド数
topics/errors.md             NetworkError、on_exception → on_stop、止まった時刻と実時間、logging、timeout、インタプリタの終了（examples/errors.py）
topics/update-rules.md       学習則をノードの中に書く。双方向の Port、目標の中継、新しさの判定、LearnClock（examples/feedback_alignment.py）
topics/pc.md                 時定数の選び方、learn_interval の目安、安定の条件、落とし穴、自作の PC ノード（LearnClock と functional）、
                             MNIST（examples/mnist。最後の半端なバッチは mask を False にして詰める）
examples.md                  「やりたいこと → プログラム → ページ」の表（全ファイルを載せる）
api/node-and-ports.md        Node, Port, MultiPort, Channel, MessageSpec, TensorSpec, DictSpec
api/network.md               Network, Group
api/run.md                   run, Run, NetworkError
api/time.md / api/testing.md / api/pc.md
```

`concepts.md` の「ノードが守る規則」は、次の項目を持ちます。

- `time.sleep` を使わない（ブロックするノードには `dedicated_thread=True`）
- 1 回だけ走る（終わった後は、属性を読んでよい）
- 走っている間は、ホストからノードの属性を読まない
- `on_start` では、状態に依存する出力をしない（復元はその後に入る）
- `on_stop` で状態を変えない
- 実行中に変わる値は `register_state` する
- hook は通常の PyTorch と同じく勾配を記録する。パラメータの更新は `torch.no_grad()` の中で行い、パラメータから計算した値を次の更新へ持ち越すときは detach する
- 例外で終わった後の時刻は止まっているので、減速のような処理は実時間で測る

### 4.6 `examples/`

```
README.md                    規約（手引きであってテストではない。snippet の印。with で run を終える。just type が静的に検査する）
quickstart.py                2 ノードの Affine。write と wait_for
leaky.py                     連続時間のノード Leaky（厳密解）。__main__ で鎖を走らせ、厳密解と比べる
test_leaky.py                同じ Leaky のテスト（uv run pytest examples/test_leaky.py。CI は実行しない）
online_learning.py           ストリームから学び続ける、ノードの内側の BP（torch_node.py を吸収）
feedback_alignment.py        新規。PC 以外の、ノードをまたぐ学習則（双方向の Port、y の中継、新しさの判定）
pc_regression.py             新規。pc.Chain と pc.show で、合成データの回帰を数十行で書く
save_and_restore.py          状態を保存し、新しいインスタンスで続きから走らせる
branching.py                 MultiPort, send_to, Channel, fan-out
group.py                     Group とパス
subscribe.py                 subscribe
timers.py                    Timestamp、実行ごとの Timer、time scale、global_timer で 2 つの実行を同時に止める（pause.py を吸収）
errors.py                    例外と NetworkError.__cause__、on_exception → on_stop
mnist/main.py                pc.Chain で組む MNIST と、BP との比較（plain な pc.Layer を使う）
mnist/watch.py               dt のヒストグラムによる監視（main から --watch で使う）
mnist/metrics.py             Epoch, correct, evaluate（bp.py から移す）
mnist/bp.py                  BP のベースラインだけ
mnist/data.py                MNIST の読み込みと、mask 付きのバッチ（pc に依存しない。素の dict）
```

### 4.7 `specs/`（現在の設計と、その理由だけを書く。日付、経緯、エラー文言は書かない）

```
README.md                  目次（読む順）と「知識の置き場所の表」（9 節）
overview.md                目的と不変の原則
architecture.md            層と各モジュールの責務、Rust は仕組み / Python は方針という境界、free-threaded の規則、
                           内部の名前の規約（D51）（層の一覧の正本は tests/test_layers.py）
scenarios.md               提供シナリオ S1〜S9、サポートしないもの、判定規則 G1〜G6、ガード台帳 K1〜K30（ガードの正本）
nodes.md                   ノード、hook と呼ぶ順、Driver、状態の規則、period、CUDA
execution-model.md         スロット、連続時間、RunClock、停止 / 取得 / 復元、状態の木の意味、終了と例外、lifecycle の遷移表
network.md                 Port / MultiPort / Channel、connect、パス、__dict__ の走査、起動の 3 段
messages.md                spec、DLPack で必ずコピーすること、codec
native-core.md             Rust の責務、M:N スケジューラ、ABI のレコード、ツールチェーン（python-environment.md を吸収）、性能の予算
update-rules.md            差し替え可能性と、pc が _core の外の利用者であること
pc.md                      mujou.pc の現在の設計（状態、メッセージ、on_update の段、LearnClock、時定数、Chain、mask がすべて False なら θ が動かないこと）
pc-alm-continuous-time.md  連続時間形の解説（古い記述を直す）
research/pc-experiments.md 比較実験の表、却下した案、監査の経緯（memory/agents からの数値もここへ移す）
research/benchmarks.md     性能の基準値（ステップ 1 で記録し、以後の比較に使う）
open-decisions.md          本当に未決定のものだけ
```

---

## 5. 主要な設計判断

議論で食い違った論点は、すべてここで裁定します。反証で改めた判断には、その旨を書きます。

### D1 層の強制

- **採用:** `tests/test_layers.py` の 1 本で検査する（AST で読み、DAG と `_native` の許可リストを見る）。
- **却下:** ruff の `banned-api`。ruff の禁止と自作のテストを併用する案。
- **理由:** `_core` の中の層の順序まで検査できるのは、テストだけだからです。仕組みを 2 つ持つと、正本も 2 つになります。

### D2 起動の順序と、断られた run（改訂）

- **採用:** 起動を 3 段にします。
  - **段 1（検査。何も変えない）:**
    - `tree.walk(root)` で木を作り、K2、K3、K6、K13、K15、K24 を検査する
    - 走った印（K14）、`state` の木の形（K21）、`timer` と `state` の組（K23）、`inputs` の Port（K17）、`workers`（K20）を見る
  - **段 2（native の準備。新しいオブジェクトだけを作り、書く）:**
    - タイマーを決める。`state` があれば、保存した時刻と time scale から始まる新しい `Timer`（testing では `VirtualTimer`）。無ければ、渡された `timer` か `global_timer`
    - `ReadOnlyTimer(timer)` を作る
    - `wiring.wire` で Outlet / Slot / InputGroup / Wire / Driver を作る。initial、復元するスロットの値、inputs を書く。値の K9 はここで、パス付きの label で出る
  - **段 3（取り消せない操作）:** 全ノードに走った印を付け（`run.py` の WeakSet）、`_native.start` を呼ぶ。
  - 段 2 までに断られた run は、利用者のオブジェクトを何も変えていません。同じインスタンスのまま、直してすぐに再実行できます。
  - 段 3 で失敗するのは、hook の例外と内部エラーだけです。そのときノードはもう走っているので、もう一度 run すると K14 で知らせます。
- **却下:**
  - 「インスタンスは何があっても使い捨て」（最初の案）
  - 今の `unbuild`、`ours`、`start_once`
  - 段 3 の `_native.start` の中で、復元の時刻をタイマーに読み込む案（前版と、反証 M2 の案）。R8 を断られたときの代案として残し、その場合は印を各ノードの `Driver.start` が付ける
- **理由:**
  - 印を最後に付ければ、巻き戻すものがありません。再実行を許すのは、そのためのコードを書いた結果ではありません。順序から自然に成り立つ副産物です。
  - 復元の時刻をほかの実行のタイマーに読み込む場合を、そもそも無くしました（D50）。そのため、段 3 で利用者の誤りが見つかることはありません。
  - inputs の時刻は、段 2 で書いた時点の、その実行のタイマーの時刻です。復元した実行では、タイマーが最初から保存した時刻にあるので、時刻の軸がずれません（反証 B3）。

### D3 build というフェーズ

- **採用:** `Graph`、`Network.build()`、`_BUILT`、`BUILD_LOCK`、`start_once`、`unbuild_tree`、`network.unbuild`、`Port._frozen` / `freeze` / `thaw` / `is_frozen` を削除します。「走った」という事実は、`run.py` の印だけが持ちます。
- **却下:** `check()` だけを公開して残す案。
- **理由:**
  - 利用者に見える効果は「事前の検査」だけで、それは `run` もします。
  - 凍結の副作用は、「build した後は Group に入れられない」という罠を生みます。
  - 並行 build は提供シナリオの外です（G5）。

### D4 ルートと Group

- **採用:** `run(root)` は 1 つだけ受け取ります。並べるのは `Group(**named)` だけにし、位置引数は廃止します（`mujou.testing.run` も同じ）。
- **却下:**
  - `run(*components)`（reuse、core-boundary）
  - `Group(*positional)` を残す案（simplicity、reuse）
  - `Network(**children)` で `Group` を吸収する案（reuse、dedup、user-first）
- **理由:**
  - root を 1 つにすると、`RunGroup`、`_Tree.by_run`、パス文字列の `.` の有無による推測が消えます。
  - 位置引数を残すと、同じノードのパスが、組み方によって `""`、`"0"`、`"node"` と変わります。保存した木が合わなくなります。
  - `Network(**children)` は、継承して組む使い方と、その場で並べる使い方を、1 つの基底のコンストラクタに混ぜます。`nn.Module` と `nn.Sequential` を分けているのと同じ理由で、`Group` の 1 語を残します。

### D5 パスと label の表現

- **採用:** パスは、ドットでつないだ文字列にします。
  - ノード単体をルートにしたときのパスは `""` です。
  - 表示用の呼び名は `tree.name(path)`（`path or type(root).__name__`）の 1 関数が決めます。
  - ポートの label は、`name(owner_path) + "." + attr` の 1 つの規則にします。ルートのノードのポートなら `"Leaky.prev"`、子なら `"body.l1.prev"` です（反証 C10）。
- **却下:** パスをタプルで持つ案（simplicity）。ルートを常に Network にする案。
- **理由:**
  - タプルにすると、利用者が `tree["nodes"][("layers", "0")]` と書くことになり、負担が移るだけです。
  - ルートを常に Network にすると、ノード単体のテストが重くなります。
  - label の規則を 1 つにすれば、ノードとポートを同じ規則で名指しできます。

### D6 相手の数と接続のデータ（改訂）

- **採用:** `Port` と `MultiPort` の 2 クラスにし、接続は各ポートが「相手 → Channel」の dict（挿入順）で持ちます。
  - `Port`: `recv()` が 1 つの値を返すので、**受け取る Port** の相手は高々 1 つです。**送るだけの Port** は何本でも繋げられ、`send` は全員に配ります。
  - `MultiPort`: 同じ種類の相手を束ねます。相手ごとの `recv()` の dict を connect の順で返し、`send_to` で相手を選んで送ります。
  - `connect` は、両端の dict に同じ Channel を入れます。同じ相手にもう一度繋ぐと、同じ Channel を返します。自己接続も、dict の 1 項目になるだけです。
  - K11 は `connect` で、両端を `for p, q in ((self, other), (other, self))` で見ます。受け取る `Port` に、別の相手がすでにいれば拒みます（反証 C8）。
  - `prev` と `next` は、属性名の慣習として残します。
- **却下:**
  - 「`Port` は常に 1 本まで」（simplicity、linus、structure の撤回後）
  - `ManyPort` という名前（reuse）
  - 自己接続と二重接続を K として拒む案（反証 B1 の第 1 案）
- **理由:**
  - 制約は「`recv()` が 1 つの値を返す」ことから導かれます。条件付きの例外ではなく、データの形の帰結です。
  - Outlet はもともと「outbox 1 つと、相手ごとの channel スロット」を持ち、`send` で全部に書きます（`session.py:_add_outlet`）。fan-out は、今のスロットのモデルのまま成り立ちます。
  - 今は、`MultiPort` どうしを 2 回繋ぐと 2 本の Channel ができます。自己接続を許すと、同じ list に同じ Channel が 2 回入り、`Channel.other` と channel の番号が壊れます（反証 B1、`port.py:315`、`session.py:144`）。相手をキーにした dict なら、これらは表せなくなり、状態の木の `"to"` も相手のパスで一意に引けます。
  - 2 回目の connect を「同じ 1 本」と読むのは、「a と b は繋がっている」という集合の意味どおりです。
  - `feedback_direction_names` を改めるので、ユーザーの確認が要ります（R2）。

### D7 コアの carry

- **採用:** 廃止します（**ユーザーの確認が要る**。今日の #89 の決定を覆すため）。PC は、`y`、`mask`、`time` を `Activity` の普通のフィールドとして運び、ノードが自分で次のメッセージに写します。
- **却下:** `send_carry=spec` と `relay=port` に分け、コアには「空の Carry」を持たせて残す案（structure、reuse、dedup、yagni、core-boundary、linus の当初の案）。
- **理由:**
  - 使い手は PC の 1 人しかいません。carry を残す理由は「活動とターゲットを 1 つのメッセージに入れて、組み違えを防ぐ」ことでした（`memory/project_carry.md`）。同じメッセージのフィールドにしても、同じことが成り立ちます。
  - 廃止すると、次のものが一度に消えます。
    - Port の型引数 2 つと引数 3 つ、`carried`
    - `Binding._message` の 3 通りの分岐、`relay_from` の後付けの初期化
    - Schema の carry、Rust の `Message::carry`、`send_with(relay)`、`new_carry`
    - 状態の木の `carries` と、スロットの項目の 4 番目の要素
    - carry に関わるガード（K8 の carry 分、relay の局所性、`recv_carry` への write の拒否）
  - `memory/project_carry.md` が状態の木に carry を入れた理由は、「recv の後に中継元が更新されると、復元したときにずれる」ことでした。しかし pc の全ノードは、同じ update の中で recv してから send しています（`layer.py:223→262`、`output.py:144→164`。invariants の検証でも確認済み）。フィールドで運ぶなら隠れた状態が無いので、ずれは起きません。
  - 失うのはゼロコピーです。コストは 1 ホップあたりの y、mask、time のコピーです。分類（y が (B,)）では h より小さい量ですが、回帰や自己符号化（y が (B, d)）では h と同じ量になり、CUDA のノードでは各ホップで device との往復も起きます（反証 S6）。MNIST と `pc_regression` の密な y の両方で測り、結果を `specs/research/benchmarks.md` に記録します。GPU の効率は設計目標ではないので、測った結果は記録だけでよいとします。
  - 確認が得られなかった場合の代案は、却下した案にします（引数を分ける、空の Carry で送信を 1 本にする、`carried()` の key 引数を消す、`Port[S, R, C = Never]`）。

### D8 「まだ届いていない」の表し方

- **採用:**
  - `recv_with_time()` は `tuple[R, Timestamp | None]` を返します。`None` は「書き手がまだ送っていない（initial か zeros）」を表します。
  - `has_received` は削除します。ホストの `read_with_time` / `subscribe` / `wait_for` も、同じ型にします。
  - コアの `Message` は `time_ns: Option<u64>` を持ち、`sent` フィールドを吸収します。
  - 状態の木の項目 E も `{"value", "time_ns": int | None}` にします。
- **却下:** 現状を維持する案（core-boundary と yagni の反対意見:「利用者に None の絞り込みを強いる」）。`Run.read` も常にタプルで返す案（simplicity）。
- **理由:**
  - 今の `has_received()` は、スロットを読み直しています。`recv_with_time()` との間に新しいメッセージが届くと、値と「送られたか」が別のメッセージの情報になります（`input.py:83-84` に競合がある）。1 回の読み出しから得れば、競合ごと消えます。
  - 使っている側では分岐が減ります。`examples/online_learning.py` の `if not has_received(): return` と、その後の `if t == self.last: return` は、`if t is None or t.ns == self.last: return` の 1 本になります。
  - `read` は値だけを返すまま残します。ノード側の `recv` / `recv_with_time` と対称になり、quickstart も短く書けます。

### D9 Network が公開する Port

- **採用:** `Network` に代入した `Port` は、ただの参照です。`tree.walk` は、Network の `__dict__` の Port を読み飛ばします。
- **却下:** 登録して、所有者を検査する案（現状、`graph.py:157`）。
- **理由:** ホストは Port オブジェクトそのもので読み書きするので、公開することに機能上の意味がありません。木の外の Port を使ったときは、K17 / K18 が、使ったその行で知らせます。

### D10 Port の実行時の状態（改訂）

- **採用:** Port は宣言（spec、initial、相手 → Channel の dict）だけを持ち、実行時の状態は持ちません。
  - 実行時の読み書きは `Wire` が担います。`Wire` は label、outlet、sources、device のただのデータと、recv / send の本体からなり、`port.py` に置きます。
  - ノードごとの `Driver` が、`wires: dict[Port | MultiPort, Wire]` を持ちます。
  - Port のメソッドは、`port.running()`（`_native.current()` が返す、いま hook を走らせている Driver）の `wires` から自分の Wire を引きます。引けなければ K12 です。
- **却下:** Port に `_Wire` を差す案（前版）。`binding.py` を独立の層にする案（structure）。`SingleBinding` / `MultiBinding`（core-boundary）。
- **理由:**
  - 所有者の検査が、dict の lookup 1 回に吸収されます。次の場合が、どれも同じ 1 つの失敗になります。
    - 他のノードのポートを使う
    - ホストや `__init__` の中で使う
    - run の外で使う
    - `on_start` で作ったポートを使う
  - 走り終わったポートに古い状態が残らないので、凍結・解除・差し直しが要りません。
  - 他のモジュールからポートの `_` 属性に触る箇所も無くなります（D51。前版の `wiring.py` が `_Wire` を差していた箇所は、反証 A7 の指摘どおり pyright の警告になっていた）。

### D11 ノードが見る時計（改訂）

- **採用:** native の `RunClock` を、Rust ランタイムの**唯一の時刻源**にします（反証 M1）。公開名は `ReadOnlyTimer` です。
  - dt、`last_update_ns`、スロットの時刻は、`RunClock.now = min(timer.now, end)` で取ります。期限まで待つのは Timer の上で行い、end を付けたときに起こします。
  - end は、正常終了では停止を求めた時刻、例外による終了では失敗を検出した時刻（`on_exception` の前）です。
  - Outlet、InputGroup、`start`、Driver が、同じ 1 つのオブジェクトを持ちます。`Node.timer` は、いま走っている Driver の時計を返します（1 つの実行の全ノードは同じ時計を持つので、どのノードから読んでも同じ）。
  - `ReadOnlyTimer(timer)` は誰でも作れます。end を付けられるのはランタイムだけなので、作っても害がありません（反証 A2）。
- **却下:** `ReadOnlyTimer` を消して本物の `Timer` を返し、実行時に拒む案（simplicity）。今の `stopped` / `itertools.repeat` / `_wrap`。「直接は作れない」を守る仕組み。
- **理由:**
  - 今の dt と `last_update_ns` は、共有の `Timer` から取っています（`runtime/pool.rs:146,344`、`runtime/node.rs:161,185`）。停止を求めた後に走った更新は、end より後の時刻を記録しえます。今は `session.py:301-320` の max でこれを吸収していますが、時刻源を `RunClock` にすれば、データの形のせいで起きなくなります。正常終了の後の木では、時刻が全スロットの時刻と全ノードの最後の更新以上になります。
  - 1 つのデータ構造で、`stopped`、`set_timer`、`_settle`、`_ended`、`_stop_timer`、max による推論、「例外の後はノードに timer が無い」という特別扱いが、すべて消えます（`session.py:255-320` で確認）。
  - 「1 つの実行の時刻は、1 本のタイマーの上にある」という不変条件が、型で守られます。読むだけ、という制約も型で表せます。
  - 時刻が止まった後の `on_exception` / `on_stop` は、`self.timer` で経過時間を測れません。ロボットの減速のような処理は実時間で測るよう、`concepts.md` に書きます。

### D12 ホストが時間を操作する窓口

- **採用:** `mujou.time` の関数 7 個を削除します。ホストは `run.timer.*` を使い、複数の実行をまとめて止めるときだけ `mujou.time.global_timer.*` を使います。名前は `global_timer` にします。
- **却下:** `GLOBAL` のメソッドの別名として残す案（yagni の当初の案）。名前を `GLOBAL` や `timer` にする案。
- **理由:**
  - 別名を残すと、「独自の `Timer` で走らせた run に `mujou.time.sleep` を使う」という落とし穴が残ります。
  - `GLOBAL` は、状態の変わるオブジェクトなのに定数の書き方になります。`timer` は、`run.timer` と紛らわしい名前です。

### D13 VirtualTimer

- **採用:** `mujou.time.VirtualTimer` に置き、native の `Timer` を `extends` するサブクラスにします。`VirtualTimer(*, start=Timestamp(0))` です。
- **却下:** `Timer(virtual=True)` のフラグ。`mujou.testing` に置く案。
- **理由:** `advance` という能力を型が持てば、`NotVirtual` の分岐は要りません。`use_virtual_clock` と `load_state_dict({'time_ns': 0})` の 2 手順を、4 か所で繰り返すことも無くなります。

### D14 Timer と Timestamp の実体

- **採用:** native の型をそのまま再エクスポートします。docstring の正本は Rust の doc comment です（`project_core_stub`）。pyclass の `module` を `"mujou.time"` にし、repr、docs、pickle で `mujou.time.Timestamp` と出るようにします（反証 D）。
- **却下:** `Timestamp` を Python に移し、`_native` は `int` の ns を返す案（core-boundary）。
- **理由:** Python に移すと、`now()` のたびに値を包む Python の層が戻ってきます。

### D15 テストの入口

- **採用:** `mujou.testing.run` と `ManualRun.step(dt, *nodes)` にします。docstring には次のことを書きます（3.3 節）。
  - period を無視すること
  - 最初の dt の起点
  - 省いたときの順序（木の順、Gauss-Seidel）
  - 同じノードを 2 回渡したときのふるまい
- **却下:**
  - `mujou.run(..., manual=True)`（linus は撤回）
  - `ManualRun.advance(seconds, *nodes)`（dedup は撤回）
  - `nodes` を省いたときに「期限に達したノードだけを期限の順に」更新する案（反証 1-4）
- **理由:**
  - bool の引数によって戻り値の型と既定のタイマーが変わるのは、フラグによる特別扱いです。
  - `advance` は、ノードを更新しない `VirtualTimer.advance` と同じ名前で、別の意味になります。
  - 手動モードで周期を無視するのは、ユーザーの決定です（`project_node_period`）。周期どおりに決定的に回したい需要は、まだ使い手がいません。pc の鎖は Chain が period をそろえるので、困りません。

### D16 Node の基底と Driver（改訂）

- **採用:**
  - `mujou.Node` は純 Python にします。Rust は、`_core/driver.py` の `Driver`（ノード 1 つにつき 1 つ）の 8 メソッド（start / restore / update / pause / resume / exception / stop / capture）を、hook 1 回につき 1 回だけ呼びます。
  - Driver は、どのメソッドでも次の順で動きます。
    1. ノード専用の contextvars の Context の中に入る
    2. CUDA のノードなら device と stream を設定する
    3. hook を呼ぶ（hook はいつも属性から引く）
    4. update と restore の後は、stream を同期する
  - 「スレッドが前に走らせたノード」を覚えて設定を省く仕組みは、消します。消すのは core の `NodeHooks::enter`、`LAST_NODE`、`Entry::Required` / `Skip` です。
  - 次のものも削除します（**CLAUDE.md の一文を書き換えるので、ユーザーの確認が要る**）。
    - `_native.Node` の基底
    - private hook 4 つ
    - `CACHED_HOOKS`
    - `attribute()` による名前の付け替え
    - `component.py` の `isinstance(_, _native.Node)`
- **却下:**
  - `_native.Node` を残す案（linus、user-first）
  - Python の Node に `_mujou_*` を置く案（structure、simplicity）
  - Driver に `enter` を残す案（前版）
- **理由:**
  - 今は、1 回の update ごとに、`on_update` と `_after_update` の 2 回 FFI をまたいでいます（`crates/mujou-python/src/runtime.rs:321-325`）。
  - `enter` を別のメソッドにすると、ノードを切り替えるプールではほぼ毎回 2 回になります（反証 B4。`runtime/mod.rs:283-286`）。hook ごとに設定すれば、覚えておく状態ごと消えます。CPU のノードでは何もしません。CUDA の設定のコストは R13 で測ります。
  - 利用者の MRO と `help(Node)` から、private hook が消えます。
  - 利用者が `super()` を呼び忘れても、CUDA の同期は飛ばなくなります。
  - `CACHED_HOOKS` の「`on_start` までは名前で引き直す」という意味論は、「いつも属性から引く」1 つになります。

### D17 「いま hook を走らせている Driver」の正本（改訂）

- **採用:** bindings（`crates/mujou-python/src/runtime.rs`）の thread-local `CURRENT` の 1 か所に置きます。
  - PyDriver は、Python を呼ぶ直前に `CURRENT` を付け、戻ったら外します。`_native.current()` で公開します。
  - K12 は Driver の `wires` の lookup で行います。K16 は、Python の `run._host()` と、bindings の `Timer` の制御メソッドが、同じ `current()` を見て行います。
  - 正常終了の後の capture も、呼び出し側のスレッドで、PyDriver が同じように付け外しします。
  - core の `RUNTIME_OF_THREAD`（このスレッドがどのランタイムのものか）は、内部の `Reentrant` 検査のために残します。「hook の中か」とは別の事実です。
- **却下:** core の thread-local に置く案（前版）。Python の `context.py` に置き、`Driver` が設定する案（dedup、yagni、linus の最終案）。
- **理由:**
  - core は Python のオブジェクトを持てないので、core に置くと、(ランタイム, 番号) から Python のノードを引く段取りが要ります（反証 C7）。
  - hook を呼ぶ者（PyDriver）が、呼んでいる間だけ印を持てば、事実の置き場所も、付け外しの場所も 1 つです。
  - `Timer.sleep` の検査は native の中でしかできないので（D12、D14 で Python の Timer の包みを消すため）、Python 側に置くと、Rust から Python を呼び戻すことになります。
  - recv はもともと `slot.read()` で FFI をまたいでいます。そこに数十 ns の呼び出しが 1 回増えるのは、`from_dlpack` とコピーの数 µs に比べて小さい量です。切り替える前に、ステップ 1 の基準値と比べて測ります（R9）。

### D18 hook の `super()`

- **採用:** override した hook の最初で `super()` を呼ぶ慣習を残します（`feedback_super_calls`）。基底の hook は何もしないので、呼ばなくても壊れません。
- **却下:** 「Driver があるので `super()` は要らない」（core-boundary、linus）。
- **理由:** ユーザーが明示した指示です。自作の基底クラスを継承したときの保険にもなります。

### D19 `on_pause` / `on_resume` / `on_exception`

- **採用:** 3 つとも残します。
- **却下:** 3 つとも削除する案（simplicity、yagni）。`on_pause` / `on_resume` だけを削除する案（dedup、user-first）。
- **理由:**
  - `specs/nodes.md` は、`on_exception` の存在理由を「`on_stop` だけでは正常終了と異常終了を区別できず、アクチュエータを安全な姿勢で止められない」としています。`on_pause` / `on_resume` の理由は「取得のための停止中の I/O の一時停止と、再開時の値の破棄」です。どちらも、ロボティクスという本来の用途に紐づいたユーザーの決定です（`project_node_lifecycle`）。
  - repo の中に使い手がいないことは、ライブラリの拡張点を消す根拠になりません。YAGNI はガードに使う基準です。
  - ライフサイクルの状態機械を 1 本にすれば（D30）、残すコストは小さくなります。

### D20 `super().__init__()`（改訂）

- **採用:** 必須にします。呼ばなかったことは、run の段 1 で `tree.walk` が K2 として知らせます。判定は、`Node.__init__` が置く設定の属性が `__dict__` に無いことです（`node.initialized(n)`）。
  - `super()` の前の代入は、普通の代入として通ります（`__setattr__` を持たないため）。
  - `super()` を呼ばない場合の救済（クラス属性の既定値 7 つ、`_on_enter` での generator の作り直し）は削除します。
  - `feedback_super_calls` の「テストの中のノードは省略してもよい」は削除します。
- **却下:** 最初の代入で知らせる案（前版。`__setattr__` が要る。D49）。run の開始時に、遠回しな `AttributeError` に任せる案。
- **理由:**
  - 段 1 は、断られても同じインスタンスで再実行できる場所で、文言はクラス名を名指しします。
  - 素の `AttributeError`（`'Leaky' object has no attribute '_Node__config'`）では原因が分からないので、G1 として残します。

### D21 状態に入るもの（改訂）

- **採用:** 規則を 1 文にします。「テンソルの属性、`state_dict` と `load_state_dict` を持つ属性（クラスそのものは除く）、`register_state` した名前、そして乱数生成器の状態」が状態に入ります。
  - どれも、取得と復元のときに `__dict__` から決めます。属性が状態かどうかは、最後に代入した値で決まります。
  - 設定（コンストラクタの引数から作る tau などのハイパーパラメータ）は、状態に入れません。カウンタや最後に見た時刻のように実行中に変わる値は、`register_state` します。
  - 乱数生成器の状態は、既定の `state_dict` が `"generator"` のキーで必ず含めます（`generator` は予約名の property なので、利用者の属性と衝突しない）。
  - 公開の Protocol 名は作りません。
- **却下:**
  - 「`super().__init__()` の後に代入した、ポートと callable 以外の属性は、すべて状態」とする案（反証 1-1）
  - `nn.Module` を特別扱いする現状
  - `mujou.typing.Stateful` を公開する案（reuse）
  - duck typing をしない案（dedup、yagni）
- **理由:**
  - 1-1 の案は、設定をすべて状態にします。そうすると、次の問題が出ます。
    - コンストラクタの引数を変えて作り直したネットワークに古い木を戻すと、新しい tau が黙って古い値で上書きされる。1-1 が `Leaky` の例で指摘した害が、既定の動きになる
    - 状態から外す仕組み（新しい概念）が要る
    - ロボットのノードが持つ I/O のハンドルやロガーは weights_only で読めないので、外し忘れると取得が壊れる
  - `nn.Module` と同じ線引き（テンソルと部品は状態、Python の属性は設定）なので、PyTorch の利用者の予想に合います。
  - 1-1 が指摘した 2 つの穴は、別の形で塞ぎます。
    - 乱数生成器が漏れていた: 今は docstring で、利用者に自分で足すよう頼んでいた（`node.py:235-236`）。既定の `state_dict` が含める
    - 例が誤った型を教えていた: `Leaky` の tau は登録しない。`OnlineLinear` の `last` と `elapsed` は登録する（7 節）
  - `nn.Module` の特別扱いが、規則の一例に吸収されます。`pc.LearnClock` のように使い手が 2 人いる部品も、代入するだけで木に入ります。
  - 型そのもの（`self.kind = nn.Linear`）も `state_dict` 属性を持つので、`isinstance(v, type)` を除きます（反証 C2）。
  - 取得と復元のときに `__dict__` を 1 回読むだけなので、代入のたびに検査する今の `__setattr__` より速くなります。
  - 隣のノードを属性に持つことは、K3 が拒みます。`torch.optim` は free-threaded で GIL を有効にしてしまう（`examples/mnist/bp.py:53`）ので、例では勧めません。

### D22 既定の `load_state_dict` と hook の順（改訂）

- **採用:** strict にします。`state_dict` と同じ規則で決めたキーの集合が一致することを求め（K22）、1 項目ずつ戻します。
  - テンソルは、shape を照合して `copy_` する
  - `state_dict` を持つ物は、`load_state_dict` を呼ぶ
  - `register_state` した値は、`copy.deepcopy` を代入する
  - `"generator"` は、`set_state` で戻す
  - 未登録のテンソルから新しい属性を作る処理と、木全体の `deepcopy` は削除します。
  - 呼ぶ順は、今と同じ `on_start` → `load_state_dict` → `on_update` にします。
- **却下:** 全部を検査してからコピーする原子性。順を `__init__` → `load_state_dict` → `on_start` に変える案（反証 1-2）。
- **理由:**
  - 原子性: `load_state_dict` はランタイムの中で呼ばれ、失敗すれば `NetworkError` で実行が終わります。「何も変わらない」ことに意味がありません。
  - 順序: 変えると、`on_start` で状態を作るノードが、復元した値を `on_start` で黙って上書きします。今の例（`examples/online_learning.py:60-64`）と、ユーザーの決定（`project_node_lifecycle`「on_start は状態の確保」）が、この書き方です。1-2 が挙げた害（`on_start` が復元する前の値を見る）は、`on_start` で状態に依存する出力をしなければ起きません。前者は黙って状態を失い、後者は出力の規則で避けられるので、順は変えません。
  - `concepts.md` の規則と `on_start` の docstring に、次のように書きます。「`on_start` では、状態に依存する出力（送信、アクチュエータへの指令）をしない。復元はその後に入る」。
  - 例では、状態を `__init__` で作る書き方にそろえます。PyTorch と同じで、`on_start` は I/O を開くのに使います。1-2 が指摘した「2 通りの書き方」の迷いは、例を 1 つにそろえれば消えます。
  - 木全体を `deepcopy` するとテンソルのメモリが 2 倍になります。共有を避ける必要があるのは `register_state` の普通の値だけなので、その値だけをコピーします。

### D23 状態の木（改訂）

- **採用:** 素の dict にします。ノードの木をミラーした入れ子で、メッセージの項目は常に同じ形 E にします。

```python
{
    "timer": {"time_ns": int, "time_scale": float},
    "nodes": {
        "<ノードのパス>": {
            "state": "<Node.state_dict() の戻り値（dict。既定なら 'generator' を含む）>",
            "last_update_ns": int,
            "ports": {
                "<属性名>": {
                    "out": E,                        # 送るポートの outbox（ホストが読む）
                    "to": {"<相手のポートのパス>": E},  # 相手ごとの channel スロット（相手は D6 の dict で一意）
                    "in": E,                         # 未接続で受け取る Port（ホストが書く）
                },
            },
        },
    },
}
# E = {"value": Tensor | dict[str, Tensor], "time_ns": int | None}
```

- 取得の手順は次のとおりです。
  1. 停止している間、`Driver.capture` が、ノードの stream の上で state_dict のテンソルを複製する
  2. ホストに返す前に、stream を同期する（反証 S1）
  3. 再開した後に、ホストのスレッドで CPU へ移して返す
- 正常終了の後の取得は、`Runtime.capture` が、呼び出し側のスレッドで lock を取って行います。2 つのホストスレッドが同時に呼んでも、hook は重なりません（反証 S1）。
- 例外で終わった後の `state_dict` は、`NetworkError` を出します。失敗したノードは更新の途中で止まっているので、木を作りません（反証 S2、今の `session.py:446` を維持）。
- `torch.save(tree)` と `torch.load(..., weights_only=True)` が、そのまま通ります。
- **却下:** `StateDict` クラス、`StateDict.nodes` の属性アクセス（user-first）、3 キーの木（simplicity の当初の案）、位置で意味が決まるタプル（現状）、`"to"` を channel の番号で引く案（反証 B1 の第 2 案）。
- **理由:**
  - `.cpu()`、dict のサブクラス、`__module__` の書き換え、`add_safe_globals` が消えます。
  - channel を相手のパスで引くので、接続の順序に依存しません。番号で引くと、相手の取り違えを K21 で検出できません。
  - スロット名の規約（`/out`、`/ch{k}`、`/in`）は、コアから出ていきます。
  - 停止中に CPU へコピーすると、停止の時間が延びます（core-boundary）。

### D24 1 回の実行の実体（改訂）

- **採用:** `Run` が `Session` を吸収します。配線は `wiring.py`（純関数）、状態の段取りは `state.py` に置きます。
  - 終了の状態は Python で持たず、Rust の `Runtime` に聞きます。`Runtime` は、`stop()`（待たない）、`join(timeout) -> bool`、`finished`、`check()`（失敗なら `NetworkError`）を返します。
  - `Run.shutdown` は `stop()` と `join(timeout)` の 2 行です。
- **却下:** Python に enum（`RUNNING` / `STOPPING` / `ENDED` / `FAILED`）を置く案（前版）。
- **理由:**
  - 1 対 1 の委譲層で、docstring は `Run` に、実装は `Session` にありました。
  - Python の enum は、Rust のライフサイクルと同じ事実を、2 か所で持つことになります（反証 B5）。今の `is_finished` と `request_stop`（`session.py:241-268`）が担っていた口も、これで閉じます。

### D25 torch とコアの変換、値の検査（改訂）

- **採用:**
  - 変換は `codec.py` だけが行います。spec の公開の属性（`shape`、`dtype`、`specs`）だけを読みます。
  - Tensor は「名前 `""` のフィールド 1 つ」として一様に扱います。正規化は `codec.fields(spec)` の 1 つの isinstance だけで、残りの処理は分岐しません。
  - dtype の対応表は、`spec.py` のパッケージ内部の名前 `DLPACK` が持ちます。`TensorSpec.dtypes` はそのキーで、codec はその値を読みます（反証 A6）。
  - 内部の `Spec` union と `as_spec` は削除します。
  - shape と dtype の検査は、Rust の `copy_from` に label（D5 の規則）を渡して、1 回だけ行います。Python が見るのは、Rust には見えない構造（Tensor か、dict のキーに過不足が無いか）だけです。
  - `initial` は、段 2 で Outlet に書くときに、パス付きの label で検査します（反証 B3）。
- **却下:**
  - shape と dtype を Python の `spec._check` に置き、Rust 側を内部エラーに降格する案（yagni の当初の案）
  - 今の二重の検査（`spec.py:308` と `dlpack.rs`）
  - `MessageSpec._fields` / `_flatten` / `_unflatten` の多態（前版。codec から他のクラスの `_` 属性を呼ぶことになる）
  - `Port.__init__` の時点で `initial` を検査する案（label も Schema もまだ無い）
- **理由:**
  - Rust は、memcpy の健全性のために照合をやめられません。どのみち消せない方に正本を置き、send の hot path から Python の分岐を消します。
  - dtype の表を `spec.py`（L1）に置けば、K7 の検査と codec（L2）が、下から上へ一方向に読めます。
  - `initial` の誤りは、段 2 で断られた run として同じインスタンスで直せるので、構築の行で知らせられなくても失うものは小さく、文言はパスで名指しします。

### D26 非連続のテンソル

- **採用:** stride の扱いは、Rust の `copy_from` に残します。
- **却下:** Python で `.contiguous()` してから memcpy する案（yagni）。
- **理由:** コピーが 2 回になり、DLPack で 1 回だけコピーするという原則の精神に反します。動いている部分を作り直すことにもなります。

### D27 例外

- **採用:** `NetworkError` は `_core/errors.py`（葉）で定義し、Rust は `import_exception!` で参照します。`PauseTimeoutError` は組み込みの `TimeoutError` にします。stubgen の `EXCEPTIONS` と、`test_native_stub` による例外の検査は削除します。
- **理由:** 定義、docstring の複製、公開の場所、ずれの検査が、3 か所に散っていました。利用者が `TimeoutError` を見分ける場面もありません。

### D28 timeout

- **採用:** timeout は、すべて実時間の秒で、既定は `math.inf` にします。負の値や NaN は、`wait::to_duration` の 1 か所で変換するときに、自然な例外になります。すべての docstring に「wall-clock seconds」と書き、`topics/time.md` に単位の表を置きます（反証 3-6）。
- **却下:** `state_dict` だけ既定が 30 で、有限かつ > 0 を求める現状。
- **理由:** 規則を 1 つにします。ブロックする hook を持つノードには、docs で timeout を渡すよう案内します。

### D29 インタプリタの終了

- **採用:** `run.py` の atexit だけにします（**確認が要る**）。
  - サブプロセスでインタプリタを終了させるテストを、GIL あり・なしの両方で通してから、監視用の daemon スレッドと `lifetime.running(kind)` を削除します。
  - 終わった run は、自分で後始末をします（他の run の起動に便乗した掃除はやめる）。
- **理由:** 監視スレッドは、文書化されていない CPython の終了順序に依存しています。docs では `with` を前提に書きます。

### D30 Rust のライフサイクルとコア（改訂）

- **採用:**
  - `lifecycle.rs` を、唯一の状態機械にします。専用スレッドは「1 ノード専属のワーカー」として扱い、dt の計算も、`RunClock` から取る 1 関数にします。
  - `Vec<NodeSpec>`、`enum Hook`、`NodeHooks` の既定実装を入れます。`enter` は消します（D16）。
  - `Slot` と `InputSlot` を 1 つの型にし（`Source`）、dlpack と slot の循環を断ちます。
  - `freeze.rs` を削除します。凍結は hold の 1 系統にし、`Control` の `startup: Option<PauseHold>` と `capture: Option<PauseHold>` の 2 フィールドにします（ホストの pause も hold 1 つ）。`FreezeReason` も消します。
  - `LIVE` レジストリは、移さずに消します（D50 で、使い手の K24 が無くなるため）。
  - `RunConfig.workers` は、要求の `Option<NonZeroUsize>` にします。実効のワーカー数は `min(要求か既定, プールに載るノードの数)` で、0 もありえます（反証 A3）。
  - `SlotError` は、呼び出し側が `FieldError` を包む形にします（空文字列を後から埋める処理と、文字列 `NOTHING` の比較を消す）。
  - `NodeFailure` には、ノードの名前を表示します。
  - 使われていない公開名を消し、内部でしか使わないものは `pub(crate)` にします。
- **理由:**
  - 同じ規則（停止中の stop、exception を stop より先に呼ぶこと、manual での Δt=0）が、2 か所に書かれていました。
  - 全ノードが `dedicated_thread=True` の実行は正しい使い方ですが、`NonZeroUsize` にすると K20 で誤って拒まれます（反証 A3）。
  - 統合する前に、Mode の総当たりのテストを分割して固めます（8 節のステップ 1）。

### D31 層をまたいだ名前

- **採用:** 公開している Python 側の名前に、全層をそろえます（`read`、`time_scale`、`advance(seconds)`、`on_update` に対応する `Hook::Update`）。
- **却下:** `scale` にそろえる案（core-boundary）。
- **理由:** grep 1 回で、Python から Rust の実装まで辿れるようにするためです。

### D32 `_native` の表面（改訂）

- **採用:** 14 名にします（`Timer`、`VirtualTimer`、`ReadOnlyTimer`、`Timestamp`、`global_timer`、`Schema`、`Outlet`、`Slot`、`InputGroup`、`Message`、`BufferView`、`Runtime`、`start`、`current`）。
  - `start(nodes, clock, *, manual, workers) -> Runtime` の 1 回の呼び出しで起動するので、「作ったが始まっていない」状態がありません。
  - `nodes` は、Python の NamedTuple `NodeSpec(driver, name, period, dedicated_thread, last_update_ns, state)` のレコードの列です。並べたリストの長さを検査する引数は、表せなくなります。
  - `Runtime` は `step(order)`、`capture(build, timeout)`、`stop()`、`join(timeout) -> bool`、`finished`、`check()`、`last_update_ns()`、`workers`（実効のワーカー数）を持ちます。
  - 値と carry の平らな列をやめ、値の列だけを渡します。
  - ワーカー数の既定は、bindings が Rust の `available_parallelism`（cgroup の CPU quota を数える）から決めます。GIL があれば半分にします。`_available_parallelism` は削除し、テストは `Runtime.workers` で確かめます（反証 B6）。
  - 正常終了の後の capture も、`Runtime.capture` の同じ口で行います。これで、`state_dict` が終了と競合したときの救済が要らなくなります。
- **却下:** ワーカー数の既定を Python の `os.process_cpu_count()` で決める案（前版。quota を数えないので、コンテナの中で quota を超える）。
- **理由:** 今の 19 名には、使われていない名前（`Timer.time()`）、テストでしか使わない名前（`GlobalTimer`、`_available_parallelism`）、並べたリストの長さを検査する引数があります。

### D33 更新手法の置き場所

- **採用:** `src/mujou/pc/` に実装を置き、private なモジュールに分けます。依存するのは `mujou` の公開名だけで、`_core` は pc を知りません（`test_layers.py` で強制）。`mujou.pc.functional` の shim はやめ、実装そのものにします。
- **理由:**
  - 基盤と更新手法を区別する方針（`update-rules.md`）を、物理的な木と一致させます。
  - 今の `_core/pc` が使っている機能は、公開名（Node、Port、DictSpec、TensorSpec、`self.timer`）だけでした（reuse の確認）。移すのは、import の付け替えで済みます。

### D34 pc の部品（改訂）

- **採用:** 公開する部品は `LearnClock` と、その戻り値の型 `Tick` です。
  - `LearnClock(*, period, learn_interval, max_dt)` は `tau` を受け取りません。
    - max_dt の既定（`min(tau.h / 2, 3 * tau.lam)`）は、PC の安定性の事情なので、`Layer` / `Output` が決める
    - `theta=inf` は、`Layer` / `Output` が `learn_interval=math.inf` を渡すことで表す。格子の点が無限の先にあるので、学習の時刻に達しない
  - `tick(dt, time_ns, now_ns) -> Tick(dt, new, learn, elapsed)` は、純粋な遷移です。
    - max_dt での打ち切りもここで行い、打ち切った dt を返す
    - `learn` を返した時点で、学習したものとして Σdt と次の格子を数え直す
    - `learned()` と `elapsed` の property は消す
  - `learn_interval=None`（毎回学習する、連続時間形そのもの）は残します（invariants の reviewer も賛成。テスト 14 で収束を確かめてある）。
  - `functional` には入れず、`mujou.pc` 直下に置きます。
- **却下:**
  - `Latent`、`vjp` のタプル入力、`test_parts`、`Stateful` Protocol（reuse は撤回）
  - `tick` が `tuple[bool, bool]` を返し、学習の後に `learned(now_ns)` を呼ばせる形（前版、反証 2-4）
  - `LearnClock` に `TimeConstants` を渡す形（前版、反証 2-2）
- **理由:**
  - `LearnClock` には、Layer と Output の 2 人の使い手がいます。PC 以外の則（`examples/feedback_alignment.py`）でも、lam と theta の意味のない値を渡さずに使えます。
  - 呼ぶ順の契約は、D34 自身が `Latent` を却下した理由でした。今のコードでは、`learn` が True の更新で、必ず `_learned()` を呼んでいます（学ぶときも、mask が無くて捨てるときも。`layer.py:231`、`output.py:164-166`）。記録を `tick` に移しても、意味は変わりません。
  - 記録する時刻は、学習の後の `now` から、`tick` の `now_ns` に移ります。手動モードでは同じで、実時間の run では、更新の途中で格子の点を越えた場合だけ 1 区間ずれえます。algorithm-reviewer の確認を条件にします（10 節の Q2）。
  - 名前付きの組なら、2 つの bool の位置を取り違えません。
  - テストの `Clock` は `Learner._tick` を写したもので、独立したオラクルになっていませんでした。仕様の式から書いた期待値に置き換えます。

### D35 PC の入口（改訂）

- **採用:** `pc.Chain(*modules, x=, y=, loss=, tau=, learn_interval=, max_dt=, device=)` のクラス 1 つにします。
  - 子のパスは `"input"`、`"layer0"`、…、`"output"` で、`modules[i]` が `layer{i}` です（反証 2-6）。
  - period はすべて `tau.h / 4` とし、docstring に書きます。period の引数は足しません。
  - `learn_interval` の目安は docstring に書きます（D39）。
- **却下:** `pc.Sequential`、`chain()` と `Chain` の 2 つの名前、module のリストを渡す形、`input=` / `target=`、パスを 1 から始める案（前版）。
- **理由:**
  - `Sequential` は、同じ式の中で `nn.Sequential` と並び、読み違えを生みます。
  - `*modules` は `nn.Sequential` と同じ形なので、番号も 0 から始めます。
  - `x` / `y` は `Sample` のキーと一致し、`input` は組み込み関数の名前を隠してしまいます。
  - period を変えたい利用者は、個別のノードで組みます。

### D36 予測を見せる helper（改訂）

- **採用:** `pc.show(run, chain, sample, *, seconds) -> Prediction` にします。docstring の先頭は「学習する前の予測（prequential）を返す。その後、`seconds` の間サンプルを提示し続けてから戻る」とします。
  - 評価は、mask をすべて False にした `Sample` で行います。評価では、伝播に足りる短い `seconds` でよいと書きます（反証 2-5）。
  - `ManualRun` の上では返らない（誰も時計を進めない）ことも書きます。テストは実時間の run で書きます（反証 C4）。
- **却下:** `y=None` で評価に切り替える案、`present`、`unlabeled`。
- **理由:** 教師の有無は、もともと `Sample` の mask が表しています。`y=None` を特別扱いすると、分岐が 1 つ増えます。使い手は `pc_regression.py` と `mnist` の 2 つです。

### D37 Activity（改訂）

- **採用:** `{h, jumps, y, mask, time}` にし、`activity_spec(h, y)` を公開します。`epoch` は `jumps` に改名します。
  - `time` は「`write` が返した時刻の ns（shape ()、int64）」と docstring に書きます（反証 2-8）。
  - `Input` は、ホストが書く前（時刻 `None`）には、`time` が 0 で `mask` がすべて False の Activity を送ります。
  - `LearnClock` の初期値は「まだ入力を見ていない（-1）」なので、この最初の Activity を新しい入力とみなし、学習の格子を 0 から始めます。時刻 0 に書いたサンプルも、同じ格子に乗ります。
  - 学習するかどうかは mask で決まります（反証 S3、`learner.py:95-96`）。`test_a_first_input_written_at_time_zero_is_learned` は残します。
- **理由:**
  - carry の廃止（D7）と、データセットのエポックとの衝突を避けるためです。
  - docs とテストで手書きしていた `DictSpec` が消えます。
  - 番兵の値と初期値の関係を仕様として書けば、「mask が False だから無害」が偶然ではなくなります。

### D38 PC のノードの引数（改訂）

- **採用:**
  - h と y の spec を、1 つのデータ `act: DictSpec[Activity]` で運びます（反証 3-2）。
    - `Input(*, x, y, period, device)` と `Layer` は、出力の `act` を公開する
    - `Layer` / `Output` は、前のノードの `act` を `act=` で受け取る
    - B（バッチの大きさ）は `act` の y から読む
  - B の照合（K28）は `activity_spec` の 1 か所だけにします。`Layer` が自分の出力の `act` を `activity_spec(TensorSpec(mu.shape), y)` で作るので、module の出力の先頭の次元の検査も、同じ 1 か所を通ります。
  - `Input` の period は必須にします。`Input` には tau が無く、既定値のためだけに tau を受け取っていたからです（反証の nit）。`Layer` / `Output` の period の既定は `tau.h / 4` です。
  - `carry_spec`、`batch_of`、`check_batch`、`input_shape=`、`dtype=`、`h_in=` / `y=` / `h_out` は削除します。
  - 状態の属性は、`_` を付けない意味のある名前にし、そのまま状態の木のキーにします。名前は `h`、`lam`、`mu`、`jump`、`jumped`、`jumps`、`clock`、`module` です。
  - `jumped` のフラグは残します（反証 S4）。`h.add_(jump); jump.zero_()` は無条件に行ってよいことにします。
- **理由:**
  - 前版でも、手組みでは層ごとに `h_in=` と `y=` を書いていました。h と y は必ず組で流れるので、組のまま渡せば、引数が減り、照合も構築の時点で自明になります。
  - private な属性名（`_mu`、`_seen_time`）が、事実上の公開の契約になっていました。
  - `jumped` は `jumps` を進める条件で、跳びの大きさとは独立です（`layer.py:259`）。「跳びが非零か」で代用すると、勾配が 0 の学習で次のノードが付け替えなくなり、Jacobi の参照実装と一致しなくなりえます。

### D39 learn_interval の下限（改訂）

- **採用:** 検査はしません（前版の K30 を台帳から外します）。`Chain`、`Layer`、`Output` の docstring に「credit の往復（2 × ホップ数 × period）より長くすることを勧める。数周期以下で発散を観測した」と書きます。今の docstring の「condition of stability」という言い過ぎも直します。
- **却下:** `Chain` の構築時に `ValueError` にする案（前版）。各ノードで検査する案（深さを知らない）。
- **理由:**
  - 根拠は spec 自身の実測です（`specs/pc-module.md:237`）。ResBlock 8 ノードの鎖で、往復は 18 周期でした。
    - `None` と 16 周期は、credit に収束した
    - 4 周期は有界だが、credit の約 4 倍になった
    - 発散したのは、2 周期と 1 周期だけ
  - したがって、2 × 深さ × 周期は安定の境界ではありません。`ValueError` にすると、安定な設定を拒みます。
  - 数えているのはノード数ではなく、ホップ数です（18 = 2 × 9）。
  - 反証 M3 と、algorithm-reviewer の判断待ち（前版の R13）は、これで閉じます。

### D40 `functional`

- **採用:**
  - `vjp` の δ は callable の 1 通りにし、`wrt_input` を削除します。`vjp` と `loss_vjp` の骨格を `_grad` 1 つにまとめます。`rows` は `_rows` にします。
  - `Output` の教師の有無の分岐は、`loss_vjp` の mask の処理に一本化します。残すのは「mask がすべて False なら学習しない」だけです。
  - loss の shape の検査は、`loss_vjp` の 1 か所に置きます（`Output` の構築時の probe も、同じ関数を通る）。
- **理由:**
  - ライブラリ自身は callable しか使っていません。`wrt_input=False` を使う場所もありません。
  - 一本化すると、教師なしのときの `g_in` は、厳密な `zeros_like`（`output.py:162`）ではなく $J^\top 0$ になります。ŷ が有限なら 0 で、ŷ が inf のときだけ NaN を送り返します。ŷ が inf なら、どのみち壊れています。docstring に書き、algorithm-reviewer の確認を条件にします（10 節の Q3）。

### D41 テストの配置（改訂）

- **採用:**
  - src のミラーにします（CLAUDE.md の規則）。中身は、できるだけ公開名を通して書きます。
  - `tests/mujou/nodes.py` に、共通のテスト用ノードを 1 回だけ定義します。
  - 型の横断テストは `test_pyright.py` の 1 本にまとめます。
  - `test_native_time.py` は `test_time.py` に統合します（Timer そのものが native になるため）。
  - `tests/test_claude_hooks.py` は、今の場所に残します（反証 C9）。
- **却下:**
  - 公開モジュールのミラー（dedup は撤回）
  - codec と wiring の単体テストを消す案（reuse）
  - `.claude/hooks` のテストを別の pytest に分ける案（前版）
- **理由:**
  - 単体テストを消すと、壊れたときに、どの層かが分からなくなります。全 dtype のビット一致は、最重要のオラクルです。
  - coverage の対象は、もともと `src/mujou` だけです。hooks のテストを分けると、`--ignore` と CI の変更が要るだけになります。

### D42 examples（改訂）

- **採用:**
  - MNIST は `examples/mnist/` に残します（`feedback_examples_are_guides`）。
  - `leaky.py` と `test_leaky.py` で、書くこととテストすることをつなぎます。
  - `feedback_alignment.py` を足し、PC 以外の、ノードをまたぐ学習則の手本にします（反証 2-2）。
  - `carry.py` は削除します。`pause.py` は `timers.py` に、`torch_node.py` は `online_learning.py` に吸収します。ファイル名に番号は付けません。
  - examples は実行しませんが、`just type` が pyright（standard）で静的に検査します（反証の 4 節）。
- **却下:** `benchmarks/mnist/`（structure、dedup、linus）。番号付きのファイル名（user-first は撤回）。`just docs-build` の中で検査する案。
- **理由:**
  - ユーザーは、MNIST を実用的な手引きとして examples に置くと決めています。番号を付けると、並べ替えのたびにファイル名が変わります。
  - 「examples はテストで実行しない」という規則を保ったまま、API の変更で手本が黙って壊れることを防げます。型の検査は `just type` の責務なので、そこに置きます。

### D43 docs（改訂）

- **採用:** tutorial を 6 ページ、topics を 8 ページ、API を 6 ページにします。
  - 注意書きの正本は `concepts.md` の 1 節に置き、ほかのページからはリンクだけを張ります。
  - fan-out と MultiPort の合流は、`tutorial/1-run.md` の後半に入れます。
  - `topics/update-rules.md` を足し、tutorial 4 の最後からリンクします。
  - tutorial 5 の最後に「次は `examples/mnist`」と書きます。
- **理由:** 学習（mujou の存在理由）が、7 番目と 11 番目のページにしか出てきませんでした。分岐と合流は概念 2 の中心なのに、topics にしかありませんでした。自作の更新手法へ向かう道も、ありませんでした（反証の 4 節）。

### D44 ガード台帳の置き場所

- **採用:** 正本は `specs/scenarios.md` の 1 つです（K ごとに、知らせる場所を 1 か所、テストを 1 本）。利用者に向けては、例外を出す API の docstring の `Raises:` に書きます（mkdocstrings で docs に出る）。docs には、台帳の写しの表を置きません。
- **却下:** `docs/errors.md` の表も正本にして、テストと一致させる案（yagni）。`specs/guards.md` と docs の 2 か所（user-first）。
- **理由:** 表が 3 か所にあると、それ自体が三重の重複になります。

### D45 import の規則

- **採用:** サブモジュールは、すべて明示的に import します（`import mujou.time`、`import mujou.testing`、`import mujou.pc as pc`）。`mujou.typing` は削除します。
- **理由:** 規則を 1 文で言えるようにし、`__init__` が pc を読むことによる循環の芽を摘むためです。

### D46 `memory/agents/`

- **採用:** 作り直しの範囲の外として、触りません。specs からの参照だけを断ちます（数値は `specs/research/` へ移す）。
- **理由:** CLAUDE.md が、中間メモの置き場所として定めている運用です。

### D47 `Timestamp.float()`

- **採用:** 削除し、`float(ts)` に一本化します。`Timestamp(ns: int)` のコンストラクタは、テストで期待値を作るために公開のシグネチャとして書きます（反証 2-8）。

### D48 time scale の API

- **採用:** メソッドの対（`time_scale()` / `set_time_scale(x)`）にします。
- **却下:** property。
- **理由:** frozen な pyclass に setter を持たせないこと、`is_paused()` と書き方をそろえることの 2 点です。

### D49 属性のレジストリ（新規。ユーザーの確認が要る）

- **採用:** インスタンスの `__dict__` を、唯一のレジストリにします。`Component` は `__setattr__` / `__delattr__` を持たず、レジストリも持ちません。
  - `tree.walk` が、run の段 1 で `vars(c)` を代入の順に読みます。
    - Node では、`Port` / `MultiPort` の値がそのノードのポートになる
    - Network では、`Node` / `Network` の値が子になり、`Port` は読み飛ばす
  - K3（ノードがノードを持つ）と K6（list・tuple・dict に入れた Port・Node・Network。1 段だけ見る）も、walk が知らせます。
  - 走った後に代入しても、何も起きません。宣言は、次の run（K14 で拒まれる）まで読まれないからです。
- **却下:** 代入時に `__setattr__` で登録する、今の形（`project_port_registry`、2026-10-17 のユーザーの決定）。
- **理由:**
  - 今は、属性の順序と所属を `__dict__` とレジストリの 2 か所で持ち、代入と `del` のたびに照らし合わせています。1 か所にすると、次のものがすべて消えます。
    - `_NOT_INITIALIZED` の番兵
    - 解除の処理
    - `copy.copy` 対策の置き換え式のレジストリ
    - `is_data_descriptor`
    - `super().__init__()` の前の代入を拒む K2
    - 構築後の変更を拒む `RuntimeError`
  - `on_update` の中の `self.y = ...` まで、Python の `__setattr__` を通らなくなります。今は、すべての代入で isinstance を調べています（`component.py:52-58`、`node.py:525` の注記）。
  - dataclass の Network も、そのまま動きます。
  - 失うのは、K3 と K6 を代入したその行で知らせられることです。代わりに、run の段 1 で `パス.属性名` を名指しして知らせ、その run は同じインスタンスで直せます。
  - 1-6 が指摘した「Port を list に入れると黙って登録されず、K12 が誤った誘導をする」誤りも、K6 が Node にも効くので、段 1 で正しく知らせます。
  - 断られた場合の代案は、今の `__setattr__` を残し、K6 を Node の Port にも広げる形にします。その場合、代入時の検査は、構築中（走った印が無い間）だけ行います。

### D50 復元した実行のタイマー（新規。ユーザーの確認が要る）

- **採用:** `state=` を渡した実行は、保存した `time_ns` と `time_scale` から始まる新しい `Timer` で走ります（`mujou.testing.run` では `VirtualTimer(start=...)`）。
  - `timer=` と `state=` を同時に渡すと K23 にします。
  - タイマーへの時刻の読み込み、`LIVE` レジストリ、前版の K24（ほかの実行が使っているタイマーの巻き戻し）は、どれも削除します。
  - 「復元した実行は `global_timer.pause()` では止まらない。`run.timer.pause()` を使う」と docs に書きます。
- **却下:**
  - 渡された timer か `global_timer` に時刻を読み込み、ほかの実行が live なら拒む案（今の実装、前版）
  - 反証 M2 が求めた形。向きを問わず拒み、検査・読み込み・live の計上を `_native.start` の中で 1 つの lock の下で行う
- **理由:**
  - 前版のままでは、反証が 4 つの穴を指摘しました。
    - 例 7(e) は timer を渡さないので、グローバルの時計を黙って巻き戻すか、ほかの run の有無によって K24 で失敗する（1-3）
    - live を `RunClock` の寿命で数えると、終わった実行が残り続ける（A1）
    - 段 2 で時刻を読み込むと、D2 の「段 2 までは何も変えない」が崩れる（A2）
    - 先へ跳ばしても、ほかの実行の dt が同じように跳ぶ（M2）
  - 復元した実行に専用の時計を与えれば、これらは表せなくなります。時刻の読み込みも数え上げも要りません。
  - inputs も、最初から正しい時刻の軸で刻まれます（D2）。
  - CLAUDE.md の不変の原則の段落にある「既定はグローバルタイマー」に、「復元した実行を除く」を足すことになるので、ユーザーの確認が要ります（R8）。不変の原則の中身（1 つの実行に時間軸は 1 本）は変わりません。
  - 断られた場合の代案は、反証 M2 の形にします。K24 は向きを問わずに拒み、検査・読み込み・live の計上を `_native.start` の中で 1 つの lock の下で行います。live は start で数えて終了で減らし（A1）、印は各ノードの `Driver.start` が付けます（D2）。

### D51 内部の名前の規約（新規）

- **採用:**
  - `_core` の private なモジュールの、`_` の無いモジュールレベルの名前は、パッケージ内部の API です（`spec.DLPACK`、`port.Wire`、`port.running`、`tree.walk`、`node.is_state`、`node.initialized` など）。利用者の公開名は、`mujou`、`mujou.time`、`mujou.testing`、`mujou.pc` が再エクスポートしたものだけです（`test_init.py`）。
  - 公開クラス（Node、Port、MultiPort、Network）の `_` 属性には、そのクラスの外から触りません。実行時の事実を実行側に移した（D10、D17、D49）ので、触る必要がある箇所は 0 です。
  - Node の設定は、名前修飾した 1 属性（`__config`）に置きます。利用者の `self._device` などと衝突しません。
  - フレームワークは、ノードの設定を公開の property（`period`、`device`、`dedicated_thread`、`generator`）だけから読みます。サブクラスが予約名の property を置き換えても、食い違いません。
  - 走った印は、`run.py` の `WeakSet` が持ちます。印を付けるのも見るのも `run.py` だけです。そのため、Node のサブクラスで `__eq__` を定義する（hash できなくなる）ことは、サポートしないものに入れます。
  - src の `# pyright: ignore[reportPrivateUsage]` は 0 件にします（ステップ 11 の完了条件）。
- **却下:** 「friend 関数 0」とだけ書く前版の主張。内部用のメソッドを、`_` を付けずに公開クラスに置く案（利用者の名前空間に漏れる）。予約名の検査（`_check_reserved`）。
- **理由:**
  - pyright の `reportPrivateUsage` は、同じモジュールの関数からの読み書きも警告します（10 節で確かめた）。前版の設計のままでは、6 か所が残っていました（反証 A7）。
  - 事実をそれを使う側に置けば、規約は 1 文で言え、例外がありません。

### D52 ノードをまたいだ状態の共有（新規）

- **採用:** 2 つのノードが同じテンソルか、同じ `state_dict` を持つ物（重みの tying など）を状態として持つことを、`tree.walk` が拒みます（K24）。
  - 判定は、各ノードの `__dict__` の状態の値の `id` を集めた表で行います（`node.is_state` を共有する）。
  - 見つけるのは同じオブジェクトだけです。view による部分的な共有は、サポートしないものに書きます。
- **却下:** サポートしないものに書くだけにする案。
- **理由:**
  - 重みの tying は、PyTorch でよく使う書き方です。2 つのノードから in place で書かれると、free-threaded ではデータ競合になります。「1 つの状態に書き手は 1 人」にも反し、黙って誤った結果になります（G1。反証 S5）。
  - `Chain(m, m, ...)` のように同じ module を 2 回渡す誤りも、同じ検査が捕まえます。
  - 検査は、段 1 の id の集合 1 つです。

### D53 dtype の既定（新規）

- **採用:** `TensorSpec(shape, dtype=torch.float32)` にします。`None` を受け取って `torch.get_default_dtype()` に解決する処理は消します（反証 3-4）。
- **理由:**
  - module のレベルで定義した spec が、import の順と既定の dtype の設定によって変わらないようにするためです。
  - `None` を既定値に置き換えれば、分岐が 1 つ消えます。

### D54 更新手法の案内（新規）

- **採用:**
  - 概念 9 を「学習則はノードの中に書く。BP でも PC-ALM（`mujou.pc`）でも、自作の則でもよい」にします。
  - `topics/update-rules.md` と `examples/feedback_alignment.py` で、ノードをまたぐ自作の則の型を示します。示すのは、双方向の Port、目標の中継、新しさの判定、`LearnClock` です。
  - hook は通常の PyTorch と同じく勾配を記録するので、例と docs からは `torch.enable_grad()` と `# mujou: local-autograd` を外します（`project_grad_on_by_default`）。
- **却下:** 「ノードをまたぐ学習には `mujou.pc` を使う」（前版の概念 9）。
- **理由:**
  - 前版の文は、更新手法を差し替えられるという不変の原則を、利用者に向けた言葉で否定していました（反証 2-2）。
  - 例の `enable_grad` と印は、ユーザーの決定（hook は勾配を記録する。印は `src/mujou/` の中だけで使う）に反していました。反証 2-3 の「hook は `no_grad` の中で走る」という前提は誤りで（`node.py:98-100`。`:393` は `load_state_dict` の中）、直すべきなのは例の方です。

---

## 5b. 提供するシナリオとガードの方針

### 提供するシナリオ（サポートするもの）

| ID | シナリオ | 利用者の操作 |
|---|---|---|
| S1 | ノードを書く | `class X(mujou.Node)`。`__init__` で `super().__init__(...)` を呼ぶ。ポートと状態を属性に代入し、hook を override する。自作の学習則もここに書く |
| S2 | ネットワークを組む | `Network` のサブクラスか `Group(**named)` に属性で並べ、`connect` で繋ぐ（循環してよい） |
| S3 | 走らせる | ホストのスレッドで `mujou.run(root)` を呼ぶ。`with` の中で write / read / subscribe / wait_for / state_dict を使う。subscribe を別のスレッドで回し、main から shutdown してよい。notebook で、段 1 か段 2 で断られた run を直して再実行してよい。終わった後に、ノードの属性（学習した module など）を読んでよい |
| S4 | 時間 | グローバルタイマーか、実行ごとの `Timer` を使う。sleep / pause / time scale |
| S5 | 保存と再開 | `state_dict` → `torch.save` → 同じ形に作り直した新しいインスタンスで `mujou.run(..., state=tree)`（専用のタイマーで続く） |
| S6 | テスト | `mujou.testing.run` と `ManualRun.step(dt, *nodes)` |
| S7 | PC-ALM で学習する | `pc.Chain` / `pc.show`、個別のノード（`act` で繋ぐ）、`pc.LearnClock` と `pc.functional` による自作ノード |
| S8 | ノードの例外 | 全ノードが止まり（`on_exception` → `on_stop`）、ホストに `NetworkError` が届く |
| S9 | CUDA | ノードの `device="cuda"` |

### サポートしないもの（備えも書かない）

- 同じネットワークを、複数のスレッドから同時に run すること
- 走っている run の構造を変えること（走った後の `connect` は、何もしない）
- 手で書き換えた状態の木、mujou 以外が作った状態の木
- ノードの `copy.copy`
- Node のサブクラスで `__eq__` を定義すること（`eq=True` の dataclass を含む。hash できなくなる）
- 型注釈に反する値（pyright か IDE を使うことを前提にし、docs と examples もそう書く）
- ノードが内部を辿って、タイマーやランタイムを操作すること（悪意のある利用者）
- `with` も `shutdown` も使わず、non-daemon のスレッドで subscribe したまま、インタプリタを終えること
- CPU と CUDA 以外の device
- 未接続の `MultiPort` に、ホストが書くこと（K17 で「ホストの入力には Port を使う」と案内する）
- KeyboardInterrupt を受けた位置ごとの特別な扱い（Ctrl-C で `with` を抜ければ shutdown される、という規則だけを持つ）
- 復元した実行を、`global_timer` やほかの実行と同じタイマーで走らせること
- view による、ノードをまたいだ部分的なテンソルの共有（K24 が見つけるのは同じオブジェクトだけ）
- 周期どおりに決定的に回す手動モード

### 残すガード（台帳。これに無い検査は書かない）

| K | シナリオ | 利用者の誤り | 知らせる場所（1 か所） | 残す理由 |
|---|---|---|---|---|
| K1 | S1/S2 | Node / Network のクラス属性に、`Port`・`Node`・`Network` を置く | `Component.__init_subclass__`（クラス定義の時点で `TypeError`） | `__dict__` に入らないので黙って無視され、インスタンスの間で共有される（G1） |
| K2 | S1 | `super().__init__()` を呼ばない | `tree.walk`（run の段 1、`TypeError`「X.__init__ が super().__init__() を呼んでいない」） | 素の `AttributeError` では原因が分からない |
| K3 | S1 | ノードの属性に Node / Network を代入する | `tree.walk`（`TypeError`、パス.属性名） | 子が走らないまま、黙って進む |
| K4 | S9 | `device` が cpu / cuda 以外、または CUDA が使えないのに cuda | `Node.__init__`（`ValueError`） | stream の同期が無いまま走るか、エラーが実行中の遠い場所で出る |
| K5 | S1 | `period` が負、または NaN | `Node.__init__`（`ValueError`、`not period >= 0` の 1 行） | 起動後に、ノードの外で壊れる |
| K6 | S1/S2 | Node か Network の属性に、Port・Node・Network を含む list・tuple・dict を代入する | `tree.walk`（`TypeError`「1 つずつ属性に代入するか、MultiPort を使う」） | 黙って抜け落ち、Port なら後で K12 が誤った誘導をする |
| K7 | S1 | `TensorSpec` にサポート外の dtype を渡す | `TensorSpec.__init__`（`TypeError`、`TensorSpec.dtypes` を示す） | Schema を作る時点という、遠い場所で壊れる |
| K8 | S1 | `DictSpec` のキーが TypedDict と一致しない | `DictSpec.__init__`（`TypeError`） | 黙って欠ける |
| K9 | S1/S3/S5 | send / send_to / write / inputs / initial / 復元する値の構造（Tensor か、dict のキーの過不足）の誤り。shape・dtype の誤り | 構造は `codec.to_core`（`TypeError`）、shape・dtype は Rust の `copy_from`（`ValueError`）。どちらも D5 の label とフィールド付き。inputs・initial・復元は段 2 | 最も多い誤り。それぞれが、自分にしか見えないものを 1 回ずつ見る |
| K10 | S2 | `connect` で spec が逆向きに一致しない | `Port.connect`（`TypeError`、両端を名指しする） | 根本の誤り |
| K11 | S2 | 受け取る `Port` に、別の相手を繋ぐ（どちらの端から繋いでも） | `Port.connect`（`ValueError`「MultiPort を使う」。両端を見る） | recv が曖昧になり、黙って片方だけを読む |
| K12 | S1 | Port の recv / recv_with_time / send / send_to を、所有ノードの hook の外で使う（`__init__`、他のノード、ホスト、run の外、`on_start` で作った Port）。`Node.timer` を hook の外で読む | `port.running()` と、Driver の `wires` の lookup（`RuntimeError`「ポートと timer は、それを持つノードの hook の中で使う。ポートは __init__ で宣言する。ホストは run.read / run.write / run.timer を使う」） | 局所性の不変原則。今の `_bound` と `_check_owner` の 2 か所が、dict の lookup 1 回になる |
| K13 | S2 | 同じオブジェクト（Node・Network・Port）が、木の 2 か所にある | `tree.walk`（`ValueError`、両方のパスを示す） | 1 つのノードが 2 回走るか、1 つのポートに書き手が 2 人できる |
| K14 | S3 | 走ったことのあるインスタンスを、もう一度走らせる | `run` の段 1（`RuntimeError`「新しいインスタンスを作る。終わった実行のノードの属性は読める」、文言は 1 種類） | notebook で起きやすい |
| K15 | S2 | 繋いだ相手のノードを、木に入れ忘れる | `tree.walk`（`ValueError`） | recv が zeros のまま、黙って進む |
| K16 | S3/S4 | hook の中から `mujou.run`、`Run` の操作、`Timer` の制御（sleep / pause / resume / set_time_scale / advance）を呼ぶ | Python の `run._host()` と、bindings の `Timer` の制御メソッド（述語は `current()` の 1 つ、`RuntimeError`） | ワーカーを占有して、ほかのノードの dt を伸ばす。デッドロックする |
| K17 | S3 | 書けない Port（接続済み、`MultiPort`、送るだけ、この run の外）に `write` / `inputs` する。終了後に `write` する | `Run.write` と `run` の段 1（`ValueError` を 1 条件 1 文言で。終了後は `RuntimeError`） | 黙って誰も読まない |
| K18 | S3 | 送らない Port や、ほかの run の Port を read / read_with_time / subscribe / wait_for する | `Run` の lookup の 1 か所（`ValueError`） | 1 回の lookup で済む |
| K19 | S6 | run に無いノードを `step` に渡す | `ManualRun.step`（`ValueError`） | 黙って飛ばされる |
| K20 | S3 | 明示した `workers` が 1 未満 | `run` の段 1（`ValueError`） | 印を付けた後で native の変換に失敗すると、インスタンスを使い切ってしまう |
| K21 | S5 | 形の違うネットワークに木を復元する（`timer` の項目、ノードのパス、ポートの属性、スロットのキーの集合） | `state.restore`（`run` の段 1、`ValueError`、足りない項目と余分な項目を列挙する） | S5 で最もありうる誤り |
| K22 | S5 | クラスを変えた後で、古い状態を読む（キーの過不足、テンソルの shape の違い） | 既定の `Node.load_state_dict`（`ValueError`、パスとキー付き。`NetworkError` の `__cause__` として届く） | `copy_` は broadcast で黙って通ってしまう |
| K23 | S4/S5 | `timer=` と `state=` を同時に渡す | `run` の段 1（`ValueError`「復元した実行は、保存した時刻から始まる専用のタイマーで走る。timer を省く」） | 渡したタイマーを使うと思い込んだまま、別の時計で走る |
| K24 | S1/S2 | 2 つのノードが、同じテンソルか同じ `state_dict` を持つ物を状態に持つ（重みの tying） | `tree.walk`（`ValueError`、両方のパス.属性名） | free-threaded でデータ競合になり、黙って誤った結果になる |
| K25 | S7 | `TimeConstants` の h が有限で > 0 でない、lam または theta が > 0 でない（inf は可） | `TimeConstants.__post_init__`（`ValueError`） | NaN になるか、学習しない |
| K26 | S7 | `learn_interval` が None でも > 0 でもない（inf は可）、`max_dt` が > 0 でない、`period > max_dt` | `LearnClock.__init__`（`ValueError`） | 学習しないか、時間を黙って捨てる |
| K27 | S7 | 平均を取る損失（`nn.MSELoss()` など）を渡す | `functional.loss_vjp` の (B,) の検査（`Output` の構築時の probe も同じ関数を通る。`ValueError` で `pc.mse` を案内する） | 黙って誤った勾配になる |
| K28 | S7 | h と y の先頭の次元（B）が違う（module の出力の先頭の次元が B でない場合を含む） | `activity_spec`（`ValueError`） | mask の行が、黙って broadcast される |
| K29 | S7 | `vjp` の δ と μ の shape が違う、`mse` の y と ŷ の shape が違う | `functional.vjp` / `mse`（`ValueError`） | broadcast が黙って通る |
| K30 | S1 | `register_state` に、まだ代入していない名前を渡す（綴りの誤り、property） | `Node.register_state`（`AttributeError`「代入してから登録する」） | `__dict__` から決めるので、綴りを誤った名前は黙って状態から漏れる |

ガードではない意味論として、次のものを残します。

- `TimeoutError`（timeout を過ぎた）、`NetworkError`（例外による終了と、その後のホストの呼び出し）
- `Run.__exit__` が shutdown すること
- `wait_for` が、条件を満たす前に実行が終わったときの `RuntimeError`。終了後の `ManualRun.step` の `RuntimeError`
- `VirtualTimer.advance` を pause 中に呼んだときの `RuntimeError`（タイマーの契約）
- `BufferView.__dlpack__` の `BufferError`（DLPack の約束）
- 抽象メソッドは `abc.abstractmethod` で表す（`raise` を書かない）
- Rust の `Reentrant` の検査（コアの内部のデッドロックを防ぐもの。Python からは K16 が先に止めるので届かない）
- FFI の健全性の検査（`SlotError::Internal` などの内部エラー。文言は作り込まない。利用者に届いたら mujou のバグ）

### 消すガード

| 消すもの（今の場所） | 判定 |
|---|---|
| `Component.__setattr__` / `__delattr__` の登録と、代入時の検査（super の前の代入、ノードへのノードの代入、構築後の変更） | D49（K2・K3・K6 を段 1 の walk へ移す） |
| `_NOT_INITIALIZED`、`copy.copy` 対策の置き換え式のレジストリ、`is_data_descriptor` による property の除外 | G6、G5 |
| `period` の bool / 有限 / 584 年で飽和する扱い、`dedicated_thread` の型（`node.py:189-195`） | G3（負と NaN だけを K5 に残す） |
| `checked_period` / `checked_dedicated` / `_check_reserved` による、予約名の検査（`inspect.getattr_static`） | G6（フレームワークは公開の property だけを読む。D51） |
| `generator` / `timer` を `super().__init__()` の前に読んだときの `RuntimeError`、`super()` を呼ばない場合の救済 | G5、G6（K2 だけにする） |
| `register_state` が property を `ValueError` で拒むこと | K30 に吸収する（property は `__dict__` に無い） |
| 既定の `load_state_dict` の `_mapping` / `_check_module` / stride 0 の検査 / 未登録のテンソルから属性を作る処理 / 事前の一括検査 | G2、G6（K22 だけにする） |
| `Port._init` の `TypeError` 7 個（spec の型、send_carry の型、multi と recv_carry の組み合わせなど） | G3、G6（表せない形にする） |
| `connect` の「自分自身に繋ぐ」「両端とも spec が無い」 | D6 の dict が吸収する、G5 |
| `connect` の「build 済みの木のポート」の `RuntimeError` | G5（走った後の構造の変更はサポートしない。再実行は K14 が拒む） |
| `Port._bound` の `RuntimeError` と、`binding._check_owner` の二重化 | K12 の dict の lookup 1 つにする |
| `Channel.__init__` / `Graph.__init__` / `Run.__init__` の「直接作るな」 | G5（直接作っても、どこにも登録されないだけ。docstring に「connect が作る」「run() が返す」と書く） |
| `Channel.other` の `ValueError`、`send_to` の channel の `ValueError`、`carried` の key の `TypeError` / `KeyError`、「受けない / 送らない」の `TypeError` | G2、G3（型では `Never`）、carry の廃止 |
| `Binding._message` の carry の 3 通りの分岐と、`TypeError` 3 つ | carry の廃止 |
| `spec.py` の `_check_tensor`（Python での shape・dtype の検査）、layout の検査 | Rust の K9 に一本化する、G2（`__dlpack__` が拒む） |
| `TensorSpec` の dims の検査（int か、1 以上か）、`rand` の浮動小数点の検査、`seed` と `generator` を両方渡した `ValueError` | G2、API の変更（`rand(*, generator=)`） |
| `DictSpec` の is_typeddict / optional / annotation / キーが無い の検査 | G2、G3（K8 のキーの一致だけを残す） |
| `graph.py` の「build 済みの木に属する」の 2 種類の文言 | D3 で状態ごと消す |
| `graph.py` の「すでに所有されている port」 | K13（同じオブジェクトが 2 か所）に吸収する |
| `graph.py` の「ノードの無い木」 | G5（空の実行は何もしない） |
| `graph.py` の「公開した Port の所有者が内側にいない」 | D9 で消える |
| `graph.py` の「relay 元が同じノードに無い」 | carry の廃止 |
| `graph.py` の `_check_declarations` | K1 に移す |
| `network.py` の Group の名前の検査 5 種類と、型の検査 | G5（提供シナリオでは、リテラルのキーワードしか書かない） |
| `run.py` の `start_run` の workers の型と上限（`sys.maxsize`）、components の型 | G3（K20 の値だけを残す） |
| `run.py` の `state_dict` の timeout の有限性、`session._wait` の負と NaN、終了後に使わない timeout の検査 | D28（`wait::to_duration` の自然な変換） |
| `run.py` の `wait_for` の `TimeoutError` の言い換え | G2 |
| `session.py` の `_writable` の 4 つの理由と、`recv_carry` を持つ Port への write の拒否 | K17 の 1 条件 1 文言にまとめる、carry の廃止 |
| `session.py` の `state_dict` が「待つ間に終わった」ときの `RuntimeError` の救済 | D32（Rust の capture が両方の場合を扱う） |
| `session.start` の except の分類、`lifetime.running(Session)` による掃除、`set_timer(None)` | D2、D11 |
| `state.py` の `read_tree` の int / bool / タプルの長さの検査 | G5（K21 のキー集合の一致だけ） |
| `timer.py` の `stopped` による到達不能化（`itertools.repeat`、`object.__new__`） | G5、D11 |
| `lifetime.py` の監視 daemon スレッド | G5（検証してから。D29） |
| BUILD_LOCK、`start_once` のロック、KeyboardInterrupt の位置ごとの扱い | G5 |
| ほかの live な実行があるタイマーへの、時刻の読み込みの拒否（`freeze_for_startup`、`LIVE`） | D50（表せなくする。K23 だけを残す） |
| Rust の `check_start`（4 本の並べたリストの長さ、period 0） | G4（`Vec<NodeSpec>`、`NonZero`） |
| Rust の `Runtime::step` の `NotManual` / index の範囲外、`InputGroup.write` の別のグループと重複、`borrow_values` の個数、`Timestamp(ns=)` の範囲、`TimerError::NotVirtual` | G4（内部エラーに降格する。`VirtualTimer` 型で表す） |
| `pc` の `TimeConstants` の型 / bool の検査、`Learner` の tau の型・bool・module の device・出力が Tensor か | G3、G2（probe の forward で torch が出す） |
| `pc` の `batch_of` / `check_batch`、`carry_spec` の batch の照合 | K28（`activity_spec` の 1 か所）にまとめる |
| `pc.functional._check_delta`（tensor の δ の検査）、`learn_` の長さの検査 | δ の Tensor 版を削除、`zip(strict=True)` |
| `Output` の構築時と毎回の、loss の検査の二重化 | K27 の 1 か所にする |
| `learn_interval < 2·depth·period` の `ValueError`（前版の K30） | D39（推奨値として docstring に書く） |
| stubgen の `EXCEPTIONS` と、`test_native_stub` の例外のずれの検査 | D27 |

---

## 6. 現在 → 理想の対応表

### 6.1 `src/mujou`（ファイル）

| 現在 | 理想 | 操作 |
|---|---|---|
| `__init__.py` | `__init__.py` | 公開名を 16 から 13 にする。サブモジュールは読み込まない。`__version__` は importlib.metadata から取る |
| `time.py`（関数 7 個 + 再エクスポート） | `time.py` | 関数を削除し、`VirtualTimer` と `global_timer` を再エクスポートに足す |
| `testing.py` | `testing.py` | `run` / `ManualRun` の再エクスポートだけにする。`VirtualTimer` は `mujou.time` へ |
| `typing.py` | — | 削除（`MessageSpec` はトップレベルへ） |
| `py.typed` | `py.typed` | 変更なし |
| `pc/__init__.py` | `pc/__init__.py` | 実装を `pc/_*.py` から引く。公開名は 15 |
| `pc/functional.py`（shim） | `pc/functional.py` | 実装そのものにする |
| `_core/__init__.py` | `_core/__init__.py` | 変更なし |
| `_core/_native.pyi` | `_core/_native.pyi` | 再生成（14 名） |
| `_core/binding.py` | `_core/port.py`（`Wire` と recv / send の本体）、`_core/driver.py`（`wires` の dict）、`_core/codec.py`（`views_value` / `message_value`） | 分けて削除する。`AnyPort` / `AnySlot` / `as_spec` / `relay_from` / carry は削除 |
| `_core/component.py` | `_core/component.py` | K1 の `__init_subclass__` だけにする。`_ports` / `_children` / `__setattr__` / `__delattr__` / `ports_of` / `children_of` / `_register` / `_unregister` / `is_data_descriptor` / `_check_not_built` / `_NOT_INITIALIZED` / `isinstance(_native.Node)` は削除 |
| `_core/graph.py` | `_core/tree.py` | `walk(root) -> Tree` だけを残し、`__dict__` の走査にする。`Graph` / `_BUILT` / `_Tree` / `BUILD_LOCK` / `start_once` / `unbuild_tree` / `build_node` / `is_built` / `ports` / `_named` / `_where` / `_check_declarations` は削除する（名前の付け方は `tree.name` と label の規則へ、宣言の検査は K1 へ） |
| `_core/hooks.py` | bindings の `CURRENT` と `_native.current()` | 削除 |
| `_core/lifetime.py` | `_core/run.py` の atexit | 統合する（`Stoppable` / `running` / 監視スレッドは削除） |
| `_core/network.py` | `_core/network.py` | `Network`、`Group(**named)` だけにする。`Network.build` / `_graph` / `unbuild` / `RunGroup` / `component_module` への代入は削除 |
| `_core/node.py` | `_core/node.py`（利用者の API、設定、状態の規則、`is_state`、`initialized`）、`_core/driver.py`（Context、CUDA の stream と device、hook の後の同期、capture と restore の入口） | 分割する。`set_timer` / `checked_period` / `checked_dedicated` / `capture_state` / `_check_reserved` / クラス属性の既定値 / `_on_enter` / `_after_update` / `_capture_state` / `_restore_state` / `_STATE_NAMES` のレジストリ / `_state_names` 系の関数 / `_mapping` / `_check_tensor` / `_check_module` は削除または移動 |
| `_core/port.py` | `_core/port.py` | `Port[S, R]`、`MultiPort[S, R]`、`Channel`、`Wire`、`running()` にする。`PrevPort` / `NextPort` / `Single` / `Multi` / overload の複製 / friend 関数 7 個 / carry は削除 |
| `_core/run.py` | `_core/run.py` | `Session` を吸収する。`start_run(kind, manual)` は `_start(manual)` にする。`_host` を 1 つにし、走った印の WeakSet を持つ。`ManualRun` は非総称にする |
| `_core/session.py` | `_core/wiring.py`（配線）、`_core/run.py`（ライフサイクルとホストの I/O）、`_core/state.py`（取得と復元の段取り）、`_core/driver.py`（CUDA の同期） | 分けて削除する。`_settle` / `_ended` / `_ended_timer` / `_stop_timer` / `_ended_tree` は削除 |
| `_core/spec.py` | `_core/spec.py`（公開の型と `DLPACK`）、`_core/codec.py`（`fields`、`split`、`host_tensors`、`join`） | 分割する。`Spec` alias、`_check_tensor`、`typed_dict`、`dtype=None` の解決は削除。`DLPACK_DTYPES` は `DLPACK` に改名する。`TensorSpec.dtypes` を足す |
| `_core/state.py` | `_core/state.py` | `capture` / `restore` にする。`StateDict` / `SlotSpecs` / `make_tree` / `read_tree` / `_slot_restores` / `add_safe_globals` は、新しい形に置き換える |
| `_core/timer.py` | native の `Timer` / `VirtualTimer` / `ReadOnlyTimer` / `global_timer` | 削除（`host_only` は K16 として bindings へ） |
| `_core/pc/__init__.py` | — | 削除 |
| `_core/pc/carry.py` | `pc/_messages.py` | 移動と改名。`Carried` / `carry_spec` / `batch_of` / `check_batch` は削除。`activity_spec(h, y)` を公開し、K28 を置く |
| `_core/pc/functional.py` | `pc/functional.py` | 移動。δ は callable だけにし、`wrt_input` と `_check_delta` は削除、`_grad` にまとめ、`rows` は `_rows` にする |
| `_core/pc/input.py` | `pc/_input.py` | 移動。`x=` / `y=` / `period=`（必須）、`act`、時刻 None のときは time 0 |
| `_core/pc/layer.py` | `pc/_layer.py` | 移動。`act=`、`LearnClock` を合成で持ち、on_update を段ごとに分ける。`jumped` のフラグは残す |
| `_core/pc/learner.py` | `pc/_clock.py`（`LearnClock`、`Tick`） | 継承から合成へ。`Learner` は削除 |
| `_core/pc/losses.py` | `pc/_losses.py` | 移動 |
| `_core/pc/output.py` | `pc/_output.py` | 移動。`act=`、教師の有無の分岐を一本化する |
| `_core/pc/time_constants.py` | `pc/_time_constants.py` | 移動。型の検査を外す |
| — | `_core/errors.py`、`_core/codec.py`、`_core/tree.py`、`_core/wiring.py`、`_core/driver.py`、`pc/_clock.py`、`pc/_chain.py` | 新規 |

### 6.2 公開名

| 現在 | 理想 |
|---|---|
| `mujou.Channel`（`ends`、`other`） | 変更なし（直接作ることを拒む検査は消す） |
| `mujou.DictSpec`（`specs`、`typed_dict`） | `DictSpec`（`specs`。`typed_dict` は削除） |
| `mujou.Graph`（`nodes`、`port_path`） | 削除 |
| `mujou.Group(*components, **named)` | `Group(**named)` |
| `mujou.Network`（`build`） | `Network()`（`build` は削除。子は `__dict__` から読む） |
| `mujou.NetworkError` | 変更なし（Python で定義し、`__module__` は `mujou`） |
| `mujou.NextPort` | `MultiPort`（相手ごとに受けるとき）、または `Port`（送るだけのとき） |
| `mujou.Node`（hook 8 つ、`register_state`、`state_dict` / `load_state_dict`、`device` / `period` / `dedicated_thread` / `generator` / `timer`） | 純 Python。hook 6 つと状態の 2 メソッドを分けて書く。`state_dict` は `dict` を返し、乱数生成器の状態を含む。`load_state_dict` は strict。`timer` は hook の中だけ |
| `mujou.Node` の private hook（`_on_enter` / `_after_update` / `_capture_state` / `_restore_state`） | `_core.driver.Driver` の `start` / `restore` / `update` / `capture`（enter は無い） |
| `mujou.PauseTimeoutError` | 組み込みの `TimeoutError` |
| `mujou.Port[S, R, M, CS, CR](send, recv, multi, initial, send_carry, recv_carry)` | `Port[S, R](*, send, recv, initial)` |
| `Port.connect` / `recv` / `send` / `channels` / `send_spec` / `recv_spec` | 変更なし（connect は同じ相手に同じ Channel を返す。channels は connect の順） |
| `Port.recv_with_time` | 時刻が `Timestamp \| None` になる |
| `Port.send_to` | `MultiPort.send_to` だけにする |
| `Port.has_received` / `carried` / `send_carry_spec` / `recv_carry_spec` / `carry_source` / `multi` | 削除 |
| `mujou.PrevPort` | `Port` |
| `mujou.Run`（`timer`、`write`、`read`、`read_with_time`、`subscribe`、`wait_for`、`state_dict`、`shutdown`、`join`） | 変更なし（時刻は `Timestamp \| None`、`state_dict` は dict を返し、timeout の既定は inf） |
| `mujou.StateDict`（`to`、`cpu`） | 削除（素の dict） |
| `mujou.TensorSpec`（`dtype=None`、`rand(seed=, generator=)`） | `TensorSpec(shape, dtype=torch.float32)`、`rand(*, generator=)`、`dtypes` を足す |
| `mujou.__version__` | 変更なし（importlib.metadata） |
| `mujou.run(component, *components, ...)` | `run(root, ...)`（`state` があれば専用の Timer、`timer` との同時指定は K23） |
| `mujou.time`（`import mujou` で読み込まれる） | 明示的に import する |
| `mujou.typing` | 削除 |
| — | `mujou.MultiPort`、`mujou.MessageSpec`（新規、移動） |
| `mujou.time.now` / `sleep` / `pause` / `resume` / `is_paused` / `get_time_scale` / `set_time_scale` | 削除 |
| `mujou.time.Timer()`（`get_time_scale`） | `Timer(*, start=, time_scale=)`（`time_scale()`） |
| `mujou.time.ReadOnlyTimer`（`get_time_scale`。直接は作れない） | `ReadOnlyTimer(timer)`（`time_scale()`。実体は `RunClock`） |
| `mujou.time.Timestamp`（`float()`） | `Timestamp(ns)`（`.float()` は削除） |
| — | `mujou.time.VirtualTimer(*, start=)`、`mujou.time.global_timer` |
| `mujou.testing.run(component, *components, timer=, ...)` | `run(root, *, timer: VirtualTimer \| None, inputs, state)` |
| `mujou.testing.ManualRun[T]`（`step(*nodes)`） | `ManualRun`（`step(dt, *nodes)`） |
| `mujou.testing.VirtualTimer` | `mujou.time.VirtualTimer` |
| `mujou.typing.MessageSpec` | `mujou.MessageSpec` |
| `mujou.typing.Single` / `Multi` | 削除 |
| `mujou.pc.Activity`（`h`、`epoch`） | `Activity`（`h`、`jumps`、`y`、`mask`、`time`） |
| `mujou.pc.Carried` / `carry_spec` | 削除 |
| `mujou.pc.Input(shape, carry=, period=, dtype=, device=)` | `Input(*, x, y, period, device="cpu")`、`act` |
| `mujou.pc.Layer(module, input_shape=, carry=, tau=, learn_interval=, period=, max_dt=, dtype=, device=)` | `Layer(module, *, act, tau, learn_interval=None, period=None, max_dt=None, device="cpu")`、`act` |
| `mujou.pc.Output(module, loss=, input_shape=, carry=, ...)` | `Output(module, *, act, loss, tau, ...)` |
| `mujou.pc.Prediction` / `Sample` / `TimeConstants` / `mse` / `cross_entropy` | 変更なし |
| `mujou.pc.functional.vjp` / `loss_vjp` / `relax_` / `learn_` | 変更なし（vjp の引数を絞る） |
| — | `pc.Chain`、`pc.show`、`pc.LearnClock`、`pc.Tick`、`pc.activity_spec`（新規） |

### 6.3 `_native` の名前

| 現在 | 理想 |
|---|---|
| `__version__` | 削除（importlib.metadata） |
| `timer`（`GlobalTimer` のインスタンス） | `global_timer`（普通の `Timer`） |
| `GlobalTimer` | 削除 |
| `Timer`（`time` / `time_ns` / `now` / `sleep` / `time_scale` / `set_time_scale` / `pause` / `resume` / `is_paused` / `state_dict` / `load_state_dict` / `use_virtual_clock` / `advance` / `reader`） | `Timer(*, start, time_scale)`（`now` / `sleep` / `pause` / `resume` / `is_paused` / `time_scale` / `set_time_scale`） |
| — | `VirtualTimer(*, start)`（`advance`） |
| `TimerReader` | `ReadOnlyTimer(timer)`（`RunClock`） |
| `Timestamp` | 変更なし（`float()` は削除。module は `mujou.time`） |
| `Schema(values, carry)` | `Schema(fields)` |
| `Outlet`（`send(values, relay=)`、`send_to`、`channel(k)`、`outbox`） | `Outlet`（`send(values)`、`send_to(k, values)`、`outbox`、`channels`、label と clock を受け取る） |
| `Slot` / `InputSlot`（`read`、`wait_newer`、`restore`、`new_carry`） | `Slot`（`read`、`wait_newer`、`restore(values, time_ns)`） |
| `InputGroup` | 変更なし（`write` は `Timestamp` を返す） |
| `Message`（`buffers`、`carry`、`timestamp`、`sent`） | `Message`（`values`、`timestamp: Timestamp \| None`） |
| `Carry` | 削除 |
| `BufferView` | 変更なし |
| `Node` | 削除（Driver を呼ぶ） |
| `Runtime(nodes, names, timer)` + `start(7 kwargs)` + `is_finished` / `request_stop` / `shutdown` | `start(nodes, clock, *, manual, workers) -> Runtime`（`step` / `capture` / `stop` / `join` / `finished` / `check` / `last_update_ns` / `workers`） |
| `NetworkError` / `PauseTimeoutError`（`create_exception!`） | Python の `mujou._core.errors.NetworkError` と、組み込みの `TimeoutError` |
| `_available_parallelism` | 削除（既定は bindings が決め、テストは `Runtime.workers` で見る） |
| — | `current()` |

### 6.4 `crates/`

| 現在 | 理想 | 操作 |
|---|---|---|
| `mujou-core/src/lib.rs` | `lib.rs` | モジュールの宣言を変える |
| `mujou-core/src/time.rs` | `time.rs` | `Source` enum、start と scale からの生成、hold を 1 系統に、`RunClock`。`time()`、状態の読み込み（`load_state`）、`use_virtual_clock` は削除し、`Wait` / `wait_until_ns_or` / `interrupt_waiters` / `hold` は `pub(crate)` にする |
| `mujou-core/src/dlpack.rs` | `dlpack/ffi.rs`、`dlpack/import.rs`、`dlpack/export.rs` | 分割する（`FieldError` を返す） |
| `mujou-core/src/slot.rs`（1613 行） | `schema.rs`、`buffer.rs`、`slot.rs` | 分割する。`Slot` と `InputSlot` を統合し、carry と名前の組み立てを削除し、`Message.time_ns: Option<u64>` にする。時刻は `RunClock` から取る |
| `mujou-core/src/runtime/mod.rs` | `runtime/mod.rs` | `NodeSpec`、`RunConfig`、`Hook`、既定実装。Shared の impl をここに集める。`LAST_NODE`、`Entry`、`NodeHooks::enter`、`freeze_for_startup` の live の検査、`SharedError` の再公開、`Runtime::failure` は削除 |
| `mujou-core/src/runtime/node.rs` | `runtime/lifecycle.rs` | 統合して削除する |
| `mujou-core/src/runtime/pool.rs` | `runtime/sched.rs`（advance と record は lifecycle へ） | 改名して分割する。時刻は `RunClock` から取る |
| `mujou-core/src/runtime/control.rs` | `runtime/control.rs` | `stop()` で `clock.end()`。終了後の capture は、呼び出し側のスレッドで lock を取って行う |
| `mujou-core/src/runtime/freeze.rs` | `Control` の `Option<PauseHold>` の 2 フィールド | 削除（`LIVE` / `Live` / `lock_live` / `FreezeReason` も削除） |
| `mujou-core/src/runtime/error.rs` | `runtime/error.rs` | `Hook`、ノードの名前 |
| `mujou-core/tests/runtime.rs`（1846 行） | `tests/runtime/{main,lifecycle,period,sched,capture,failure,manual,restore}.rs` | 分割する |
| `mujou-core/tests/time.rs` | `tests/time.rs` | RunClock と、start からの生成の項目を足す |
| `mujou-python/src/lib.rs` | `lib.rs` | 宣言だけにする（`wait_interruptibly` は `wait.rs` へ） |
| `mujou-python/src/time.rs` | `time.rs` | `VirtualTimer`、`ReadOnlyTimer`、`global_timer`、K16、module="mujou.time"。`GlobalTimer` / `time()` / `time_ns()` / `use_virtual_clock` / `state_dict` / `load_state_dict` は削除。秒の変換は `wait.rs` へ |
| `mujou-python/src/slot.rs` | `slot.rs` | 値だけの列、`Slot` を 1 型、`Carry` は削除 |
| `mujou-python/src/runtime.rs` | `runtime.rs`、`errors.rs` | `Node` pyclass / `CACHED_HOOKS` / `attribute()` / `create_exception!` / `_available_parallelism` / Context の管理は削除。`start`、`PyDriver`、`CURRENT`、`current` を足す。ワーカー数の既定（GIL で半分）は残す |
| `mujou-python/src/dlpack.rs` | `dlpack.rs` | 変更なし（legacy の再試行は、torch の最低版を決めてから判断する） |
| `mujou-python/src/hint.rs` | `hint.rs` | `Views` を消し、残りを最小にする |
| — | `mujou-python/src/wait.rs`、`errors.rs` | 新規 |
| `mujou-stubgen/src/main.rs` | `main.rs` | `EXCEPTIONS` / `add_exceptions` を削除する |
| `Cargo.toml` / `Cargo.lock` / 各 crate の `Cargo.toml` | 同じ | 依存の変化を反映する |

### 6.5 `tests/`

| 現在 | 理想 |
|---|---|
| `tests/mujou/conftest.py`（`virtual_timer`、`virtual`、`started`、`start`、`start_manual`、`step`） | `conftest.py`（`virtual`、`start`、`start_manual`）。グローバルの仮想時計の fixture は削除し（`use_virtual_clock` と `load_state_dict` が無くなる）、`step` は `ManualRun.step` に置き換える |
| `tests/mujou/test_init.py`（mujou と pc の公開名） | `test_init.py`（全公開パッケージ） |
| `tests/mujou/test_testing.py` | `test_testing.py`（`step(dt)` の順序・最初の dt・period の無視・K19。公開名の部分は `test_init.py` へ） |
| `tests/mujou/test_time.py` | `test_time.py`（`_core/test_timer.py` と `test_native_time.py` を統合し、TestPause を 1 つにする。global_timer は同一性と smoke だけ） |
| `tests/mujou/test_typing.py` | `test_pyright.py` |
| `tests/mujou/_core/test_node_typing.py`、`test_port.py` の TestPortTypes / TestCarryTypes | `test_pyright.py` |
| `tests/mujou/_core/test_component.py` | `_core/test_component.py`（K1。登録・解除・shallow copy・property のテストは削除） |
| `tests/mujou/_core/test_network.py` | `_core/test_network.py`（Group、dataclass の Network）と `_core/test_tree.py`（K2, K3, K6, K13, K15, K24。並行 build、exposed、ノードなしのテストは削除） |
| `tests/mujou/_core/test_node.py` | `_core/test_node.py`（K4, K5, K22, K30、状態の規則、乱数生成器）、`_core/test_driver.py`（スレッドの設定、CUDA）。`test_works_without_calling_the_base_init` と bool の検査のテストは削除 |
| `tests/mujou/_core/test_port.py` | `_core/test_port.py`（実行時の振る舞い、K10〜K12、同じ相手への 2 回目の connect。TestCarrySpecs、TestCarried は削除） |
| `tests/mujou/_core/test_run.py`（2310 行） | `_core/test_run.py`、`_core/test_state.py`、`_core/test_wiring.py`、`_core/test_driver.py`（CUDA、hook の順序）。並行 run、断られた run の「同じ形での」再実行、TestInterpreterExit の subscribing thread、workers の bool と上限、timeout の検査、ほかの run のタイマーへの復元は削除 |
| `tests/mujou/_core/test_spec.py` | `_core/test_spec.py` と `_core/test_codec.py`（fields / split / join） |
| `tests/mujou/_core/test_state.py`（2 つ組の偽の木） | `_core/test_state.py`（実際の形の往復、K21） |
| `tests/mujou/_core/test_timer.py` | `test_time.py` と `_core/test_run.py`（run とタイマー、K16、K23） |
| `tests/mujou/_core/test_native.py`、`test_native_stub.py` | `_core/native/test_module.py`（例外の検査は削除） |
| `tests/mujou/_core/test_native_slot.py` | `_core/native/test_slot.py`（dtype 表は `spec.DLPACK` を import し、古い `mujou._spec` のコメントを消す） |
| `tests/mujou/_core/test_native_runtime.py` | `_core/native/test_runtime.py`（不正な引数のテストは削除。`_available_parallelism` の代わりに `Runtime.workers`） |
| `tests/mujou/_core/test_native_time.py` | `test_time.py` |
| `tests/mujou/_core/pc/test_carry.py` | `pc/test_messages.py` |
| `tests/mujou/_core/pc/test_functional.py` | `pc/test_functional.py` |
| `tests/mujou/_core/pc/test_input.py` / `test_output.py` / `test_losses.py` / `test_time_constants.py` | `pc/` の同名のファイル（型と bool の検査のテストは削除） |
| `tests/mujou/_core/pc/test_layer.py`（Reference、Clock） | `pc/_reference.py`、`pc/test_layer.py`、`pc/test_clock.py`（Clock の写しは、仕様から書いた期待値に置き換える） |
| `tests/test_claude_hooks.py` | 変更なし |
| — | `tests/test_layers.py`、`tests/mujou/nodes.py`、`_core/test_codec.py`、`_core/test_tree.py`、`_core/test_wiring.py`、`_core/test_driver.py`、`pc/test_clock.py`、`pc/test_chain.py`、`pc/test_show.py`（新規） |

### 6.6 docs（`docs/en/*` と `docs/ja/*` の両方。API のページは en だけ）

| 現在 | 理想 |
|---|---|
| `index.md` | `index.md`（「run by the tests」の誤記を直し、学習するノードを見せる） |
| `concepts.md` | `concepts.md`（9 つの概念と、規則の正本） |
| `getting-started/installation.md` | `install.md` |
| `getting-started/quickstart.md` | `tutorial/1-run.md`（後半に fan-out と合流） |
| `user-guide/index.md` | 削除（nav が地図を兼ねる） |
| `user-guide/writing-a-node.md` | `tutorial/2-node.md`（基本）、`topics/ports.md`、`topics/threads-and-devices.md` |
| `user-guide/building-a-network.md` | `tutorial/1-run.md`（基本）、`topics/ports.md`（carry の節は削除） |
| `user-guide/running.md` | `topics/host-io.md`、`topics/threads-and-devices.md`（workers と GIL） |
| `user-guide/group.md` | `topics/groups.md` |
| `user-guide/state.md` | `tutorial/6-save-and-resume.md`（木の形は `api/run.md` の docstring を参照する。復元した実行のタイマー） |
| `user-guide/time.md` | `topics/time.md`（3 列の表は消え、単位の表を置く） |
| `user-guide/torch.md` | `tutorial/4-learn-in-a-node.md`、`topics/threads-and-devices.md` |
| `user-guide/errors.md` | `topics/errors.md` |
| `user-guide/testing.md` | `tutorial/3-test.md` |
| `user-guide/pc.md` | `tutorial/5-learn-with-pc.md`、`topics/pc.md`（Input の「Nothing」の誤りを直す） |
| — | `topics/update-rules.md`（新規） |
| `examples/index.md` | `examples.md`（全ファイルを載せる） |
| `api/index.md` | 削除（nav にする） |
| `api/node.md`、`api/port.md`、`api/typing.md` | `api/node-and-ports.md` |
| `api/network.md`（run / Run / StateDict が同居している） | `api/network.md`（Network、Group）と `api/run.md`（run、Run、NetworkError） |
| `api/errors.md` | `api/run.md` |
| `api/time.md`、`api/testing.md`、`api/pc.md` | 同名（中身を新しい名前にする） |
| `docs/assets/*`、`javascripts/mathjax.js`、`stylesheets/extra.css`、`overrides/main.html`、`hooks/anchors.py` | 変更なし |
| `mkdocs.yml` | 新しい nav。stub に docstring が無いという古いコメントを消す |

### 6.7 examples

| 現在 | 理想 |
|---|---|
| `README.md` | 規約を更新する（`test_leaky.py` の扱いと、`just type` による静的な検査を書く） |
| `quickstart.py` | `quickstart.py`（README の最小例と同じにする） |
| `leaky_chain/leaky.py`、`leaky_chain/main.py` | `leaky.py`（sys.path を使わない 1 ファイル。状態は `__init__` で作る） |
| `testing_a_node.py` | `test_leaky.py`（`leaky.py` の Leaky を import する） |
| `online_learning.py`、`torch_node.py` | `online_learning.py`（素の PyTorch の autograd、`last` と `elapsed` を `register_state`） |
| `pause.py`、`timers.py` | `timers.py` |
| `carry.py` | 削除 |
| `branching.py`、`group.py`、`subscribe.py`、`errors.py`、`save_and_restore.py` | 同名（新しい API にする） |
| `mnist/main.py` | `mnist/main.py`（`pc.Chain`）と `mnist/watch.py` |
| `mnist/bp.py` | `mnist/bp.py` と `mnist/metrics.py` |
| `mnist/data.py` | `mnist/data.py`（pc に依存しない） |
| — | `pc_regression.py`、`feedback_alignment.py`（新規） |

### 6.8 specs / memory / その他

| 現在 | 理想 |
|---|---|
| `specs/README.md` | 目次と、知識の置き場所の表 |
| `specs/overview.md` | 「pc は未実装」を直す |
| `specs/nodes.md` | Driver、hook の順、状態の規則。救済と bool の検査、内部の hook、`CACHED_HOOKS` の記述は削除 |
| `specs/execution-model.md` | RunClock、lifecycle の遷移表、状態の木の意味、復元した実行のタイマー。Run の実装の箇条と、ReadOnlyTimer の到達不能化は削除 |
| `specs/network.md` | Port / MultiPort、接続の dict、`__dict__` の走査、起動の 3 段。Run のメソッドの表、エラー文言、並行 run、KeyboardInterrupt は削除 |
| `specs/messages.md` | spec、codec、DLPack。carry の節は削除 |
| `specs/native-core.md` | 境界の契約、ABI のレコード、`python-environment.md` を吸収、memory/agents への参照を断つ |
| `specs/python-environment.md` | `native-core.md` |
| `specs/torch.md` | `architecture.md`（モジュールの表）、`nodes.md`（CUDA）、`native-core.md`（性能） |
| `specs/update-rules.md` | 書き直す（pc は `_core` の外の利用者） |
| `specs/pc-module.md`（445 行） | `pc.md`（現在の設計）と `research/pc-experiments.md`。実装の順番とテスト計画は削除 |
| `specs/pc-alm-continuous-time.md` | 同名（刻みの分割と、BP 一致のテストについての古い記述を直す） |
| `specs/open-decisions.md` | 本当に未決定のものだけ（確認待ちの 7 件は、一括で確認してから消す） |
| — | `specs/architecture.md`、`specs/scenarios.md`、`specs/research/benchmarks.md` |
| `README.md` | 「PC-ALM 未実装」を直し、最小例を quickstart に一致させる |
| `CONTRIBUTING.md` | 層のテストと、examples の静的な検査について書き足す |
| `pyproject.toml` | `--doctest-modules` を削除する。`numpy` は実行時の依存から外し、examples の dependency group へ移す。pyright の executionEnvironments に examples（standard）を足す |
| `justfile` | `type` で examples も検査する。stub の埋め込みスクリプトは `scripts/stub.py` に出す |
| `.github/workflows/test.yml` | 変更なし（`just type` が examples を含む） |
| `benchmarks/mn_scheduler.py` | 新しい `_native.start` に合わせる。recv / send のオーバーヘッドの測定（`benchmarks/ports.py`）を足す |
| `memory/*.md`、`.claude/skills/*`、`CLAUDE.md` | 9 節 |
| `memory/agents/` | 触らない |

---

## 7. 利用者のコード例（理想の API）

### (a) 最小のネットワークを動かす

```python
from typing import override

import torch

import mujou
from mujou import Network, Node, Port, TensorSpec

VECTOR = TensorSpec((3,))


class Affine(Node):
    """Sends a * x + b."""

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        self.a, self.b = a, b                     # 設定。状態ではない
        self.prev = Port(recv=VECTOR)
        self.next = Port(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(self.a * self.prev.recv() + self.b)


class Pipeline(Network):
    def __init__(self) -> None:
        super().__init__()
        self.scale = Affine(2.0, 0.0)
        self.shift = Affine(1.0, 1.0)
        self.scale.next.connect(self.shift.prev)  # spec が合わなければ、この行で TypeError（K10）
        self.prev, self.next = self.scale.prev, self.shift.next  # ただの参照として公開する


net = Pipeline()
x = torch.tensor([1.0, 2.0, 3.0])
with mujou.run(net) as run:
    run.write({net.prev: x})  # 形が違えば "scale.prev: expected shape (3,) ..."（K9）
    y, t = run.wait_for(net.next, lambda y: torch.equal(y, 2 * x + 1), timeout=5.0)
print(y, t)
```

### (b) 自分のノードを書く（連続時間、状態、fan-out と合流、学び続けるノード）

```python
import math
from typing import TypedDict, override

import torch
from torch import nn

from mujou import DictSpec, MultiPort, Network, Node, Port, TensorSpec

SCALAR = TensorSpec(())


class Leaky(Node):
    """dy/dt = (x - y) / tau, integrated exactly over each dt."""

    def __init__(self, tau: float, *, period: float = 0.01) -> None:
        super().__init__(period=period)
        self.tau = tau                                  # 設定。状態ではない
        self.y = SCALAR.zeros(device=self.device)       # テンソルの属性は自動で状態に入る
        self.prev = Port(recv=SCALAR)
        self.next = Port(send=SCALAR)                   # 送るだけなので何本でも繋げる

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        a = math.exp(-dt / self.tau)
        self.y = a * self.y + (1 - a) * self.prev.recv()
        self.next.send(self.y)                          # send はコピーするので、送った後に変えてよい


class Sum(Node):
    def __init__(self) -> None:
        super().__init__()
        self.prev = MultiPort(recv=SCALAR)    # 同じ種類の相手を束ねる: recv() は dict[Channel, Tensor]（connect の順）
        self.next = Port(send=SCALAR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(sum(self.prev.recv().values(), SCALAR.zeros()))


class Fork(Network):
    """Filters one signal fast and slow, and adds the two back."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = Leaky(0.01)
        self.fast, self.slow = Leaky(0.05), Leaky(0.5)
        self.sum = Sum()
        self.pre.next.connect(self.fast.prev)          # 送るだけの Port は何本でも繋げる
        self.pre.next.connect(self.slow.prev)
        self.fast.next.connect(self.sum.prev)
        self.slow.next.connect(self.sum.prev)
        self.prev, self.next = self.pre.prev, self.sum.next


class Example(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


EXAMPLE = DictSpec(Example, x=TensorSpec((32, 4)), y=TensorSpec((32, 1)))


class OnlineLinear(Node):
    """Follows the gradient flow dW/dt = -dL/dW / tau, learning from each new batch."""

    def __init__(self, tau: float) -> None:
        super().__init__(period=0.005, seed=0)
        self.tau = tau                                  # 設定
        self.f = nn.Linear(4, 1)                        # state_dict を持つので自動で状態に入る
        nn.init.normal_(self.f.weight, std=0.1, generator=self.generator)  # seed が効くのは generator だけ
        self.last = -1                                  # 最後に学んだバッチの時刻（ns）
        self.elapsed = 0.0                              # 前に学んでからの秒
        self.register_state("last")                     # 実行中に変わる値は登録する
        self.register_state("elapsed")
        self.prev = Port(recv=EXAMPLE)
        self.loss = Port(send=SCALAR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.elapsed += dt
        ex, t = self.prev.recv_with_time()
        if t is None or t.ns == self.last:              # まだ届いていない、または新しくない
            return
        self.last = t.ns
        loss = (self.f(ex["x"]) - ex["y"]).pow(2).mean()   # hook は通常の PyTorch と同じく勾配を記録する
        loss.backward()
        with torch.no_grad():                           # パラメータの更新はグラフに入れない
            for p in self.f.parameters():
                p.sub_(p.grad, alpha=self.elapsed / self.tau)   # 更新量は経過時間で決める（ステップ数ではない）
                p.grad = None
        self.elapsed = 0.0
        self.loss.send(loss.detach())
```

`elapsed` が長いと 1 歩が大きくなります。そのため `examples/online_learning.py` では、経過時間を上限の刻みに分けて進めます。

### (c) PC で MNIST 風に学習する

```python
import torch
from torch import nn

import mujou
import mujou.pc as pc
from mujou import TensorSpec

B = 64
shown = 0.2                                   # 1 バッチを見せる時間（タイマーの秒）
tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=1.28)   # 1 回の学習は学習率 ≈ learn_interval / theta
net = pc.Chain(
    nn.Sequential(nn.Linear(784, 256), nn.ReLU()),
    nn.Sequential(nn.Linear(256, 256), nn.ReLU()),
    nn.Linear(256, 10),
    x=TensorSpec((B, 784)),
    y=TensorSpec((B,), torch.int64),          # B はここで 1 回だけ書く
    loss=pc.cross_entropy,                    # nn.CrossEntropyLoss() なら構築時に ValueError（K27）
    tau=tau,
    learn_interval=0.75 * shown,              # credit の往復（2 × 3 ホップ × tau.h / 4）より十分長い
)

prequential = correct = 0
with mujou.run(net) as run:
    for x, y in train_batches(B):             # 学習しながら予測する: 学習モードは無い
        labeled = torch.ones(B, dtype=torch.bool)
        p = pc.show(run, net, pc.Sample(x=x, y=y, mask=labeled), seconds=shown)
        prequential += int((p["y_hat"].argmax(1) == y).sum())   # 学習する前の予測で測る
    for x, y in test_batches(B):              # mask がすべて False なら θ は動かない（h と λ は動く）
        unlabeled = torch.zeros(B, dtype=torch.bool)
        p = pc.show(run, net, pc.Sample(x=x, y=y, mask=unlabeled), seconds=0.02)  # 伝播に足りればよい
        correct += int((p["y_hat"].argmax(1) == y).sum())
    lam = run.state_dict()["nodes"]["layer0"]["state"]["lam"]   # パスは input, layer0, layer1, output
trained = net.layer0.module                   # 終わった後は、ノードの属性を読んでよい
```

自作の PC ノードは、次の手順で書きます。

1. `self.clock = pc.LearnClock(period=self.period, learn_interval=..., max_dt=...)` を属性に置く（それだけで状態に入る）
2. 毎回の更新で `t = self.clock.tick(dt, int(act["time"]), self.timer.now().ns)` を呼ぶ
3. `pc.functional.relax_(..., t.dt, tau)` で活動を動かす
4. `t.learn` なら、`pc.functional.learn_(params, g_theta, t.elapsed, tau, batch)` で学習する

ほかの pc のノードと繋ぐポートの spec には、前のノードの `act`（または `pc.activity_spec(h, y)`）を使います。PC 以外の則でノードをまたいで学ぶ書き方は、`examples/feedback_alignment.py` と `topics/update-rules.md` にあります。

### (d) テストを書く

```python
# examples/test_leaky.py（uv run pytest examples/test_leaky.py）
import math

import torch

import mujou.testing
from leaky import Leaky


class TestLeaky:
    def test_follows_the_exact_solution(self) -> None:
        tau, h, steps = 0.1, 0.001, 100
        node = Leaky(tau)
        expected = 0.0
        with mujou.testing.run(node, inputs={node.prev: torch.tensor(1.0)}) as run:
            for _ in range(steps):
                run.step(h)   # タイマーを h 進めてから全ノードを木の順に 1 回ずつ更新する（period は無視する）
                expected += (1 - math.exp(-h / tau)) * (1.0 - expected)
            y, t = run.read_with_time(node.next)
        assert t is not None and t.ns == steps * 1_000_000
        torch.testing.assert_close(y, torch.tensor(expected))
```

### (e) 保存と復元

```python
import torch

import mujou
import mujou.time

net = Pipeline()
with mujou.run(net) as run:
    run.write({net.prev: torch.ones(3)})
    run.timer.sleep(1.0)
    tree = run.state_dict()                   # 素の dict。テンソルは CPU のコピー
torch.save(tree, "ckpt.pt")

# {"timer": {"time_ns": ..., "time_scale": 1.0},
#  "nodes": {"scale": {"state": {"generator": ...}, "last_update_ns": ...,
#                      "ports": {"prev": {"in": E}, "next": {"out": E, "to": {"shift.prev": E}}}}, ...}}

again = Pipeline()                            # 実行は 1 回きり。同じ形の新しいインスタンスで再開する
tree = torch.load("ckpt.pt", weights_only=True)   # 素の型だけなので safe globals は要らない
with mujou.run(again, state=tree) as run:     # 形が違えば、足りない項目と余分な項目を列挙して ValueError（K21）
    assert run.timer is not mujou.time.global_timer   # 保存した時刻から始まる、この実行だけの Timer
    run.timer.sleep(1.0)                      # 保存した時刻の続きから進む

# mujou.run(net) をもう一度呼ぶと
#   "Pipeline has already run; create a new instance (its attributes stay readable)"（K14）
# mujou.run(again2, state=tree, timer=some_timer) は ValueError（K23）
```

### (f) 時刻

```python
import mujou
import mujou.time

timer = mujou.time.Timer()                    # この実行だけの時間軸（0 から始まる）
with mujou.run(net, timer=timer) as run:
    run.timer.set_time_scale(10.0)            # 実時間 1 秒あたり、タイマーは 10 秒進む
    run.timer.sleep(1.0)                      # タイマーの 1 秒（実時間では 0.1 秒）
    t = run.timer.now()                       # write や read_with_time が返す時刻と同じ軸

# 既定のグローバルタイマーで動く 2 つの実行の状態を、同じ時刻で取る（復元した実行は専用のタイマーなので含めない）
with mujou.run(Pipeline()) as run_a, mujou.run(Pipeline()) as run_b:
    mujou.time.global_timer.pause()
    a, b = run_a.state_dict(), run_b.state_dict()
    assert a["timer"]["time_ns"] == b["timer"]["time_ns"]
    mujou.time.global_timer.resume()
```

ノードの中では、`self.timer.now()` で時刻を読むだけです。`self.timer` は `ReadOnlyTimer` なので、制御するメソッドを持ちません。timeout は実時間の秒で、`sleep` と `dt` はタイマーの秒です。

---

## 8. 移行計画

### 共通の規則

- 各ステップの後で、`just test-fast` と `just type` が通る状態を保ちます。commit の前には `just run` を通します（CLAUDE.md の規則）。
- Rust を変えるステップでは `just stub` を実行し、生成した `_native.pyi` を同じ commit に入れます。
- 公開 API を変えるステップでは、その API を使う examples と docs を機械的に直します（全体の組み直しはステップ 12 で行う）。
- テストの移動とコードの変更は、別の commit に分けます。移すテストは、最後の置き場所へ 1 回だけ移します。
- 後で消す暫定コードは書きません。各ステップは、それだけで完結させます。
- ブランチは `refactor/2026-10-20/restructure` を使います。1 commit に 1 つの関心事を入れます。
- 10 節の確認事項に関わるステップは、ユーザーの確認が取れるまで着手しません（4 は R6 と R8、5 は R3、6 は R1、8 は R2、9 は R4、R5、R7）。

### ステップ 0: 確認事項を一括で聞く

| 項目 | 内容 |
|---|---|
| (a) 目的 | 10 節の R1〜R8 を、推奨案とともにユーザーに聞く（メインエージェントが中継する）。R1 の答えで、ステップ 6 の形が決まる |
| (b) 触るファイル | なし |
| (c) 完了条件 | 回答を `memory/` に、1 決定 1 ファイルで記録する |
| (d) 並列 | 1、2 と並列に進めてよい |

### ステップ 1: 安全網を張り、基準値を記録する（振る舞いは変えない）

| 項目 | 内容 |
|---|---|
| (a) 目的 | 作り直しの回帰を捕まえるオラクルと、性能の基準値を先に固める。後で書き直すテストは、まだ動かさない |
| (b) 触るファイル | <ul><li>`tests/mujou/nodes.py` を新設する</li><li>`crates/mujou-core/tests/runtime.rs` を `tests/runtime/*.rs`（Mode の総当たりを含む）に分割する</li><li>`tests/test_layers.py` を、今の違反を許可リストに持つラチェットとして入れる</li><li>`pyproject.toml` から `--doctest-modules` を外す</li><li>今の API の上で、新しいオラクルを足す（`tests/mujou/_core/pc/test_layer.py` に「mask がすべて False の間は全ノードの θ がビット一致」）</li><li>`benchmarks/ports.py`（recv / send）を足し、`benchmarks/mn_scheduler.py`、MNIST の 1 epoch、GPU のノードの capture の値を `specs/research/benchmarks.md` に記録する</li></ul> |
| (c) 完了条件 | <ul><li>`just test`（全部）と `just type` が通る</li><li>分割の前後で、Rust のテスト関数の数が一致する</li><li>`test_layers.py` が、今の許可リストで通る</li><li>基準値が記録されている</li></ul> |
| (d) 並列 | 0、2 と並列に進めてよい（3 は、この後） |

### ステップ 2: Rust コアの木を一方向にする（Python から見える ABI は変えない）

| 項目 | 内容 |
|---|---|
| (a) 目的 | dlpack と slot の循環を断ち、`slot.rs` を schema / buffer / slot に分ける。`Slot` と `InputSlot` を、コアの中で 1 つの型にする。`SlotError` は `FieldError` を包む形にする |
| (b) 触るファイル | `crates/mujou-core/src/{lib.rs, schema.rs, buffer.rs, slot.rs, dlpack/ffi.rs, dlpack/import.rs, dlpack/export.rs}`、`crates/mujou-python/src/{slot.rs, dlpack.rs}` |
| (c) 完了条件 | <ul><li>`just test-rs`、`just test-fast`、`just type` が通る</li><li>`just stub` の差分が無い</li><li>`native/test_slot` の全 dtype のビット往復が通る</li></ul> |
| (d) 並列 | 0、1、3 と並列に進めてよい（ファイルが別） |

### ステップ 3: Rust ランタイムの状態機械を 1 本にする

| 項目 | 内容 |
|---|---|
| (a) 目的 | <ul><li>`node.rs` と `pool.rs` を、`lifecycle.rs` と `sched.rs` に統合する</li><li>`enum Hook`、`Vec<NodeSpec>`、`NodeHooks` の既定実装を入れる（`enter` はステップ 5 で Driver と一緒に消す）</li><li>Shared の impl を `mod.rs` に集める</li><li>`NodeFailure` にノードの名前を出す。使われていない公開名を消す</li><li>bindings の `Runtime` を、レコードを受け取る `_native.start` にする。`stop` / `join` / `finished` / `check` / `workers` を返す（`session.py` を合わせる）</li><li>実効のワーカー数を `min(要求か既定, プールのノード数)` にし、0 を許す。既定は bindings の `available_parallelism`（GIL で半分）にする</li></ul> |
| (b) 触るファイル | `crates/mujou-core/src/runtime/*`、`crates/mujou-core/tests/runtime/*`、`crates/mujou-python/src/runtime.rs`、`src/mujou/_core/session.py`、`benchmarks/mn_scheduler.py`、`tests/mujou/_core/test_native_runtime.py`、`tests/mujou/_core/test_run.py`（`_available_parallelism` の代わりに `Runtime.workers`） |
| (c) 完了条件 | <ul><li>ステップ 1 で固めた `tests/runtime/*` が、全 Mode で通る</li><li>全ノードが `dedicated_thread=True` の実行が走る</li><li>`benchmarks/mn_scheduler.py` の更新回数 / 秒と p99 の遅延が、基準値の範囲に収まる（専用スレッドの fast path が落ちたら、`sched.rs` の中だけで戻す）</li><li>`just test-fast`、`just type`、`just stub` が通る</li></ul> |
| (d) 並列 | 1 の後。2 と並列に進めてよい |

### ステップ 4: 時間を RunClock の 1 本にする（確認 R6、R8 の後）

| 項目 | 内容 |
|---|---|
| (a) 目的 | <ul><li>`RunClock` をランタイムの唯一の時刻源にする（dt、`last_update_ns`、スロットの時刻。end は停止の要求か、失敗の検出の時刻）</li><li>`Timer(*, start, time_scale)`、native の `VirtualTimer`（`Timer` のサブクラス）、普通の `Timer` の `global_timer`、`time_scale` の名前をそろえる</li><li>`state=` を渡した実行は、保存した時刻から始まる新しい Timer で走らせる。`timer=` との同時指定は K23 にする。タイマーへの時刻の読み込み、`LIVE`、`freeze.rs`（`Control` の 2 フィールドへ）を削除する</li><li>bindings の thread-local `CURRENT` と `_native.current()` を入れる（ここではノードのオブジェクトを持つ）。`Timer` の制御メソッドと `run._host()` の K16 を、これで行う</li><li>Python の `_core/timer.py`、`hooks.py`、`stopped`、`set_timer`、`_settle`、`_ended_timer`、`_stop_timer` を消す</li><li>`mujou.time` の関数 7 個を消し、`VirtualTimer` を `mujou.time` へ移す。`ManualRun.step(dt, *nodes)` にする</li><li>`conftest.py` の、グローバルの仮想時計の fixture を消す</li></ul> |
| (b) 触るファイル | `crates/mujou-core/src/{time.rs, slot.rs}`、`runtime/{mod,lifecycle,sched,control}.rs`、`runtime/freeze.rs`（削除）、`crates/mujou-core/tests/{time.rs, runtime/*}`、`crates/mujou-python/src/{time.rs, wait.rs, runtime.rs, lib.rs}`、`src/mujou/{time.py,testing.py}`、`_core/{timer.py（削除）, hooks.py（削除）, session.py, run.py, node.py, binding.py, state.py}`、`tests/mujou/{conftest.py, test_time.py, test_testing.py}`、`tests/mujou/_core/{test_timer.py, test_run.py, test_state.py}`、`mujou.time.*` を呼んでいる examples と docs |
| (c) 完了条件 | <ul><li>新しいテストが通る（「正常終了の後の木の時刻 ≥ すべてのスロットの時刻と最後の更新」「pause 中の取得で、複数の実行が同じ時刻」「復元した実行は保存した時刻から専用の Timer で続き、global_timer は動かない」「timer と state の同時指定は K23」）</li><li>hook の中の `Timer.sleep` を拒むテスト（今の `test_timer.py:271,287,309,323`）が通り続ける</li><li>`grep -r "mujou.time.sleep\|get_time_scale\|use_virtual_clock" src examples docs` が 0 件</li><li>`just test-fast`、`just type`、`just stub` が通る</li></ul> |
| (d) 並列 | 3 の後。5 とは直列に進める（どちらも bindings の `time.rs`・`runtime.rs`・`lib.rs` と stub に触る） |

### ステップ 5: Rust と Python の境界を細くする（確認 R3 の後）

| 項目 | 内容 |
|---|---|
| (a) 目的 | <ul><li>`errors.py` と `import_exception!`、`TimeoutError` への統一</li><li>`Driver` を入れる。`_native.Node` / `CACHED_HOOKS` / `attribute()` / private hook 4 つ、core の `NodeHooks::enter` / `LAST_NODE` / `Entry` を削除する。`CURRENT` が Driver を持つようにする。contextvars の Context は Driver が持つ</li><li>`codec.py` を作り、shape・dtype の検査を、Rust の label 付きの検査だけにする（dtype の表は `spec.DLPACK`）</li><li>stubgen の `EXCEPTIONS` を削除し、`__version__` を importlib.metadata から取る</li><li>carry の引数は今のまま触らない（ステップ 6 で 1 回だけ変える）</li></ul> |
| (b) 触るファイル | `crates/mujou-core/src/runtime/{mod,lifecycle,sched}.rs`、`crates/mujou-python/src/{errors.rs, runtime.rs, slot.rs, time.rs, lib.rs}`、`crates/mujou-stubgen/src/main.rs`、`src/mujou/_core/{errors.py, driver.py, codec.py, node.py, spec.py, binding.py, session.py, component.py}`、`src/mujou/__init__.py`、関係するテスト |
| (c) 完了条件 | <ul><li>`native/test_runtime` の「hook ごとの FFI は 1 回」が、全 Mode で通る</li><li>`test_driver`（super を呼ばない override でも CUDA の同期が飛ばない、contextvars がノードについて回る）が通る</li><li>`test_codec` の全 dtype が通る</li><li>recv / send のベンチマークの悪化が、基準値から 5% 未満（R9）。CUDA のノードの hook ごとの設定のコストを記録する（R13）</li><li>`just test-fast`、`just type`、`just stub` が通る</li></ul> |
| (d) 並列 | 4 の後 |

### ステップ 6: コアの carry を廃止する（確認 R1 の後）

| 項目 | 内容 |
|---|---|
| (a) 目的 | PC は `y`、`mask`、`time` を `Activity` のフィールドとして運び、コアから carry を消す。R1 を断られたら、D7 の代案を同じ位置で実装する |
| (b) 触るファイル | `crates/mujou-core/src/{schema.rs, buffer.rs, slot.rs}`、`crates/mujou-python/src/slot.rs`、`src/mujou/_core/{port.py, binding.py, session.py, state.py, codec.py}`、`_core/pc/{carry.py, input.py, layer.py, output.py, learner.py}`、`examples/carry.py`（削除）、関係する docs |
| (c) 完了条件 | <ul><li>`tests/mujou/_core/pc` の全オラクル（Jacobi との 1e-12 一致、不動点での BP 一致、α=0、有限差分、復元後に同じ軌道を辿ること、ノードの分け方を変えても同じ結果）が、期待値を書き換えずに通る</li><li>MNIST の 1 epoch の時間の悪化が 5% 未満。密な y（回帰）の悪化を記録する（R12）</li><li>`grep -ri carry src crates` が 0 件</li><li>`just test-fast`、`just type`、`just stub` が通る</li></ul> |
| (d) 並列 | 5 の後。終わったら 7 と 10a を並列に進めてよい |

### ステップ 7: 「まだ届いていない」を時刻 None で表す

| 項目 | 内容 |
|---|---|
| (a) 目的 | <ul><li>`Message.time_ns: Option<u64>` にして、`sent` を吸収する</li><li>`recv_with_time` / `read_with_time` / `subscribe` / `wait_for` の時刻を `Timestamp \| None` にする。`has_received` を削除する</li><li>今の状態の木の時刻の項目を、`int \| None` にする（木の全体の書き直しはステップ 9 で 1 回だけ）</li><li>`Input` は、時刻 None のときに time 0 を送る</li></ul> |
| (b) 触るファイル | `crates/mujou-core/src/{buffer.rs, slot.rs}`、`crates/mujou-python/src/slot.rs`、`src/mujou/_core/{port.py, binding.py, session.py, state.py}`、`_core/pc/input.py`、`examples/online_learning.py`、関係するテスト |
| (c) 完了条件 | <ul><li>`native/test_slot` の「送られたものだけが sent」のテストを、時刻 None の形に書き換えて通す。wait_newer と InputGroup のバージョンが変わらない（R15）</li><li>`test_a_first_input_written_at_time_zero_is_learned` が通る</li><li>`just test-fast`、`just type`、`just stub` が通る</li></ul> |
| (d) 並列 | 6 の後。10a と並列に進めてよい（ファイルが別: 7 は `input.py`、10a は `layer.py` / `output.py` / `learner.py`） |

### ステップ 8: Port を整理する（確認 R2 の後）

| 項目 | 内容 |
|---|---|
| (a) 目的 | <ul><li>`Port[S, R]` / `MultiPort[S, R]`（キーワード専用）、相手 → Channel の dict、同じ相手への 2 回目の connect、両端を見る K11 を入れる</li><li>`binding.py` を削除し、`Wire` を `port.py` に置いて Driver の `wires` が持つようにする（K12 は lookup）</li><li>`PrevPort`、`NextPort`、`Single`、`Multi`、`multi=`、`mujou.typing` を削除し、`MessageSpec` をトップレベルへ移す</li></ul> |
| (b) 触るファイル | `src/mujou/_core/{port.py, binding.py（削除）, driver.py, session.py, state.py}`、`src/mujou/typing.py`（削除）、`src/mujou/__init__.py`、`_core/pc/*`、すべての examples と docs、テスト |
| (c) 完了条件 | <ul><li>`test_pyright` で、`MultiPort.recv()` の dict 型と、`Port[Never, T].send` の禁止が通る（R14）</li><li>`test_port`（fan-out、K10〜K12、同じ相手への 2 回目、自己接続）が通る</li><li>影響する examples を手で実行する</li><li>`just test-fast`、`just type`、`just stub` が通る</li></ul> |
| (d) 並列 | 7 と 10a の後 |

### ステップ 9: 組み立てと実行の木を作る（確認 R4、R5、R7 の後）

| 項目 | 内容 |
|---|---|
| (a) 目的 | <ul><li>`__dict__` を唯一のレジストリにする: `Component` を K1 だけにし、`tree.py`（K2, K3, K6, K13, K15, K24）を作る</li><li>起動の 3 段。`Run` が `Session` を吸収し、`wiring.py` と `state.py`（新しい素の dict の木。ここで 1 回だけ作る）を作る。走った印は WeakSet（K14）</li><li>`Graph` / `build` / `unbuild` / `RunGroup` / `BUILD_LOCK` / `graph.py` / `session.py` を削除する。`run(root)`、`Group(**named)` にする</li><li>`load_state_dict` を strict にする。乱数生成器を状態に入れ、クラスそのものを状態から除く。`register_state` は K30</li><li>`lifetime.py` を atexit だけにする（サブプロセスのテストを通してから）</li></ul> |
| (b) 触るファイル | `src/mujou/_core/{component.py, graph.py→tree.py, network.py, node.py, driver.py, run.py, session.py（削除）, state.py, lifetime.py（削除）, wiring.py}`、`src/mujou/{__init__.py, testing.py}`、関係するテスト、examples、docs |
| (c) 完了条件 | <ul><li>新しいテストが通る。対象は次のとおり<ul><li>段 1・段 2 で断られた run は、同じインスタンスで再実行できる</li><li>段 3 の後は K14</li><li>木の形と `"to"` のキー</li><li>dataclass の Network が動く</li><li>atexit だけで止まる（サブプロセス、3.14t と GIL ありの両方）</li></ul></li><li>`test_layers.py` の許可リストから、component↔graph、network→component の書き換え、port↔binding が消える</li><li>`just test-fast`、`just type` が通る</li></ul> |
| (d) 並列 | 8 の後 |

### ステップ 10: `mujou.pc` を作り直す

| 項目 | 内容 |
|---|---|
| 10a (a) 目的 | `LearnClock(*, period, learn_interval, max_dt)` と `Tick` を、`Learner` から切り出す（`_core/pc` の中で）。`theta=inf` は `learn_interval=inf` で表す。`test_clock.py` を仕様の式から書く |
| 10a (b) 触るファイル | `src/mujou/_core/pc/{learner.py, layer.py, output.py}`、`tests/mujou/_core/pc/*` |
| 10a (c) 完了条件 | <ul><li>pc の全オラクルが、期待値を書き換えずに通る</li><li>algorithm-reviewer が Q1 と Q2 を確認する</li><li>`just test-fast`、`just type` が通る</li></ul> |
| 10a (d) 並列 | 6 の後。7 と並列に進めてよい |
| 10b (a) 目的 | <ul><li>`src/mujou/pc/` へ移し、公開 API だけに依存させる（`test_layers` で pc→`_core` を禁止する）</li><li>`act` の受け渡し、`Input(period=)` を必須にする、`jumps`、`activity_spec` の公開（K28）、`carry_spec` などの削除</li><li>`functional` を単純にし、`Output` の分岐を一本化する（algorithm-reviewer が Q3 を確認する）</li><li>`Chain`（`layer0` から）と `show` を足し、テストを `tests/mujou/pc/` へ移す</li></ul> |
| 10b (b) 触るファイル | `src/mujou/_core/pc/*`（削除）、`src/mujou/pc/*`、`tests/mujou/_core/pc/*` → `tests/mujou/pc/*` |
| 10b (c) 完了条件 | <ul><li>pc の全オラクルが通る</li><li>`test_chain`（Chain が手組みと `torch.equal` で一致する）が通る</li><li>`test_show` が、実時間の run で通る</li><li>mask がすべて False の間の θ の不変が通る</li><li>`just test-fast`、`just type` が通る</li></ul> |
| 10b (d) 並列 | 9 の後 |

### ステップ 11: ガードの最後の棚卸し

| 項目 | 内容 |
|---|---|
| (a) 目的 | `specs/scenarios.md` の台帳 K1〜K30 に無い検査を消し、K ごとにテストを 1 本置く。各ステップで触ったコードのガードは、そのステップで直しておく。ここは最後の掃除 |
| (b) 触るファイル | `src/` の全体、`specs/scenarios.md`、テスト |
| (c) 完了条件 | <ul><li>`src` の `reportUnnecessaryIsInstance`、`not checked at runtime`、`pyright: ignore[reportPrivateUsage]` が 0 件</li><li>`src` のすべての `raise` が、台帳の K か意味論（5b 節の一覧）に対応する（対応表を PR に載せる）</li><li>`just test-fast`、`just type` が通る</li></ul> |
| (d) 並列 | 12、13 と並列に進めてよい |

### ステップ 12: docs と examples を組み直す

| 項目 | 内容 |
|---|---|
| (a) 目的 | 4.5 節と 4.6 節の木にする。`feedback_alignment.py` と `pc_regression.py` を書く。README を直す。`just type` で examples を検査する |
| (b) 触るファイル | `docs/en/**`、`docs/ja/**`、`mkdocs.yml`、`examples/**`、`README.md`、`pyproject.toml`（numpy を examples の group へ、pyright の examples の設定）、`justfile` |
| (c) 完了条件 | <ul><li>`just docs-build`（strict）が通る</li><li>`just type` が examples を含めて通る</li><li>全 examples を手で実行する（`feedback_alignment.py` と `pc_regression.py` は、損失が下がるのを確かめる）</li><li>en と ja のページ構成が一致する</li><li>`carry.py` が残っていない</li><li>examples の表に全ファイルが載っている</li></ul> |
| (d) 並列 | 10b の後。11、13 と並列に進めてよい |

### ステップ 13: specs / memory / skill / CLAUDE.md を更新する

| 項目 | 内容 |
|---|---|
| (a) 目的 | 9 節の方針で書き直す |
| (b) 触るファイル | `specs/**`、`memory/*.md`、`memory/MEMORY.md`、`.claude/skills/*`（skill `edit-dot-claude` の手順で）、`CLAUDE.md`（確認 R3、R8 の後）、`CONTRIBUTING.md` |
| (c) 完了条件 | <ul><li>`just run` の全体が通る</li><li>`grep -r "memory/agents" specs` が 0 件</li><li>`grep -r "PrevPort\|NextPort\|carry_spec\|StateDict\|mujou.typing\|has_received" specs docs examples src` が 0 件</li></ul> |
| (d) 並列 | 11、12 と並列に進めてよい |

---

## 9. specs / docs / memory / skill / CLAUDE.md の更新方針

### 知識の置き場所の表（`specs/README.md` に置く）

| 知識 | 正本 | ほかの場所の扱い |
|---|---|---|
| 公開名の一覧 | `__all__` と `tests/mujou/test_init.py` | specs と memory からは削る |
| API の振る舞い（引数、例外） | docstring | docs は mkdocstrings で取り込む。specs には書かない |
| 状態の木の形 | `Run.state_dict` の docstring | `test_state.py` で固定する。docs と specs はリンクだけ |
| 状態に入るものの規則 | `Node` の docstring | `concepts.md` は 1 文とリンク |
| ノードが守る規則と注意書き | `docs/*/concepts.md` の 1 節 | 各ページは 1 行のリンクだけ |
| 使い方のコード | `examples/` | docs は snippet で取り込む |
| 設計とその理由 | `specs/` | skill は specs にリンクする |
| import の層 | `tests/test_layers.py` | `specs/architecture.md` は理由を書いてリンクする |
| 内部の名前の規約 | `specs/architecture.md`（D51） | refactor-conventions はリンクする |
| ガード | `specs/scenarios.md` の台帳 | 利用者向けには docstring の `Raises:` |
| PC の理論 | skill `pc-alm` | `specs/pc.md` は mujou の選択だけ |
| 実験の数値と性能の基準値 | `specs/research/` | memory/agents を参照しない |
| 決定とユーザーの好み | memory（1 ファイル 1 決定。現在の決定と理由だけ） | 履歴は git |
| dtype の対応表 | `spec.DLPACK` | `TensorSpec.dtypes` で公開する |
| hook の名前 | Rust の `Hook` と、bindings の対応表 | — |
| 属性の所属（ポート、子） | インスタンスの `__dict__` | レジストリを持たない |

### specs

- **書き直す:** `README.md`、`overview.md`、`nodes.md`、`execution-model.md`、`network.md`、`messages.md`、`native-core.md`、`update-rules.md`、`pc-alm-continuous-time.md`。
- **消す:** `torch.md`、`python-environment.md`、`pc-module.md`（中身は移す）。
- **新設:** `architecture.md`、`scenarios.md`、`pc.md`、`research/pc-experiments.md`、`research/benchmarks.md`。
- 削除するもの:
  - 日付
  - 「ユーザーの決定」の注記（91 か所）
  - エラー文言
  - Run のメソッドの表
  - 並行 run と KeyboardInterrupt の項目
  - 内部の hook と `CACHED_HOOKS` の記述
- `pc.md` に、次の 2 つを不変条件として書きます。
  - mask がすべて False の Activity では、どのノードも `learn_` を呼ばない（θ が動かない。h と λ は動く）
  - 書き込みの前の Activity の `time` は 0、`LearnClock` の初期値は -1
- `open-decisions.md` には、本当に未決定のものだけを残します。shape の決め方、配置の指定、dual leak、pinned memory、Δt の分解能、`DictSpec` のキーの二重宣言（`Annotated` 案）、10 節の R9〜R15 と Q1〜Q4 です。

### docs

6.6 節のとおりです。注意書きの重複（`time.sleep` が 6 ページ、「1 回だけ走る」が 4 ページ）は、`concepts.md` の 1 節に集めます。例と docs からは、`torch.enable_grad()` と `# mujou: local-autograd` を外します。

### memory（書き直して、覆された節と打ち消し線を残さない）

| ファイル | 書き直す内容 |
|---|---|
| `project_carry.md` | コアの carry は廃止した（R1 の結果）。理由と、PC が Activity のフィールドで運ぶこと |
| `project_network_ports.md` | `Port` / `MultiPort`、fan-out の規則、相手 → Channel の dict、Network の Port は参照 |
| `project_port_registry.md` | `__dict__` を唯一のレジストリにし、run の段 1 で読む（R7 の結果） |
| `project_group_and_typing.md` | `Group(**named)`、`mujou.typing` の廃止 |
| `project_run_node_and_write_time.md` | `run(root)` |
| `project_timer_pause.md`、`project_timer_per_run.md` | `run.timer` と `global_timer`、RunClock、`Timer(start=, time_scale=)`、復元した実行の専用のタイマー（R8 の結果）、Timestamp 同士の差は秒の float |
| `project_testing_api.md` | `VirtualTimer` は `mujou.time`、`step(dt)` |
| `project_runtime_api_revision.md` | build と Graph の廃止 |
| `project_pause_and_state.md` | 素の dict の木と、その形 |
| `project_host_io.md` | 時刻 `None`、`has_received` の廃止、inputs の時刻 |
| `project_message_types.md` | carry と「片方向スロット 2 本」の記述を、今の形に直す |
| `project_mn_scheduler.md` | ワーカー数の既定（quota を数える available_parallelism、GIL で半分）、実効の数は 0 もありうる |
| `project_pc_module.md` | 置き場所は `mujou/pc`、状態は h で持つ（u の節を消す）、`act` で繋ぐ、`LearnClock`、テストは `tests/mujou/pc` |
| `project_torch_module.md`、`project_gpu_policy.md`、`project_native_core_rust.md`、`project_core_stub.md` | Driver、`_on_enter` と `_before_hook` の記述を消す、max_dt は pc が持つ、例外は Python |
| `project_grad_on_by_default.md` | 例の参照先を `examples/online_learning.py` に直す |
| `project_shutdown_on_exception.md` | 存在しない `on_exception_capture` の記述を消す。end は失敗を検出した時刻 |
| `project_node_lifecycle.md` | hook の構成と呼ぶ順（`on_start` → `load_state_dict`）は維持する。private hook は Driver に移した。`on_start` では状態に依存する出力をしない |
| `project_open_decisions.md` | 現在の未決定事項だけ |
| `feedback_super_calls.md` | 「テストの中のノードは省略してよい」を削除する |
| `feedback_direction_names.md` | 向きは属性名の prev / next で表し、クラスは `Port` / `MultiPort` にする（R2 の結果） |
| `MEMORY.md` | 索引の行を、書き直した各ファイルの内容に合わせて直す（carry、Group と typing、PrevPort / NextPort、代入時の登録、run の位置引数、VirtualTimer、`_on_enter`）。新しく 1 行を足す（今回の作り直しの決定は `specs/architecture.md` と `specs/scenarios.md` を正本とする、という指針）。未追跡の `feedback_good_taste.md` と `feedback_guards_within_scenarios.md` は、そのまま commit する |

### skill

| skill | 変更 |
|---|---|
| `locality-and-async` | M:N の表と「Δt の上限か刻みの分割」の記述を、`specs/native-core.md` と `specs/pc.md` へのリンクにする。所有者の検査の正本は K12（Driver の wires の lookup）だと書く |
| `testing-strategy` | テストの配置を 4.4 節の形（src のミラー、`nodes.py`、`test_layers.py`、`test_pyright.py`）に書き直す。hooks のテストの場所は変えない |
| `refactor-conventions` | 3 行を足す: 依存は `test_layers.py` の DAG、ガードは `scenarios.md` の台帳に載るものだけ、クラスの外から `_` 属性に触らない（D51） |
| `pc-alm` | 138 行目の「コアの carry」を直す。7.2 節の実測値を、`specs/research/` へのリンクに置き換える |

### CLAUDE.md（R3 と R8 の確認の後に）

- ツーリングの節の「利用者が継承するのは `mujou.Node`（コアの `Node` を継承）」を、「利用者が継承するのは純 Python の `mujou.Node`。Rust は `Driver` を介して hook を呼ぶ」に改めます。
- 不変の原則の節の「既定はグローバルタイマー」に、「（復元した実行は、保存した時刻から始まる専用のタイマー）」を足します。不変の原則のほかの文は変えません。
- コマンドの表の `just type` の説明に、examples も検査することを書き足します。

---

## 10. 残るリスクと未決定事項

### 確かめた事実（判断の根拠）

| 事実 | 確かめた場所 |
|---|---|
| carry の使い手は `_core/pc` の 3 ノードと `examples/carry.py` だけ | — |
| pc の全ノードは、同じ update の中で recv してから send している | `layer.py:219-262`、`output.py:144-164` |
| `has_received` はスロットを読み直していて、`recv_with_time` と競合する | `binding.py`、`input.py:83-84` |
| 1 回の update で FFI を 2 回またいでいる | `crates/mujou-python/src/runtime.rs:321-325` |
| enter は、スレッドが別のノードに切り替わるたびに呼ばれる（プールではほぼ毎回） | `crates/mujou-core/src/runtime/mod.rs:283-286` |
| ランタイムの dt と `last_update_ns` は、共有の Timer から取っている | `runtime/pool.rs:146,344,438`、`runtime/node.rs:137,161,185` |
| 終了時刻を max(停止の時刻、最後の更新、スロットの時刻) で推論している | `session.py:301-320` |
| ほかの live な実行があるタイマーへの時刻の読み込みを、向きを問わず 1 つの lock の下で拒んでいる。live は start で数え、終了で減らす | `runtime/mod.rs:345-372`、`runtime/mod.rs:375-380`、`runtime/freeze.rs` |
| manual モードでも、hook はランタイムのスレッドで走る（grant） | `control.rs:84-88` |
| 未接続の multi のポートには、入力スロットが無い | `session.py:147` |
| `MultiPort` どうしは 2 回繋げる。自己接続は拒んでいる。channel の番号は `peer.channels.index` で引いている | `port.py:339`、`port.py:315`、`session.py:143-145` |
| hook は勾配を記録する（`no_grad` の中で走るのは `load_state_dict` のコピーだけ）。勾配を切らないのはユーザーの決定 | `node.py:98-100`、`node.py:393`、`memory/project_grad_on_by_default.md` |
| `load_state_dict` は `on_start` の後に呼ばれる。on_start は状態の確保、というのがユーザーの決定 | `crates/mujou-python/src/runtime.rs:304-306`、`specs/nodes.md:71`、`project_node_lifecycle` |
| 乱数生成器の状態は木に入らず、利用者に自分で足すよう頼んでいる | `node.py:235-236` |
| 今の例は model を `on_start` で作っている | `examples/online_learning.py:60-64` |
| Layer は、mask に True が無ければ学習を飛ばし、δ も行ごとに mask している | `layer.py:231-233`、`layer.py:249` |
| `Learner` の初期値は `_seen_time=-1`、`_learn_at=0` | `learner.py:95-96` |
| `_jumped` は `epoch` を進める条件 | `layer.py:259` |
| ノードの分け方を変えても同じ結果になることのテストがある | `tests/mujou/_core/pc/test_layer.py:590` |
| `learn_interval` の 2·深さ·周期は安定の境界ではない（16 周期と None は収束、4 周期は有界、2 周期と 1 周期だけ発散） | `specs/pc-module.md:237` |
| 今の例は経過時間を溜めて学習量を決めている | `examples/online_learning.py:66-90` |
| `on_exception` と `on_pause` / `on_resume` の存在理由はロボティクスの I/O で、ユーザーの決定 | `specs/nodes.md:66-87`、`project_node_lifecycle` |
| 属性は代入時に登録し、vars() を探索しない、というのがユーザーの決定（2026-10-17） | `memory/project_port_registry.md`、`component.py:52-58` |
| dtype の対応表は `spec.py` にある | `spec.py:18` |
| `TensorSpec(dtype=None)` は、作った時点の `torch.get_default_dtype()` になる | `spec.py:103-104` |
| `DictSpec.typed_dict` の使い手は carry だけ | `_core/pc/carry.py:113` |
| ワーカー数の既定は、quota を数える `available_parallelism` を GIL で半分にしたもの | `crates/mujou-python/src/runtime.rs:402-419` |
| グローバルの仮想時計の fixture は、`use_virtual_clock` と `load_state_dict` に依存している | `tests/mujou/conftest.py:29-37` |
| pyright の `reportPrivateUsage` は、同じモジュールの関数からの `_` 属性の読み書きも警告する（`reportPrivateUsage = "warning"`） | scratchpad の 2 ファイルで pyright を実行して確認、`pyproject.toml:97` |
| `VirtualTimer.advance` は、pause 中だと `Paused` のエラーになる | `crates/mujou-core/src/time.rs:315` |
| `Timer::new` は 0 から、scale 1 で始まる | `crates/mujou-core/src/time.rs:181-191` |
| 手動モードは周期を無視する、というのがユーザーの決定 | `project_node_period` |
| `torch.optim` は free-threaded で GIL を有効にする | `examples/mnist/bp.py:53` |
| MNIST を examples に置くのはユーザーの決定 | `feedback_examples_are_guides` |
| `benchmarks/mn_scheduler.py` はすでにある | — |

### ユーザーに確認すべきこと（推奨案付き）

| # | 確認事項 | 推奨 | 理由と、断られた場合 |
|---|---|---|---|
| R1 | コアの carry を廃止し、PC は `y` / `mask` / `time` を `Activity` のフィールドで運ぶ（今日の #89 と `project_carry` を覆す） | 廃止する | 使い手が 1 人で、組み違えの防止はフィールドでも保てる。断られた場合は D7 の代案（引数を分け、空の Carry で送信を 1 本にする） |
| R2 | `PrevPort` / `NextPort` を `Port` / `MultiPort` に置き換え、`feedback_direction_names` を「向きは属性名の prev / next で表す」に改める | 置き換える | 違いは multi の既定値だけで、相手の数を表す軸が 3 つある。向きの語彙は属性名に残るので、指示の趣旨（上下の比喩を避ける）は守れる |
| R3 | `mujou.Node` を純 Python にし、Rust は `Driver` を呼ぶ形にする。CLAUDE.md のツーリングの一文を書き換える | 書き換える | FFI が hook ごとに 1 回になり、利用者の MRO から private hook が消え、`super()` を忘れても枠組みの処理が飛ばない。断られた場合は `_native.Node` を残し、private hook を 1 つの `_dispatch` にまとめるところで止める |
| R4 | `mujou.run` の root を 1 つにし、`Group` の位置引数を廃止する（`project_run_node_and_write_time`、`project_group_and_typing` を覆す） | 廃止する | パス（状態の木のキー）が、実行のしかたで変わらなくなる |
| R5 | 捨てた Run を救う監視 daemon スレッドを消し、atexit だけにする | サブプロセスでの終了テストを、3.14t と GIL ありの両方で通してから消す | 止まらない場合はスレッドを残し、その理由を `specs/execution-model.md` に 1 段落で書く |
| R6 | `mujou.time` の関数 7 個を廃止する（`mujou.time.sleep(1)` という短い書き方が無くなる） | 廃止する | 独自の Timer の run に使うと誤る、という落とし穴を消す。ホストは `run.timer.sleep()` を使う |
| R7 | 属性の登録を、代入時の `__setattr__` からインスタンスの `__dict__` の走査に変える（`project_port_registry` を覆す） | 変える | 1 つの事実を 2 か所で持たず、代入のたびの検査、K2 の代入時の拒否、番兵、解除の処理が消える。失うのは、K3・K6 を代入の行で知らせられることだけ（段 1 でパス付きで知らせる）。断られた場合は今の形を残し、K6 を Node の Port にも広げる（構築中だけ検査する） |
| R8 | 復元した実行は、保存した時刻から始まる専用のタイマーで走らせ、`timer=` との同時指定を拒む。CLAUDE.md の不変の原則の「既定はグローバルタイマー」に「復元した実行を除く」を足す | 変える | 共有の時計の巻き戻し、`LIVE` の数え上げ、K24 の競合が、表せなくなって消える。不変の原則の中身（1 つの実行に時間軸は 1 本）は変わらない。断られた場合は、反証 M2 の形（向きを問わず拒み、検査・読み込み・計上を `_native.start` の中で 1 つの lock の下で行う。live は start で数えて終了で減らす。印は `Driver.start` が付ける） |

### 性能と検証が要るもの（推奨案付き）

| # | 事項 | 推奨と、確かめ方 |
|---|---|---|
| R9 | `_native.current()` を FFI で読むことによる、recv / send のオーバーヘッド | ステップ 5 で、ステップ 1 の基準値と比べる。悪化が 5% 未満なら採用する。超えたら、事実の置き場所を Python の葉のモジュールの `threading.local` 1 か所に移す（Driver だけが書き、bindings の Timer の検査はそれを呼び戻して読む）。2 か所には置かない |
| R10 | 状態の木を素の dict（CPU）にすることによる、GPU のノードの capture のコスト | 停止中は device の上で複製して同期し、再開した後に CPU へ移す。GPU のノードで頻繁に capture するベンチマークを測る。許せないほど悪化したら、`state_dict(device=...)` で残す場所を選べる引数を足す（それまでは足さない） |
| R11 | 専用スレッドをプールの状態機械に乗せることによる、fast path の喪失 | M:N のベンチマークを回帰の基準にする。落ちたら、`lifecycle.rs` を変えずに `sched.rs` の中だけで戻す |
| R12 | carry の廃止で、1 ホップごとに y / mask / time をコピーするコスト | MNIST の 1 epoch で、悪化が 5% 未満であることを確かめる。密な y（`pc_regression`）と CUDA のノードでの値は、記録だけする |
| R13 | Driver が CUDA のノードで、hook ごとに device と stream を設定するコスト | ステップ 5 で測る。目立つなら、PyDriver が「スレッドが切り替わったか」を引数で渡す形に戻す（core に enter のキャッシュは戻さない） |
| R14 | `Port[S = Never, R = Never]` の型パラメータの既定値と、`connect(other: Port[R, S] \| MultiPort[R, S])` を、pyright strict で推論できるか | ステップ 8 の最初に `test_pyright.py` で確かめる。駄目なら、`connect` の型を `Port[Any, Any] \| MultiPort[Any, Any]` に緩める（実行時の照合 K10 は残る） |
| R15 | `Message.time_ns: Option<u64>` で `sent` を吸収したとき、wait_newer と InputGroup のバージョンに影響しないか | ステップ 7 で、`native/test_slot` の「送られたものだけが sent」のテストを、時刻 None の形に書き換えて通す |

### algorithm-reviewer の判断が要るもの

| # | 事項 | 推奨 |
|---|---|---|
| Q1 | `theta=inf` を `learn_interval=inf` で表すこと（学習せず、跳びも数えない） | Jacobi の参照実装と、`theta=inf` のテストが変わらないことを確かめて採用する |
| Q2 | `Tick` が、学習した時刻を `learn_` の後ではなく `tick` の `now_ns` で記録すること | 手動モードでは同じ。実時間では、更新の途中で格子の点を越えたときだけ 1 区間ずれうる。ずれを許すかを判断してもらう。許さないなら、`tick` に渡す `now_ns` を、学習の後に読む形へ変える |
| Q3 | 教師なしのときの `Output` の `g_in` を、`loss_vjp` の mask 処理で $J^\top 0$ にすること | ŷ が有限なら 0。ŷ が inf のときだけ NaN になることを docstring に書いて採用する |
| Q4 | `learn_interval` の推奨の文言（2 × ホップ数 × period、数周期以下で発散を観測） | `specs/pc-module.md:237` の実測に沿って確かめる |

前版の R13（K30 の定数）、R14（`learn_interval=None` を残すか）、R15（`_jumped` の吸収）は、invariants の反証が実測とコードで答えたので、それぞれ D39、D34、D38 で決めました。

### 範囲の外として扱うもの（確認だけ）

- **`memory/agents/` の扱い:** 触りません。specs からの参照だけを断ちます。
- **legacy DLPack の再試行の削除と、torch の最低版を上げること:** 最低版を決めるまで保留します。
- **未接続の `MultiPort` にホストが書くこと:** サポートしません（K17 が「ホストの入力には Port を使う」と案内する）。必要になったら、ホストの入力を Channel の 1 本として扱う案を検討します。
- **`DictSpec(TD, **specs)` でキーを 2 重に宣言すること:** 今回は変えません（K8 が、定義の行で過不足を知らせる）。`Annotated` を使う案は、open-decisions に残します。
- **周期どおりに決定的に回す手動モード:** 使い手が現れたら検討します。

---

## 付録: 反証への対応

### 初めて使う研究者の目（user-walkthrough）

| 指摘 | 対応 | 理由と、反映した場所 |
|---|---|---|
| 0. 例のない (3) 分岐と合流、(5) PC 以外の更新手法 | 採用 | (3) は 7 節 (b) の `Fork` に、(5) は `examples/feedback_alignment.py` と `topics/update-rules.md` に入れた（D42、D43、D54） |
| 1-1 非テンソルの属性が黙って状態から漏れる。例が誤った型を教えている。generator が復元されない | 部分採用 | 乱数生成器を既定の `state_dict` に入れ、例を直した（tau は登録しない。`last` と `elapsed` は登録する）。「`__init__` の後に代入したポートと callable 以外はすべて状態」の案は不採用。設定を黙って古い値で上書きし、外す仕組みが要り、I/O のハンドルで取得が壊れるため（D21）。register_state の綴りの誤りは K30 で知らせる |
| 1-2 `load_state_dict` が `on_start` の後に呼ばれる | 不採用（規則と書き方は採用） | 順を変えると、`on_start` で状態を作るノード（今の例とユーザーの決定）が、復元した値を黙って上書きする。`on_start` で状態に依存する出力をしない規則を、`concepts.md` と docstring に書き、例は状態を `__init__` で作る書き方にそろえた（D22） |
| 1-3 既定のグローバルタイマーに復元すると、時計を巻き戻す | 採用（R8） | 復元した実行は、保存した時刻から始まる専用のタイマーで走る。`timer=` との同時指定は K23。「`global_timer.pause()` では止まらない」と docs に書く（D50） |
| 1-4 `step` が period を無視すること、更新の順序、最初の dt が書かれていない | 部分採用 | 3 点と Gauss-Seidel の順を docstring に書いた（3.3 節、D15）。省略時の意味を「期限に達したノードを期限の順に」へ変える案は不採用。周期を無視するのはユーザーの決定で、使い手のいない新しいモードになるため |
| 1-5 mask を False にしたとき、Layer も学ばないことが保証されていない | 採用 | コードはすでに mask で学習を飛ばしていた（`layer.py:231`）。`pc.md` の不変条件にし、θ のビット一致のテストをステップ 1 で足す。文言は「θ が動かない（h と λ は動く）」にした |
| 1-6 Port を list や dict に入れると、黙って登録されない | 採用 | K6 を Node にも効かせ、`tree.walk` が「1 つずつ代入するか、MultiPort を使う」と知らせる（D49） |
| 2-1 「基底の hook は何もしない」と、既定の state_dict / load_state_dict が矛盾する | 採用 | 状態の 2 メソッドを hook の行から分け、override の型（super の戻り値を拡張し、足したキーを除いて渡す）を書いた（3.1 節） |
| 2-2 ノードをまたぐ PC 以外の更新手法への道がない。`LearnClock` が `TimeConstants` を要求する | 採用 | 概念 9 を直し、`topics/update-rules.md` と `examples/feedback_alignment.py` を足した。`LearnClock` から tau を外した（D34、D54） |
| 2-3 hook が `no_grad` の中で走ることが書かれていない | 部分採用（前提は誤り） | hook は勾配を記録する（`node.py:98-100`、`project_grad_on_by_default`）。`:393` は `load_state_dict` の中。直すべきは例で、`enable_grad` と目印を外し、「hook は通常の PyTorch と同じく勾配を記録する」と書いた（D54、7 節 (b)） |
| 2-4 `LearnClock` が呼ぶ順の契約を持つ。2 つの bool の組 | 採用 | `tick` は `Tick(dt, new, learn, elapsed)` を返し、learn を返した時点で数え直す。`learned()` は消す。記録の時刻の変化は Q2（D34） |
| 2-5 `pc.show` の意味が名前から読めない | 採用 | docstring の先頭を prequential の説明にし、評価では短い seconds でよいと書いた。例 (c) の評価は 0.02 秒にした（D36） |
| 2-6 `Chain` の子のパスが 1 から始まる。period の指定がない | 採用（period の引数は足さない） | `layer0` から始める。period はすべて `tau.h/4` と docstring に書く。K30 を外したので、period に依存する検査も残らない（D35、D39） |
| 2-7 `initial` の向きと、初期値の 3 つの道 | 採用 | `initial` の docstring を直し、`topics/host-io.md` に 3 つの道と、それぞれの時刻の表を置く。inputs の時刻は、起動の時刻（D2） |
| 2-8 時刻の表現が 3 種類あり、`Timestamp` の作り方も書かれていない | 部分採用 | `Timestamp(ns)` を公開のシグネチャにし、`Activity.time` の意味を docstring に書いた。`tick` が Timestamp を受け取る案は不採用。`Activity.time` は int64 のテンソルなので、ns の int の方が変換が少ない（D37、D47） |
| 2-9 MultiPort の dict で、どの相手かが分からない。循環してよいかが書かれていない | 採用 | 並びは connect の順と定義した。「役割の違う入力は Port を分ける」「循環してよい」を `topics/ports.md` に書く（D6、3.1 節） |
| 2-10 学習した `nn.Module` を run の後で取り出せるかが書かれていない | 採用 | 「終わった後は、ノードの属性を読んでよい」を規則にし、K14 の文言にも入れた（例 (c)） |
| 3-1 PC の時間のつまみが 7 つある | 採用 | tutorial 5 は tau と seconds だけにする。`TimeConstants` の docstring に学習率との対応を書き、max_dt と period は Chain が導くと明示した |
| 3-2 手で組むと、Layer ごとに `h_in=` と `y=` を書く | 採用 | `act: DictSpec[Activity]` 1 つで渡す。B の照合は `activity_spec` の 1 か所になった（D38、K28） |
| 3-3 in place で更新する規則に理由がない | 採用 | 規則を消し、「`send` はコピーする。送った後に変えてよい」を `Port.send` に書いた。例 (b) の Leaky は付け直しで書いた |
| 3-4 `TensorSpec(dtype=None)` の意味 | 採用 | 既定を `torch.float32` にし、`None` を消した（D53） |
| 3-5 `seed` が何に効くか | 採用 | `generator` にだけ効くことと、module の初期化に `generator=self.generator` を渡す型を docstring と例に書いた |
| 3-6 timeout と sleep で単位が違う | 採用 | docstring に wall-clock と書き、`topics/time.md` に単位の表を置く（3.2 節、D28） |
| 3-7 DictSpec にキーを 2 回書く | 不採用 | Python の型は、実行時に TypedDict を生成できないので、重複は避けにくい。K8 が定義の行で知らせる。`Annotated` 案は open-decisions に残す |
| 3-8 `Run.read(MultiPort)` が何を返すか | 採用 | 「最後の `send` の値。`send_to` は outbox に書かない」と定義した（3.1 節） |
| 3-9 `NetworkError` がどの呼び出しから出るか | 採用 | 失敗の後のホストの呼び出しはすべて出す。`with` を別の例外で抜けるときはそちらを優先する、と `Run` の docstring に書く（3.1 節） |
| 3-10 `self.timer` を run の前に読んだときのふるまい | 採用（形は変えた） | `Node.timer` はいま走っている Driver の時計なので、hook の外では K12 の `RuntimeError` になる（D11） |
| 3-11 register_state の値が weights_only で読めるか | 採用 | 許す型を docstring に列挙した（3.1 節） |
| 4. 分岐と合流が tutorial にない | 採用 | `tutorial/1-run.md` の後半に入れた（D43） |
| 4. 自作の更新手法のページがない | 採用 | `topics/update-rules.md` を足し、tutorial 4 からリンクする（D43、D54） |
| 4. `examples/test_leaky.py` が CI で実行されない | 部分採用 | 実行はしない規則を保ち、`just type` で examples を pyright（standard）で検査する。docs-build ではなく型の検査の recipe に置いた（D42） |
| 4. MNIST への道 | 採用 | tutorial 5 の最後に「次は examples/mnist」と書き、`topics/pc.md` に最後の半端なバッチを mask で詰める方法を書く |

### 不変の原則と PC-ALM の数理（invariants）

| 指摘 | 対応 | 理由と、反映した場所 |
|---|---|---|
| M1 `RunClock` を Rust ランタイムの唯一の時刻源にすると明記する | 採用 | dt、`last_update_ns`、スロットの時刻を `RunClock` から取り、end の定義（停止の要求、失敗の検出）と、止まった後の hook で実時間を使う規則を書いた（D11、ステップ 4） |
| M2 K24 を巻き戻しだけに狭めるのは後退。段 2 での読み込みは D2 に反し、TOCTOU も起きる | 代替で解消 | 復元した実行に専用のタイマーを与え、共有のタイマーへの読み込みそのものを無くした（D50、R8）。断られた場合は、M2 の形そのものを代案として実装する |
| M3 K30 は入れない | 採用 | 台帳から外し、ホップ数で書いた推奨値にした。「condition of stability」の言い過ぎも直す（D39） |
| M4 例 `OnlineLinear` が、時間で更新量を決める原則を破っている | 採用 | `elapsed` を溜めて `alpha=elapsed / tau` を掛け、0 に戻す形にした（7 節 (b)） |
| S1 終了後の capture の排他と、CUDA の stream の同期 | 採用 | `Runtime.capture` が lock を取る。`Driver.capture` が複製の後に stream を同期することを責務にした（D23） |
| S2 例外で終わった後の `state_dict` の意味 | 採用 | `NetworkError` を出すことを維持すると明記した（D23） |
| S3 時刻 None のときに `Input` が送る `time` | 採用 | time 0・mask False と定義し、LearnClock の初期値 -1 との関係を `pc.md` の不変条件にした。時刻 0 の書き込みのテストは残す（D37） |
| S4 `_jumped` を吸収しない | 採用 | フラグを残し、`h.add_(jump); jump.zero_()` は無条件でよいとした（D38）。前版の R15 を閉じた |
| S5 ノードをまたいだ module やテンソルの共有 | 採用 | K24 として `tree.walk` で拒む。view の共有はサポートしないものに書いた（D52） |
| S6 R10 の計測条件（密な y、CUDA） | 採用 | MNIST に加えて `pc_regression` の密な y でも測り、記録する（D7、R12） |
| S7 粒度を変えても同じになることのテストを計画に明記する | 採用 | 4.4 節の `test_layer.py` と、ステップ 6 の完了条件に入れた |
| nit: `Chain` と手組みの比較は `torch.equal` | 採用 | `test_chain.py` とステップ 10b |
| nit: `step(dt)` の省略時の順序（Gauss-Seidel）と、Jacobi のための逆順 | 採用 | 3.3 節の docstring |
| nit: 「何も学ばない」は「θ が動かない」 | 採用 | 例 (c)、`pc.md`、1-5 の不変条件 |
| nit: `Output` の分岐の一本化で $J^\top 0$ になる | 採用（確認つき） | docstring に書き、algorithm-reviewer に回した（D40、Q3） |
| nit: `Input` は period を必須にする | 採用 | tau を受け取らず、period を必須にした（D38） |
| nit: R14（`learn_interval=None`）は残す | 採用 | `LearnClock` の中に残した（D34） |

### 実装の完全性（completeness）

| 指摘 | 対応 | 理由と、反映した場所 |
|---|---|---|
| A1 live を `RunClock` の生成で数えると、終わった実行が残る | 採用（R8 で解消） | live の数え上げごと消した（D50）。R8 を断られた場合の代案では、start で数えて終了で減らす |
| A2 `ReadOnlyTimer` を作る口が無い。復元の時刻の読み込みが D2 を崩す | 採用 | `ReadOnlyTimer(timer)` を誰でも作れるようにした（end はランタイムだけが付ける）。時刻の読み込みは無くした（D11、D50） |
| A3 全ノードが専用スレッドの実行が K21 で拒まれる | 採用 | 実効のワーカー数に 0 を許し、K20 は明示した値だけを見る（D30） |
| A4 ステップ 4 と 5 の間で K17 のテストが落ちる | 採用 | `current()` をステップ 4 で bindings に入れ、Timer の検査をそこで置き換える（ステップ 4） |
| A5 `global_virtual_timer` が消す API に依存している | 採用 | fixture を消し、振る舞いは VirtualTimer で、グローバルは同一性と smoke だけで検査する（4.4 節、6.5 節） |
| A6 dtype の表の正本を codec に置くと、層の向きが逆になる | 採用 | 表は `spec.DLPACK`（L1）に置き、codec が読む（D25） |
| A7 「friend 関数 0」は reportPrivateUsage の下で成り立たない | 採用 | pyright で確かめた。実行時の事実を実行側に移して触る箇所を 0 にし、内部の名前の規約を決めた（D51、D10、D17、D49） |
| A8 同じ層の中の import、許可リストの漏れ、`_Wire.owner` の型 | 採用 | `state.py` を `wiring.py` の上の層にし、`node.py` を許可リストに入れた。Wire は owner を持たず、device を持つ（4.1 節、D10） |
| B1 `"to"` を相手のパスで引くとキーが衝突する。自己接続は構造の前提 | 採用（データ構造で） | 接続を相手 → Channel の dict にし、二重接続と自己接続を表せなくした。`"to"` は相手のパスで一意（D6、D23） |
| B2 状態の木を 2 回書き直す順序。ステップ 5 の暫定コード。テストの往復 | 採用 | 時刻 None をステップ 7（木の書き直しのステップ 9 より前）に移し、暫定の引数分けを書かない。テストは最後の置き場所へ 1 回だけ移す（8 節） |
| B3 inputs・状態・initial の K9 の段が書かれていない | 採用 | 段 2 で、新しい native のオブジェクトに書いて検査する。initial は段 2 でパス付きで知らせる（D2、D25） |
| B4 「update ごとの FFI は 1 回」がプールでは成り立たない | 採用（形は変えた） | enter とそのキャッシュを消し、Driver が hook ごとに設定する（D16、R13） |
| B5 `Runtime` の口が抜けている。終了の事実を 2 か所で持つ | 採用 | `stop` / `join` / `finished` / `check` を Rust から返し、Python の enum は消した（D24、D32） |
| B6 `os.process_cpu_count()` は quota を数えない | 採用 | 既定は bindings の `available_parallelism` に残し、テストは `Runtime.workers` で見る（D32） |
| B7 K30 と台帳の矛盾。意味論の一覧の漏れ | 採用 | K30 を外し、wait_for の RuntimeError、終了後の step、BufferError、抽象メソッド、VirtualTimer の pause を意味論に入れた。直接の生成を拒む raise は G5 として消す（5b 節） |
| C1 `state_dict` の戻り値の型を狭めている | 採用 | 差分表に理由（基底のキーとエラーの文言のため）を書いた（3.5 節） |
| C2 duck typing の穴（型そのもの）と速度 | 採用 | `isinstance(v, type)` を除く。取得のときに 1 回読むだけなので、代入のたびより速い（D21） |
| C3 K2 の範囲が `nn.Module` より厳しくなる | 解消 | `__setattr__` を持たないので、super の前の代入は通る。K2 は呼ばなかったことだけを段 1 で見る（D20、D49） |
| C4 `pc.show` は ManualRun の上では返らない | 採用 | docstring に書き、`test_show` は実時間で書く（D36） |
| C5 VirtualTimer を共有すると、step が他の実行も進める | 採用 | `mujou.testing.run` の docstring に書いた（3.3 節） |
| C6 「並列に進めてよい」の一部は衝突する。性能の基準値 | 採用 | 4 と 5 を直列に、10a を 6 の後にした。基準値はステップ 1 で記録する（8 節） |
| C7 `current_node()` の実装の段取り。`LAST_NODE` は意味が違う | 採用 | bindings の thread-local を PyDriver が付け外しする形にし、`LAST_NODE` は消した（D17） |
| C8 K11 の 1 行は自分の側しか見ていない | 採用 | 両端を見る（D6） |
| C9 `test-hooks` を分ける理由が成り立たない | 採用 | `tests/test_claude_hooks.py` を今の場所に残す（D41） |
| C10 ルートのノードのポートの label | 採用 | `name(owner_path) + "." + attr`、`name("")` はクラス名（D5） |
| D. `send` / `recv` をキーワード専用にすることが書かれていない | 採用 | 3.1 節と 3.5 節 |
| D. `DictSpec.typed_dict` の扱い | 採用 | 使い手が carry だけなので削除（3.5 節） |
| D. `mujou.testing.run` の `*components` の削除が 6.2 節で「変更なし」 | 採用 | 6.2 節を直した |
| D. `Timestamp(ns=)` を公開するか | 採用 | `Timestamp(ns)` を公開のシグネチャにした（3.2 節） |
| D. `MessageSpec.rand` は int / bool のフィールドで torch が例外を出す | 採用 | `rand` は `TensorSpec` だけに置き、`MessageSpec` と `DictSpec` には足さない（3.1 節） |
| D. native の型の `__module__` | 採用 | pyclass の module を `"mujou.time"` にする（D14） |
| D. `CACHED_HOOKS` の意味論が消える | 採用 | 「hook はいつも属性から引く」と書いた（3.1 節、D16） |
| D. memory の表に `project_message_types.md` と `project_mn_scheduler.md` が無い。`MEMORY.md` は 1 行では足りない | 採用 | 9 節の表に足し、索引の行を書き直す方針にした |
| D. skill の `pc-alm:138` の「コアの carry」、`testing-strategy:48` のパス | 採用（パスは変えない） | `pc-alm` を直す。hooks のテストは移さないので、`testing-strategy` のパスはそのまま（9 節） |
