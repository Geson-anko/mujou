# CUDA のメッセージをデバイスに置いたまま渡す（device-tensor-passthrough）

## 概要

CUDA のノードが送ったテンソルを、スロットの中でも書き手の device に置いたまま持ち、読み手は device 上でコピーして受け取る。いまの device → host → device の往復（送るたびの同期 D2H と、読むたびの H2D）を無くす。stream をまたぐ順序は「書き手が自分の stream を待ってから公開する」で守る。2026-10-15 の決定 [project_gpu_policy.md](../../project_gpu_policy.md)（コアは GPU をサポートしない）を、ユーザー自身が 2026-10-24 に覆す。

## 結論（推奨案）

| 論点 | 推奨 | 却下した案と理由 |
| --- | --- | --- |
| device 上の値の置き場 | コアの `Message` のフィールドを「プールのバッファ」か「不透明な値（`Arc<dyn Any + Send + Sync>`）」のどちらかにする。不透明な値の中身は bindings が入れる `Py<PyAny>`（codec が作った、書き手の device の上の detach したコピーの `torch.Tensor`）。コアは device・stream・event を知らない | (a) コアのスロットに device ポインタ・stream・event を持たせる: コアに CUDA の意味論が入り、torch を import しない原則も崩れる。(b) Python 側に別の表を持ちスロットには鍵だけ置く: 版番号の一斉切り替え・`wait_newer`・寿命の管理をコアと二重に持つ。(c) `Slot<V>` の総称化: 変更が大きく得るものが無い |
| stream 間の同期 | **書き手が send の中で、コピーを積んだ自分の stream に event を記録してホストで待ち、それから公開する（S2）**。読み手は待たない（公開された値は device 上で完成している） | S1「書き手は event を記録するだけで公開、読み手が自分の stream で `wait_event`」: ホストは止まらないが、読み手の GPU の仕事が書き手の進行中の計算を待ち、読み手の `on_update` 後の同期を通じて読み手のホストまで書き手を待つ（「ノードは他のノードを待たない」が GPU 側で崩れる）。スロットの時刻も値が出来上がる前の時刻になる。差は 1 send あたり約 12 µs（下の実測）なので、不変条件を優先する。**確認事項 1** |
| キャッシュアロケータの再利用 | 読み手はコピーを積んだあと、保持テンソルに `record_stream(コピーした stream)` を付ける | 「読み手の Driver が読んだテンソルを `on_update` 後の同期まで握る」: recv のたびに list へ足す処理と、hook ごとの後始末が要る。`record_stream` は 1 か所で済み PyTorch の標準の手段 |
| `on_update` 後の stream の同期 | 残す。メッセージの正しさにはもう要らないが、Δt と実際に計算した時間を一致させ、CPU が GPU を追い越して仕事を積み続けないための同期（[specs/nodes.md](../../../specs/nodes.md#cuda) の表の 1 行目の理由のまま） | 外す: Δt の意味が変わり、GPU のキューが無制限に伸びる |
| 読むときのコピー | 保つ。読み手は毎回、自分の device の新しいテンソルを受け取る | 読み手に保持テンソルを渡す: 読み手のその場の変更が他の読み手に漏れる |
| 書くときのコピー | 保つ。書き手の device 上で `clone`（連続な row-major） | 書き手のテンソルの参照を持つ: 送った後の書き手のその場の更新が読み手に見える |
| 何をデバイスに置くか | **ノードの hook の send / send_to で、ノード自身の CUDA device の上の strided なテンソル**だけ。ホストの write・`inputs=`・`initial`・復元はホストのメモリ（今と同じ）。**確認事項 2** | すべての CUDA テンソルを置く: 書き手のスレッドの current stream と current device（torch の `__dlpack__` が要求する）が保証できず、読み手の副 stream（下記）の device も静的に決まらない |
| スカラー | 今と同じ Python の値（ホストのバッファ） | — |
| メモリ管理 | device の値は torch のキャッシュアロケータをプールとして使う（定常状態で `cudaMalloc` しない）。ホストのバッファプールは今のまま | 独自の device のプール: アロケータの二重化 |

### 実測（RTX 4090、3.14t、`torch 2.14.1+cu130`、1 回の送受信を Python だけで模した値）

| 要素数 | 今（D2H + H2D） | S2 | S1（`on_update` 後の同期なし） |
| --- | --- | --- | --- |
| 64 | 33 µs | 29 µs | 18 µs |
| 4096 | 47 µs | 30 µs | 19 µs |
| 1,048,576 | 779 µs | 33 µs | 17 µs |

S1 も `on_update` 後の同期を足すと約 40 µs で、send が `on_update` の最後にある普通のノードでは S2 と差が出ない。

## 公開インターフェース案

利用者の API は変わらない（`Port.send` / `recv`、`Run.read` などのシグネチャも戻り値の device も同じ）。変わるのは内部の口だけ。

Python（`src/mujou/`）

- `codec.to_core(spec: MessageSpec[Any], msg: object, label: str, *, device: torch.device | None = None) -> list[object]` — `device` は書き手のノードの device。CUDA で、値が `value.device == device` かつ `value.layout == torch.strided` のテンソルなら `value.detach().clone(memory_format=torch.contiguous_format)` を返す（コアがそれを保持する）。1 つでも保持する値があれば、最後に 1 回だけ `torch.cuda.current_stream(device).record_event().synchronize()`。それ以外は今どおりホストへ（`.to("cpu")`）。`None` はホストの書き込み
- `codec.from_core(spec: MessageSpec[Any], views: Sequence[_native.BufferView | torch.Tensor], device: torch.device) -> object` — 保持テンソル `t` は `t.to(device, copy=True)` で読み、続けて `t.record_stream(torch.cuda.current_stream(t.device))`。`BufferView` は今どおり
- `codec.peek(spec: MessageSpec[Any], views: Sequence[_native.BufferView | torch.Tensor]) -> object`（新規） — 保持テンソルはコピーせずそのまま、ホストのバッファは CPU の新しいテンソルにコピーして返す。状態の取得の停止中だけに使い、呼び手が後で CPU へコピーする
- `state.cpu(tree: dict[str, Any]) -> dict[str, Any]` — ノードの状態に加え、各ポートの項目の `"value"` も `torch.Tensor.cpu` で移す
- `wiring.Wiring.read_devices: tuple[frozenset[torch.device], ...]`（新規のフィールド） — 各ノード（`tree.members` の順）が読むスロットの書き手のうち、自分の device 以外の CUDA device の集合
- `driver.Driver.__init__(self, node: Node, clock: ReadOnlyTimer, wires: Mapping[AnyPort, Wire], read_devices: Collection[torch.device] = ()) -> None` — `read_devices` の各 device に副 stream を 1 本ずつ作る
- `port.Wire._values` は `to_core(..., device=self.device)` を渡す（シグネチャは不変）

Rust core（`crates/mujou-core`）

- `buffer::Held = Arc<dyn Any + Send + Sync>`（新規の型別名） — 書き手がプールの外に持つ不変な値。コアは読まず検査もしない
- `buffer::Value { Bytes(Arc<Buffer>), Held(Held) }`（新規の enum）
- `buffer::Message::values(&self) -> &[Value]`（`&[Arc<Buffer>]` から変更）
- `buffer::FieldOut<'a>`（新規） — 新しいメッセージの 1 フィールドの書き込み口
  - `fn spec(&self) -> &FieldSpec`
  - `fn bytes(&mut self) -> &mut [u8]` — 初回にプールからバッファを取る。全バイトを上書きすること
  - `fn hold(&mut self, value: Held)` — プールのバッファを取らない（取っていれば返す）
- `buffer::Fill<E>: FnOnce(&mut [FieldOut<'_>]) -> Result<(), E>`（引数を `&mut [&mut [u8]]` から変更）。`Messages::fill` は埋められなかったフィールドがあれば panic（内部の契約）
- `slot::InputGroup::write<E>(&self, slots: &[&Slot], fill: impl FnMut(usize, &mut [FieldOut<'_>]) -> Result<(), E>) -> Result<Timestamp, E>`
- `dlpack::import::check(t: &DLTensor, field: &FieldSpec) -> Result<(), FieldError>`（新規、`unsafe fn`） — `copy_from` から dtype と shape の照合を切り出す。`copy_from` はこれを呼ぶ
- `dlpack::ffi::DEVICE_CUDA: i32 = 2`（新規の定数。DLPack の `kDLCUDA`）

Rust bindings（`crates/mujou-python`）

- `dlpack::borrow_tensor` は `__dlpack__(max_version=..., stream=-1)` で呼ぶ（CUDA で torch に同期させない。CPU では -1 は None と同じ）
- `slot.rs` の `enum Value { Tensor(TensorPtr), Held(Arc<HeldTensor>), Scalar(Scalar) }` — DLPack の device 種別が `DEVICE_CUDA` なら `check` だけ行い、値のオブジェクトを `HeldTensor(Py<PyAny>)` として持つ。CPU / CUDAHost は今どおりコピー、それ以外は `UnsupportedDevice`
- `PyMessage.values -> list[BufferView | Any]`（`Value::Held` のフィールドは保持しているテンソルそのものを返す）
- `Outlet` の docstring: 「CUDA の値はコピーせずそのまま持つ。誰も書き換えず、書き込みが完了したコピーを渡すこと（codec がそうする）。torch の `__dlpack__` の制約で、呼ぶスレッドの current device はその値の device であること」

## 理論上の対応

更新手法とは無関係（PC-ALM の式は変わらない）。実行モデルへの影響だけ。

- 局所性: 読み手が読むのは今もスロットの値だけ。保持テンソルは codec の中でしか触れず、利用者には毎回コピーを渡す。detach したコピーなので計算グラフはノードをまたがない
- 非同期性: S2 では書き手は自分の stream だけを待ち、読み手は誰も待たない（スロットの mutex の短い区間を除く）。スロットの時刻は「値が device 上で出来上がった後」の時刻になり、今の意味（D2H が終わった後の時刻）と同じ
- 更新手法の焼き付け: なし。コアは「不透明な値」しか知らず、device の判断は bindings（DLPack の device 種別）と codec（torch）に閉じる
- 副 stream: ノードが自分の device 以外の CUDA device の値を読むとき、ATen はコピーを「呼んだスレッドの、元の device の current stream」に積む。ワーカーでは直前に走った別ノードの stream が残っているので、そのままだとノード同士が同じ stream で絡む。Driver がノードごとに元の device の副 stream を持ち、hook のたびに current にする

## 内部設計の詳細

### 経路ごとの動作

| 書き手 → 読み手 | 書く | 読む | ホストを経由するか |
| --- | --- | --- | --- |
| CUDA d → 同じ CUDA d のノード | d の上で clone、自分の stream を待つ | 読み手の stream で clone、`record_stream(読み手の stream)` | しない |
| CUDA d → 別の CUDA D のノード | 同上 | 読み手の d の副 stream でコピー（ATen が D の stream と event で前後を揃える）、`record_stream(副 stream)` | コードでは経由しない。P2P の無い GPU（この機械の 4090 と 4060 は `can_device_access_peer` が False）ではドライバが内部でホストを中継する |
| CUDA d → CPU のノード | 同上 | 副 stream で `.to("cpu")`（同期） | D2H 1 回（今は D2H + memcpy） |
| CPU → CUDA のノード | 今どおりホストのバッファ | 今どおり H2D | 今どおり |
| ホストの `read` / `read_with_time` / `subscribe` / `wait_for` | — | ホストのスレッドの current stream で `.to("cpu")`、`record_stream` | D2H 1 回。送るたびの D2H は無くなり、読んだときだけになる |
| ホストの `write`・`inputs=`・`initial`・復元 | ホストのバッファ（今どおり） | 今どおり | 今どおり |
| `state_dict` | — | 停止中は `codec.peek` で保持テンソルの参照だけ取り、再開後に `state.cpu` で CPU へ | 停止を延ばさない |

### Driver（`src/mujou/driver.py`）

- `self._streams`: 副 stream（`read_devices` の各 device に `torch.cuda.Stream(d)`）を先に、ノード自身の stream を最後に並べた tuple。CPU のノードで `read_devices` が空なら空（CUDA を初期化しない、今のテストを保つ）
- `_use(streams)`: 順に `torch.cuda.set_stream`。**`set_stream` は current device をその stream の device に変える**（実測で確認）ので、自分の device の stream を最後にし、CUDA のノードは最後に `set_device(自分の device)` になる。CPU のノードで副 stream だけあるときは device を気にしない
- 同期（`update` 後・`restore` 後・`capture` 前後）は今どおり自分の stream にだけ行う。副 stream のコピーは、CUDA の読み手なら ATen が自分の stream に event で繋ぎ、CPU の読み手なら `.to("cpu")` が同期なので、別に待たなくてよい

### コアの変更（`crates/mujou-core/src/buffer.rs`・`slot.rs`）

- `Messages::fill` は各フィールドに `FieldOut`（状態: 未設定 / `Bytes(Buffer)` / `Held(Held)`）を渡し、`fill` の後に `Value` に変える。プールのバッファは `bytes()` を呼んだフィールドだけが取る（保持するフィールドのためにホストのメモリを取らない）
- `zeros` は全フィールド `bytes()` を 0 で埋める
- 保持値の `Drop` はスロットの mutex の外（今の `publish` の `drop(old)`）で起きる。bindings の `HeldTensor` は `Py<PyAny>` なので、detach 中の drop は PyO3 の reference pool に回り、次に attach したとき（`py.detach` から戻るとき）に decref される

### 書き手の send の順序（codec → bindings）

1. codec: 値を検査し、保持するテンソルを clone（ノードの stream に積まれる）
2. codec: 保持するものがあれば `torch.cuda.current_stream(device).record_event().synchronize()`（ノードの stream だけ。32 本を超えて stream を共有しても、記録より後の他ノードの仕事は待たない）
3. bindings: `__dlpack__(stream=-1)` で shape と dtype を `check`、ホストの値はコピー、保持値は `Arc<HeldTensor>` にして `FieldOut::hold`
4. core: 時刻を付けて公開

## 実装ステップ

1. **core**: `Value` / `Held` / `FieldOut` / 新しい `Fill` を入れ、`Messages`・`Outlet::send`・`Slot::restore`・`InputGroup::write`・`zeros` を移す。`dlpack::import::check` を切り出し、`DEVICE_CUDA` を足す。core の既存テストのクロージャを新しい `Fill` に書き換え、保持値のテストを足す。検証: `just test-rs`
2. **bindings**: `borrow_tensor` に `stream=-1`、`Value::Held`、`copy_into` で `hold`、`PyMessage.values` が保持テンソルを返す。`UnsupportedDevice` の文言を「host memory か CUDA」に直す。docstring を直して `just stub`。検証: `just test-rs`、`tests/mujou/native/test_slot.py`
3. **codec**: `to_core(device=)`、`from_core` の保持テンソルの経路と `record_stream`、`peek`。`port.Wire._values` から `device` を渡す。検証: `tests/mujou/test_codec.py`
4. **wiring / execution / Driver**: `Wiring.read_devices` を `wire` で計算し、`execution._nodes` から `Driver` へ渡す。Driver の副 stream と `_use` の順序。検証: `tests/mujou/test_driver.py`
5. **state**: `_entry` を `peek` に、`cpu` がポートの値も移す。検証: `tests/mujou/test_state.py`
6. **CUDA の結合テスト**（下記のテスト観点）を実機の 2 枚で通す。GIL ありのビルドでも 1 回通す
7. **性能**: `benchmarks/ports.py` で CPU の recv / send が 5% 以上悪化していないことを確かめる（`Fill` と `Value` の変更、`stream=-1` の引数が CPU の hot path に入る）。CUDA の送受信の前後の値を `specs/research/benchmarks.md` に記録
8. **文書**: 下記「更新する文書」。`just run` を通す

## テスト観点

CUDA の観点は `@pytest.mark.cuda` / `@pytest.mark.cuda(2)`（`tests/mujou/conftest.py:109`、無ければ skip）。

- 正常系:
  - 値の一致: CUDA:0 → CUDA:0、CUDA:0 → CUDA:1（`cuda(2)`）、CUDA → CPU のノード、CPU → CUDA、dict のメッセージで CUDA のテンソルとスカラーが混ざる場合（`test_port.py` の `TestCuda`）
  - 受け取る値は読み手の device、`requires_grad=False`、連続（転置したテンソルを送っても `is_contiguous()`）
  - ホストの `read` / `subscribe` / `wait_for` は CPU のテンソルを返し、値が一致する（`test_execution.py`）
  - `state_dict` の木のポートの値は CPU、復元すると同じ値を読む（`test_state.py`）
  - 白箱: CUDA のノードが send した後、`outlet.outbox.read().values[0]` が `cuda:0` の `torch.Tensor` で、送った値の別オブジェクト（`data_ptr` が違う）（`tests/mujou/test_wiring.py` か `native/test_slot.py`）
  - `codec.peek` は保持テンソルをコピーせず返し、`state.capture` の直後（`cpu` の前）の木のポートの値が device にある（`test_state.py`、停止中に D2H しないことの白箱の確認）
  - Driver: CPU のノードと `cuda:1` のノードが `cuda:0` の書き手を読むとき、hook の中の `torch.cuda.current_stream(0)` が既定の stream でも書き手の stream でもない。`cuda:1` のノードの `current_device()` は 1 のまま（`test_driver.py`）
  - CUDA に触れない CPU だけの実行は CUDA を初期化しない（既存の `test_a_node_on_the_cpu_does_not_initialize_cuda` を保つ）
  - core（Rust）: `hold` したフィールドはプールのバッファを取らない（`allocated() == 0`）、メッセージを捨てると保持値が落ちる（`Arc::strong_count`）、版番号の一斉切り替えが保持値でも効く
- 異常系:
  - CUDA のノードの send で shape・dtype が違う → 今と同じ `ValueError` と文言（`check` を共有するので K9 のテストを CUDA でもパラメタ化）
  - sparse の CUDA テンソル → 今どおり `BufferError`（strided でないのでホストの経路に回る）
  - ノードの device 以外の CUDA device のテンソルを send → ホストの経路で今どおり届く
- エッジケース:
  - 読み手が受け取ったテンソルをその場で変えても、別の読み手と `run.read` の値は変わらない
  - 書き手が send した後に自分のテンソルをその場で変えても、読み手の値は送った時点のまま
  - `send_to` だけのポート、Target::All でチャネルが 3 本（1 回の clone を共有）
  - 正常終了の後の `run.read` と `state_dict`
- 数値的オラクル（理論から導いたもの）:
  - **ホストを経由しないこと**: `torch.profiler`（`ProfilerActivity.CUDA`）で `ManualRun.step` の区間を測り、`Memcpy` の名前に `Pageable` か `Pinned` を含むものが無い。対照として CPU → CUDA の実行では `Memcpy HtoD (Pageable -> Device)` が出ることも確かめる（オラクルが検出できる証拠）。P2P の無い 2 枚の間のコピーは `Memcpy DtoH (Device -> Device)` のように端点が Device なので、括弧の中の端点で判定する（実測で確認）。**profiler は triton を読み込み、3.14t で GIL を有効にする**ので、既存の `test_a_node_on_the_cpu_does_not_initialize_cuda` と同じく新しいインタプリタの子プロセスで走らせる（`slow` にしてよい）
  - **書き手の stream が遅れても正しい値を読む**: `mujou.run`（`dedicated_thread=True` の 2 ノード）で、書き手は `torch.cuda._sleep(大きな値)` を自分の stream に積んでから `x.fill_(k)` して send し、`threading.Event` で知らせる。読み手は知らせを待って recv し、`k` と一致する。`_sleep` は PyTorch 自身のテストが使う private な API なので、テストの helper 1 か所に置き理由をコメントする（代わりは 4096×4096 の行列積の繰り返し）
  - **アロケータの再利用で読みかけの値が壊れない**: 読み手は自分の stream に `_sleep` を積んでから recv（コピーが待たされる）、知らせ A を出して知らせ B を待つ。書き手は A を待ち、同じ大きさで違う値を 2 回以上 send（古い保持テンソルが解放され、次の clone がそのブロックを再利用しうる）して B を出す。読み手の値は最初の値と一致する。確率的にしか捉えられないので、実装者は `record_stream` を外すと落ちることを一度確かめ、繰り返し回数を決める
  - いまの `TestCuda.test_manual_steps_match_the_discrete_algorithm` などの離散アルゴリズムとの一致がそのまま通る

## 想定リスク・トレードオフ

- **PyO3 の reference pool**: detach 中の `Py<PyAny>` の drop は pool に回る。free-threaded の 3.14t でも pool が `py.detach` から戻るときに処理され、CUDA のメモリの解放が遅れないことを実装で確かめる（遅れるなら core の `publish` が古いメッセージを返し、bindings が attach した状態で drop する）
- **torch の `__dlpack__` は current device がテンソルの device であることを要求する**（`stream=-1` でも。実測で `BufferError`）。保持するのは hook の中のノード自身の device の値だけで、Driver が最後に自分の device を current にするので満たされる。条件を広げるとき（確認事項 2）はここが壊れる
- **送り先が 32 本を超えて stream を共有**: event で待つので正しさは保たれる（今と同じ）
- **GPU のメモリ**: 各スロットが最新の値を device に 1 つ持つ（outbox とチャネルは同じメッセージを共有）。`record_stream` した塊は、記録した stream がそこまで進むまで再利用されない
- **S2 の代償**: send が `on_update` の途中にあると、その後のホストの仕事と send 前の GPU の仕事が重ならない（1 send あたり約 12 µs）
- **CPU の hot path**: `Fill` の形、`Value` の分岐、`__dlpack__` の引数 1 つが増える。性能の予算（5% 未満）を `benchmarks/ports.py` で確かめる
- 小さいメッセージ（64 要素）は同期の遅延が支配的で、速くなるのは 10% 程度。効くのは大きいメッセージ
- インタプリタの終了時: 保持テンソルは Python のオブジェクトが持つスロットの中にあり、他のテンソルと同じく finalize で落ちる。CUDA の後始末との順序で警告が出ないか、既存の atexit のテストで見る

## 更新する文書

- `specs/messages.md`: 「コアの中の値」（ホストのメモリだけ → ホストのバッファか、CUDA のノードが送った device のコピー）、「書くとき / 読むとき」の箇条、「コアが受け取るのは…CPU か pinned だけ」、メモリ管理（device はキャッシュアロケータ、`record_stream`）、codec の節（`to_core(device=)`、`peek`）
- `specs/nodes.md` の「CUDA」: 「境界で device → host → device」を書き換え、表に「send の中（保持する値を clone した後）」の行と副 stream を足す。pinned memory の 1 行は CPU ↔ CUDA の境界と状態の取得だけの話に直す
- `specs/architecture.md`: 「torch への依存」の GPU の段落、境界の表（Rust はプールのバッファと不透明な値、device の判断は bindings と codec）
- `specs/native-core.md`: 「スロットの実装」に保持値を 1 段落
- `specs/execution-model.md`: 「状態の取得」の手順 5 と「コピーする理由」（スロットの device の値も再開後に CPU へ移す）
- `specs/open-decisions.md`: pinned memory の項目の範囲
- `specs/research/benchmarks.md`: 送受信の前後の値（新しい R 番号）
- `memory/project_gpu_policy.md`: 2026-10-24 にユーザーが覆したことを書き、新しい決定（保持の条件・S2・`record_stream`・副 stream）に書き換える。`memory/MEMORY.md` の 1 行も
- `memory/project_message_types.md`（「コアの値はホストのメモリ上の不変なバッファ」）、`memory/project_native_core_rust.md`（「torch とは DLPack で必ずコピー」）、`memory/project_torch_module.md`（Why の「GPU をコアでサポートしない代わりに」）、`memory/project_pause_and_state.md`（取得の流れ）
- `CLAUDE.md` の「ツーリング」（「torch とコアの間は DLPack で必ずコピー」「GPU はコアでサポートせず」）と skill `locality-and-async` の「ゼロコピー共有・GPU 共有はしない」（skill `edit-dot-claude` の手順で）
- docstring: `codec.py` のモジュールと `to_core`、`port.py` の `Port.send`（「A CUDA tensor is copied to the host」）と `Wire._values`、`Outlet`（Rust の doc comment、`just stub`）
- `docs/en/topics/threads-and-devices.md` と `docs/ja/topics/threads-and-devices.md` の「Devices」: 同じ device のノードの間はホストを経由しないこと、ホストが読むと CPU のコピーになること、send が自分の stream を待つこと
- `CHANGELOG.md` の `## [Unreleased]`: main にはまだ無い（`chore/2026-10-23/release-v0.1.0` にある）。その merge の後に「CUDA のノードの間のメッセージがホストを経由しなくなった」を足す

## 確認事項

1. **stream 間の同期の方式**: 暫定案は S2（書き手が send の中で自分の stream を待ってから公開し、読み手は待たない）。S1（書き手は event を記録するだけで、読み手の stream が `wait_event`）にすると 1 send あたり約 12 µs 速く、send の後の書き手の仕事と GPU が重なるが、読み手が書き手の進行中の GPU の仕事を待ち、スロットの時刻も値が出来上がる前になる。どちらにするか
2. **device に置く値の範囲**: 暫定案は「ノードの hook の send / send_to で、ノード自身の CUDA device の上のテンソル」だけ。ホストの `write`・`inputs=`・`initial`（CUDA のテンソルでも）は今どおりホストのメモリに置く（ホストのスレッドの stream と device を保証できず、副 stream の device も静的に決まらないため）。カメラの画像を GPU に載せてから書くような、ホストからの書き込みも device に置きたいか

## 参照

- 現状: `src/mujou/codec.py`（`_value` の `.to("cpu")`、`_read` の `from_dlpack(...).to(device, copy=True)`）、`src/mujou/driver.py`（`update` 後の同期、`_use`）、`src/mujou/port.py:300-345`（`Wire`）、`src/mujou/wiring.py:87-128`（`wire`）、`src/mujou/state.py:147-165`（`_entry` / `_write`）、`src/mujou/execution.py:761-791`（`_nodes` / `_synchronize`）
- `crates/mujou-core/src/buffer.rs`（`Fill`、`Messages::fill`）、`crates/mujou-core/src/slot.rs:287`（`Outlet::send`）・`:345`（`InputGroup::write`）、`crates/mujou-core/src/dlpack/import.rs:23`（`copy_from`）
- `crates/mujou-python/src/slot.rs`（`borrow_values` / `copy_into` / `PyMessage.values`）、`crates/mujou-python/src/dlpack.rs:33`（`borrow_tensor`）
- 実測のスクリプト: `/tmp/claude-1001/-data-geson-mujou/d2146d35-f3d8-44cf-90d5-5029742d85ef/scratchpad/exp/`（`bench.py`・`prof.py`・`misc.py`）
- skill `locality-and-async`、`testing-strategy`、`refactor-conventions`、`edit-dot-claude`
