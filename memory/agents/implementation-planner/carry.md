# コアの carry（付随データのゼロコピー中継）

> 2026-10-20 implementation-planner。要件は `specs/pc-module.md` の「コアの変更: carry」「メッセージと Port」。1 PR に収める。

## 概要

活動と同じメッセージに付随データ（carry）のフィールドを載せ、途中のノードは中身を torch へ移さずにバッファの `Arc` を複製して中継する。サンプルと付随データが 1 つのスロット・1 つのメッセージに入るので、読み手が組み違えることが構造的に起きない。

## 公開インターフェース案

### Python（`mujou._core.port`、`mujou.Port` / `PrevPort` / `NextPort` として公開）

型引数を 2 つ足す: `Port[S = Never, R = Never, M: (Single, Multi) = Single, CS = Never, CR = Never]`（CS: 送る付随データの型、CR: 受け取る付随データの型）。`PrevPort` / `NextPort` も同じ並び。

- `Port.__init__(self, send=_NO_SPEC, recv=_NO_SPEC, *, multi=False, initial=None, send_carry: MessageSpec[CS] | Port[Any, Any, Single, Any, CS] | None = None, recv_carry: MessageSpec[CR] | None = None) -> None`
  - `send_carry` に spec を渡すと、このポートが付随データを作る（`send` に値を渡す）。Port を渡すと、その Port で最後に `recv` したメッセージの付随データを中継する
  - multi=True の overload は `recv_carry` を受け取らない（下記の検査も行う）
- `Port.send_carry_spec -> MessageSpec[CS] | None`（中継なら中継元の `recv_carry_spec`）
- `Port.recv_carry_spec -> MessageSpec[CR] | None`
- `Port.carry_source -> Port[Any, Any, Any, Any, Any] | None`（中継元。作る側と付随データ無しは `None`）
- `Port.send(self, msg: S, carry: CS | None = None) -> None` / `Port.send_to(self, channel: Channel, msg: S, carry: CS | None = None) -> None`
- `Port.carried(self) -> CR` / `Port.carried(self, key: str) -> torch.Tensor`（overload。最後に `recv` / `recv_with_time` したメッセージの付随データを、所有ノードのデバイスの新しい Tensor へコピーして返す）

### Rust（`crates/mujou-core/src/slot.rs`）

- `Schema::new(fields: Vec<FieldSpec>, carry: Vec<FieldSpec>) -> Result<Schema, SlotError>`、`Schema::carry(&self) -> &[FieldSpec]`
- `Message { buffers: Vec<Arc<Buffer>>, carry_from: usize, time_ns: u64, sent: bool }`。`Message::values(&self) -> &[Arc<Buffer>]`、`Message::carry(&self) -> &[Arc<Buffer>]`（`buffers()` は廃止）
- `Outlet::send_with<E: From<SlotError>>(&self, target: Target, relay: Option<&[Arc<Buffer>]>, fill: impl Fill<E>) -> Result<(), E>`（中継の不一致を `SlotError` で返すため `E: From<SlotError>` を課す。bindings の `E` は既に `SlotError`）
- `Fill` のシグネチャは変えない。意味だけ変える: `relay` が `None` なら値と付随データの全フィールド（スキーマ順）、`Some` なら値のフィールドだけを埋める
- `Slot::new_carry<E>(&self, fill: impl Fill<E>) -> Result<Vec<Arc<Buffer>>, E>`、`InputSlot::new_carry` も同じ（そのスロットの書き手の付随データのプールから取る。復元用）
- `Slot::restore` / `InputSlot::restore` / `InputGroup::write` の `fill` は全フィールド
- `SlotError::CarryMismatch { slot: String, expected: String, actual: String }`（中継する付随データのフィールドがスキーマと違う、またはスキーマに付随データが無い）
- `dlpack::export_versioned(buffer: Arc<Buffer>) -> *mut DLManagedTensorVersioned`、`dlpack::export_legacy(buffer: Arc<Buffer>) -> *mut DLManagedTensor`（`Lent` は `Arc<Message>` でなく `Arc<Buffer>` を持つ）

### bindings（`mujou._core._native`、`just stub` で `_native.pyi` を再生成）

- `Schema(fields, carry=())`（`carry` は fields と同じ `(name, shape, (code, bits, lanes))` の列）
- `Message.buffers -> tuple[BufferView, ...]`（値のフィールドだけ）、`Message.carry -> Carry`
- `Carry`（frozen、不透明）: `Carry.buffers -> tuple[BufferView, ...]`。中身は `Box<[Arc<Buffer>]>`
- `BufferView` は `Arc<Buffer>` を持つ（index を持たない）
- `Outlet.send(values, relay: Carry | None = None)` / `Outlet.send_to(index, values, relay: Carry | None = None)`。`relay` 無しなら `values` は値 + 付随データの全フィールド
- `Slot.new_carry(values) -> Carry` / `InputSlot.new_carry(values) -> Carry`
- `Slot.restore` / `InputSlot.restore` / `InputGroup.write` の `values` は全フィールド

## 理論上の対応

PC-ALM の式には触れない。$x$ と $y$ を同じメッセージで運ぶので、`Output` が $h_{\mathrm{in}}$ と組む $y$ が常に同じサンプルになる（`specs/pc-module.md`「メッセージと Port」）。

- 中継は不変なバイト列の参照の複製で、autograd も torch も経由しない。ノードをまたぐ計算グラフではない
- 中継元は同じノードの Port に限る（構築時に検査）。ノードは自分の Port のスロットしか読まない、という局所性を保つ
- 待ちもバリアも増えない。付随データの読み書きは既存のスロットの mutex の短い区間に乗るだけ
- 付随データの意味（ターゲット・マスク・時刻）はコアに持ち込まない。コアにはフィールドの列としてだけ見える。PC 固有の名前は `mujou.pc` 側（`pc.carry_spec`）

## 実装ステップ

各ステップの最後に、括弧の検証を通す。commit はステップ 1+2、3+4、5、6 の 4 つ（1 と 2 は Rust の API 変更で bindings が壊れるので分けられない）。

1. **Rust コア**（`crates/mujou-core/src/slot.rs`、`dlpack.rs`）（`cargo test -p mujou-core`）
   - `Schema` に `carry` を足す。値のフィールドは 1 つ以上、付随データは 0 個以上。名前の重複は値・付随データそれぞれの中で検査（tensor の付随データは名前 `""`、値の `""` と別の名前空間）。ほかの検査は値と同じ
   - `Messages`（slot.rs:336-385）に付随データのプールを足す。`pools` を値 + 付随データのスキーマ順に並べ、`fill` は `relay` に応じて値のプールだけか全部から取る
   - `Message` を `Vec<Arc<Buffer>>` + `carry_from` にする（slot.rs:301-328）。`Buffer` は既に自分のプールの `Arc` を持つので、最後の `Arc<Buffer>` が落ちたとき元の書き手のプールへ戻る。追加の仕組みは要らない
   - `Outlet::send_with` に `relay` を足す（slot.rs:549-569）。`relay` の各 `Buffer::field()` を `Schema::carry()` と `FieldSpec` の等価で照合し、違えば `CarryMismatch`。照合は `fill` の前に行い、失敗したら何も publish しない
   - `zeros`・`restore`・`InputGroup::write` は全フィールドを埋める。`new_carry` を足す
   - `dlpack::lend` / `export_*`（dlpack.rs:252-340）を `Arc<Buffer>` 単位にする
   - モジュール doc（slot.rs:1-23）に付随データと中継を 1 段落足す
2. **bindings**（`crates/mujou-python/src/slot.rs`、`dlpack.rs`）（`just stub`、`uv run pytest tests/mujou/_core/test_native_slot.py tests/mujou/_core/test_native_stub.py`）
   - `PySchema::new` に `carry` を足す
   - `borrow_values`（slot.rs:45-80）に期待するフィールドの列を渡す形にする（中継なら値だけ、ほかは全部）。エラーの文言は今の形を保つ
   - `PyCarry` を足し、`PyMessage.buffers` を値だけにし、`PyMessage.carry` を足す。`BufferView` を `Arc<Buffer>` にし、`export_capsule`（dlpack.rs:122）も合わせる
   - `PyOutlet.send` / `send_to` に `relay: Option<PyRef<PyCarry>>`。`Arc` の複製は GIL（attach）の下で、`fill` は今どおり detach して行う
   - `new_carry` を `PySlot` / `PyInputSlot` に足す。`SlotError::CarryMismatch` は `ValueError`
   - doc comment を書き、`just stub` で `_native.pyi` を再生成する
3. **Port**（`src/mujou/_core/port.py`）（`uv run pytest tests/mujou/_core/test_port.py tests/mujou/_core/test_node_typing.py`、`just type`）
   - 型引数 `CS` / `CR` を `Port` / `PrevPort` / `NextPort` に足し、`__init__` の overload（各 2 つ）に `send_carry` / `recv_carry` を足す。multi=True の overload には `recv_carry` を置かない
   - `_init`（port.py:133）で検査する（すべて `TypeError`）: `send_carry` / `recv_carry` が spec でも Port でも `None` でもない、`send` が無いのに `send_carry`、`recv` が無いのに `recv_carry`、`multi=True` で `recv_carry`、`send_carry` に渡した Port が `recv_carry` を持たない
   - `initial=` は値だけ。付随データの初期値はゼロ（`initial_carry=` は要るまで足さない）
   - `connect`（port.py:189）で `send_carry_spec` と相手の `recv_carry_spec` も照合する。文言は「`A` carries `X` with what it sends, but `B` expects `Y`」の形で、今の mismatches に並べる
   - `send` / `send_to` に `carry`、`carried` を足し、`Binding` に委ねる。`__repr__` に付随データを出す（無ければ省く）
   - docstring に例を書く（`super()` を呼ぶ形。`feedback_super_calls.md`）
4. **構築・Binding・Session**（`graph.py`、`binding.py`、`session.py`）（`uv run pytest tests/mujou/_core/test_network.py tests/mujou/_core/test_run.py`）
   - `graph.build_tree` で、中継元の Port が同じノードの Port であることを確かめる（ports の対応表で所有ノードの添字を比べる）。違えば `ValueError`。`Network.build` と `mujou.run` の両方で効く
   - `Session._add_outlet`（session.py:88）: `Schema(fields(send), carry=fields(send_carry_spec))`。`initial` の書き込みでは付随データのゼロを末尾に足す
   - `Session._sources`（session.py:110）: `Schema(fields(recv), carry=fields(recv_carry_spec))`。未接続の Port の `InputSlot` も付随データを持つ（中身はゼロ。ホストは書けない）
   - `Binding`（binding.py）: `recv_carry` を持つ Port は、最後に受け取った `_native.Carry` を `_carry` に持つ。初期値は構築時のソースのスロットの `read().carry`（＝初期値の付随データ、ゼロ）。`recv` / `recv_with_time` のたびに、値を読んだのと同じ `Message` の `carry` に差し替える。`has_received` は差し替えない
   - `Binding.send` / `send_to`: 付随データを作る Port は `carry` 必須（無ければ `TypeError`）、値と付随データを `host_tensors` で並べて 1 回の `outlet.send` に渡す。中継する Port は `carry` を渡すと `TypeError`、中継元の Binding の `_carry` を `relay=` に渡す。付随データの無い Port に `carry` を渡すと `TypeError`
   - `Binding.carried(key)`: 所有者の検査、`recv_carry` が無ければ `TypeError`、`key` が DictSpec のキーに無いか TensorSpec に `key` を渡せば `KeyError` / `TypeError`。選んだフィールドの `BufferView` だけを `torch.from_dlpack(...).to(device, copy=True)` でコピーする
   - ホストの入出力（session.py:320-422）: `_writable` は `recv_carry_spec` を持つ Port で `ValueError`（「carries data the host cannot write」）。`_outbox` は `send_carry_spec` を持つ Port で `ValueError`。`write` / `inputs=` / `read` / `read_with_time` / `subscribe` / `wait_for` がすべてここを通る
5. **状態の木**（`state.py`、`session.py`）（`uv run pytest tests/mujou/_core/test_state.py tests/mujou/_core/test_run.py`）
   - 付随データを持つスロットの項目は `(value, time_ns, sent, carried)`、持たないスロットは今の 3 つ組のまま（既存の木と既存テストを変えずに済む。公開前なので互換のためではない）
   - 木に `"carries": {port path: carried}` を足す（`recv_carry` を持つ Port の Binding の `_carry`）。キーは `graph.port_path(port)`（スロット名の接頭辞と同じ）。付随データの Port が無ければ `{}`
   - `make_tree` / `read_tree` / `_slot_restores`（state.py:109-220）で読み書き・検査し、`Session._restore` で `slot.restore(values + carried, ...)` と `binding._carry = source.new_carry(carried)`（中継元のスロットのプールから）。`read_tree` の検査はすべてを先に行い、失敗したら何も変えない（今の方針どおり）
   - モジュール doc の木の図（state.py:6-11）を更新する
6. **仕様と docs**（`just docs-build`、最後に `just run`）
   - `specs/network.md`: 「Port」「接続」に付随データと照合、「Port の読み書き」に `send(carry=)`・`carried`・中継の規則、「ホストの入出力」に付随データの Port の `ValueError`
   - `specs/messages.md`: 「メッセージの型」「スキーマと初期値」に付随データのフィールド、「値の型」の「必ずコピー」に、中継は torch との境界を通らないコア内の共有であることを 1 文、「バッファプール」に中継したバッファは元の書き手のプールへ戻ること
   - `specs/native-core.md`「スロットの実装」: `Message` が `Vec<Arc<Buffer>>` を持つこと、送るたびに小さな確保（`Arc<Buffer>` の箱）が増えること
   - `specs/execution-model.md`「状態の木の中身」: 付随データと `carries`（ユーザーの指定先に無いが、木の定義はここなので更新する）
   - `specs/pc-module.md`「コアの変更: carry」: 確定した API（`send_carry=` / `recv_carry=`）に書き換え、`pc.Input` などの例を合わせる
   - `docs/{en,ja}/user-guide/building-a-network.md` に「Carrying data along a path」節（PC に依らない例: サンプル ID を入力ノードから出力ノードまで中継）、`running.md` にホストが読み書きできないこと、`state.md` に付随データも木に入ること。`docs/en/api/port.md` は docstring から生成されるので、新しいメソッドが出るか確かめる
   - `memory/project_carry.md` を足し、`MEMORY.md` に 1 行（API の決定と理由）

## テスト観点

### Rust（`crates/mujou-core/src/slot.rs` の `mod tests`）

- 正常系:
  - 中継したメッセージの付随データのバッファが、中継元のメッセージのものと `Arc::ptr_eq`（ゼロコピー）
  - 3 段の中継（作る Outlet → 中継 → 中継）を 1000 回回しても、中継する Outlet の付随データのプールの確保数は初期値のゼロの 1 のまま（`allocations` を値と付随データに分けて数えるヘルパーを足す）。作る側の確保数は、生きている保持者の数 + 1 以下で止まる
  - 中継先のメッセージを全部落とすと、付随データのバッファが作った側のプールの free list に戻る（free list の長さで確かめる）
  - 作る側の `fill` は全フィールドを埋め、`zeros` と `restore` は付随データも書く
  - `new_carry` はそのスロットのプールから取る
- 異常系:
  - スキーマ: 値 0 個、値・付随データそれぞれの中の名前の重複、付随データの不正な shape / dtype
  - 中継のフィールドが違う（名前・shape・dtype）、付随データの無いスキーマへの中継は `CarryMismatch` で、何も publish せず、値のバッファはプールへ戻る
  - 中継しない `fill` の失敗で何も publish しない（既存の `failed_fill_publishes_nothing` を付随データ入りでも）
- エッジケース:
  - 付随データ 0 個のスキーマは今と同じ振る舞い（既存テストがそのまま通る）
  - 0 次元の付随データ、tensor の付随データ（名前 `""`）
  - `export_*` が `Arc<Buffer>` を保持し、メッセージを落としても deleter までバッファが生きる（`lent_tensor_keeps_message_alive_until_deleted` を書き換え）
- 並行（スナップショットの一貫性）:
  - 書き手 A が値と付随データに同じ `k` を書き続け、中継スレッド B が A のスロットを `load` し、そのメッセージの値から作った値と、同じメッセージの付随データの中継で C へ送る。C の読み手 4 本が、どのメッセージでも値 = 付随データ、`k` が減らないことを確かめる
  - `readers_never_see_a_message_of_another_slot` に付随データを持つスロットと中継を混ぜる

### Python

- `tests/mujou/_core/test_native_slot.py`
  - `Schema(fields, carry=...)` の検査が `ValueError`
  - 付随データ入りの send → `Message.buffers` は値だけ、`Message.carry.buffers` は付随データだけ、全 dtype でビット一致
  - ゼロコピー: `relay=` で送ったあと、中継先と中継元の `torch.from_dlpack(view).data_ptr()` が等しい。作る側の send では等しくない
  - `relay=` のとき `values` に付随データを含めると `ValueError`（個数）、含めないと通る。付随データ無しのスキーマへの `relay=` は `ValueError`
  - `new_carry`、全フィールドの `restore`
- `tests/mujou/_core/test_port.py`
  - `__init__` の各 `TypeError`（上のステップ 3 の列挙）と、`send_carry_spec` / `recv_carry_spec` / `carry_source` / `repr`
  - `connect`: 付随データの spec が違う、片方だけが持つ、で `TypeError`（向きと Port 名を含む）。一致すれば通る。中継する Port の `send_carry_spec` は中継元の `recv_carry_spec` と同じオブジェクト
- `tests/mujou/_core/test_node_typing.py`: `carried()` が `CR`、`carried("y")` が `torch.Tensor`、`send(carry=...)` の型、multi=True で `recv_carry` を渡すと型エラー
- `tests/mujou/_core/test_network.py`（構築）: 中継元が別のノードの Port なら `Network.build` と `mujou.run` が `ValueError`
- `tests/mujou/_core/test_run.py`（`mujou.testing.run` で決定的に）
  - Source（`send_carry=spec`）→ Relay（`prev` が `recv_carry`、`next` が `send_carry=self.prev`）→ Sink の 3 ノード。Source は step `k` で値 `k` と付随データ `k` を送り、Relay は受けた値に 100 を足して送る。Sink の `carried()` は常に `recv()` の値 - 100 と一致する。手動ステップの順を Source, Sink, Relay など、2 スロット方式なら組み違う順でも一致する（スナップショットの一貫性の oracle）
  - Relay が `recv` せずに `send` すると、前回 `recv` した付随データ（最初は初期値のゼロ）を添える。`carried()` も `recv` の前はゼロ、`recv` の後は同じメッセージの値
  - ゼロコピー: Session の内部（`_slots`）から、Relay の外向きスロットと Source のチャネルのスロットの `Message.carry` の `data_ptr` が一致する
  - `carry` を渡し忘れ（作る側）、渡し過ぎ（中継・付随データ無し）、`recv_carry` の無い Port の `carried` で `TypeError`、所有者以外の `carried` で `RuntimeError`、無いキーで `KeyError`
  - ホスト: `recv_carry` を持つ未接続 Port への `write` と `inputs=`、`send_carry` を持つ Port の `read` / `read_with_time` / `subscribe` / `wait_for` が `ValueError`。付随データの無い Port は今どおり
  - CUDA のノードでは `carried()` がノードのデバイスに載る（CUDA が無ければ skip。既存の CUDA テストの印に倣う）
- `tests/mujou/_core/test_state.py`
  - 付随データのスロットの項目が 4 つ組、`carries` に Port のパスで入る
  - 数値的オラクル: 同じ手動スケジュールで、途中で `state_dict` → 新しいインスタンスに復元して続けた結果が、止めずに回した結果と一致する（`carried()` と Sink の受けた値を含む）。Relay が「`recv` してから中継元が更新され、そのあと `send`」となる順で止め、復元後の中継が `carries` の値であることを確かめる
  - 付随データの shape 違い・欠け・余り、`carries` のキーの過不足で `ValueError`、何も変わらない
- free-threaded と GIL あり: 上のテストは CI の両ビルドで回る。自由実行（`mujou.run`、workers 既定）で Source → Relay → Sink を数百ステップ回し、Sink が毎回の `recv` と `carried()` の一致を数え続ける smoke test を 1 つ（`slow`）。Rust の並行テストが本体の保証

## 想定リスク・トレードオフ

- 送るたびに `Arc<Buffer>` の箱がフィールドの数だけ確保される（今も `Arc<Message>` と `Vec` は毎回確保している）。大きな記憶域はプールのまま。問題になれば `Arc<Buffer>` 自体をプールする
- 中継する Binding が最後に受けたメッセージの付随データを持ち続けるので、作る側のプールは中継するノードの数だけバッファが多めに要る。値のバッファは持たない（`Message` でなく `Carry` を持つ理由）
- `Port` の型引数が 5 つになり、overload が重くなる。代わりに `carried() -> object` にすると PC の `Output` で cast が要る
- `carry=` 1 つで spec なら受け取る向き・Port なら送る向き、というスケッチは採らない（確認事項 1）
- 状態の木の項目の長さがスロットによって 3 と 4 に分かれる。全部 4 つ組にする案は既存テストと docs の書き換えが増えるので採らない
- 付随データの初期値を宣言する口（`initial_carry=`）は無い。ゼロ（bool なら False）で、PC の `mask` は教師なし扱いになるので害は無い
- 中継元を同じノードに限る検査を `graph.build_tree` に置く。`Port` は所有ノードを `__init__` の時点で知らないため
- node.py の変更は要らない見込み（`Port.carried` は `Binding` だけで完結し、状態の木は Session が組む）。別 agent の `register_state` の編集と衝突しない

## 確認事項

1. **Port の宣言の形**: 推奨は `send_carry=` / `recv_carry=` に分ける形（暫定案）。`self.prev = PrevPort(send=G, recv=H, recv_carry=PASSENGER)` / `self.next = NextPort(send=H, recv=G, send_carry=self.prev)` / 作る側 `self.next = NextPort(send=X, send_carry=PASSENGER)`。承認済みのスケッチ `carry=` は、spec を渡すと受け取る向きになるので、付随データを作る側（`pc.Input.next`、送る向きの spec）を書けない。両向きに付随データを持つ Port も書けない
2. **中継のスナップショットを状態の木に入れるか**: 推奨は入れる（`"carries"`、暫定案）。入れないなら、復元後の最初の `recv` までは復元した中継元のスロットの付随データを使うことになり、止めた時点で「`recv` 後に中継元が更新済み」だと、`recv` より先に `send` するノードで再開後の中継が止めずに回した場合とずれる
3. **ホストが付随データの Port を読むとき**: 推奨はエラー（`ValueError`、暫定案）。代わりに付随データを捨てて値だけ返す案もある（PC の `Output.next` は付随データを持たないので、MVP ではどちらでも困らない）

## 参照

- `specs/pc-module.md`（「メッセージと Port」「コアの変更: carry」）、`specs/network.md`、`specs/messages.md`、`specs/native-core.md`、`specs/execution-model.md`「状態の木の中身」
- `crates/mujou-core/src/slot.rs`（Message 301-328、Messages 336-385、Outlet::send_with 549-569、InputGroup::write 649-693、tests 741-）、`crates/mujou-core/src/dlpack.rs`（lend 252-340）
- `crates/mujou-python/src/slot.rs`（borrow_values 45、BufferView 163、PyMessage 203、PyOutlet 307）、`crates/mujou-python/src/dlpack.rs:122`
- `src/mujou/_core/port.py`（_init 133、connect 189、send 307）、`binding.py`（recv 64、send 85）、`session.py`（_add_outlet 88、_sources 110、_writable 382、_outbox 394、_restore 451）、`state.py`（make_tree 109、read_tree 143）、`graph.py`（build_tree 100）
- skill `locality-and-async`、`testing-strategy`、`refactor-conventions`、memory `feedback_super_calls.md`・`feedback_no_compat_before_release.md`
