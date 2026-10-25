# メッセージ

ノード間でやり取りするメッセージの型と、torch とコアの間の変換、スロットのメモリ管理を定める。実行モデル全体は [実行モデル](execution-model.md)。

## 値の型

利用者がやり取りするメッセージの中の値は、`torch.Tensor` か Python のスカラー（`int`、`float`、`bool`、`mujou.time.Timestamp`）である。ノードがポートで送受信する値も、ホストが読み書きする値も同じ。mujou は独自のテンソル型も dtype も公開しない。

スカラーを Tensor に包まないのは、時刻や回数のような 1 つの値を受け取るたびに `int()` で取り出さずに済むようにするためである。GPU のノードでは、0 次元の Tensor から `int()` で取り出すたびに device と host の同期が起きる。スカラーは device へ送らず、Python の値のまま受け渡す（2026-10-22）。

サポートする dtype は `TensorSpec.dtypes`（16 / 32 / 64 bit の浮動小数点と bfloat16、8〜64 bit の符号付き整数、`uint8`、`bool`）。表の正本は `mujou.spec.DLPACK`（torch の dtype から DLPack の dtype への対応）で、`TensorSpec.dtypes` はそのキー。torch には `torch.dtype` を DLPack の dtype に直す公開 API が無いので、torch に依存する側が表を持つ。spec.py（層 1）に置けば、`TensorSpec` の検査（K7）と codec（層 2）が下から上へ一方向に読める。

バッチを使う場合は、バッチ次元も shape の一部として扱う。フレームワークはバッチ次元を特別扱いしない。

### コアの中の値

コアは、スロットの値をフィールドごとに 2 通りで持つ。

- ホストのメモリの不変なバッファ: 連続した row-major の配置。スカラーと、CPU にあるテンソルと、下の場合以外の CUDA のテンソルはこちら
- 書き手が渡した不透明な値（コアの `Held`）: 次の密なテンソルは、codec が device の上で作ったコピーを bindings がそのまま持つ。コアは中身を読まず、device・stream・event も知らない
    - CUDA のノードが hook の `send` / `send_to` で送った、そのノードの device の上のテンソル。ただしそのポートの接続先に CUDA のノードが 1 つ以上あるときだけ（2026-10-24、接続先の条件は 2026-10-25）。読み手が CPU のノードだけか、接続先が無い（ホストだけが読む）ポートは、send でホストのメモリへ移す。配線のときに `Wire.cuda` で決める
    - ホストの `write`（と `inputs=`）で渡した CUDA のテンソル。そのテンソルの device の上に置く（2026-10-25）

バッファは `_native` の内部の型で、利用者には見せない。どちらの値も不変なので、複数の読み手が同じ値を同時に読んでも、書き手との間でデータ競合が起きない。

device の上に置くのは、CUDA のノードの間でメッセージが device → host → device と往復する無駄を無くすためである。往復は送るたびに同期の D2H を、読むたびに H2D を払い、100 万要素で 1 回の send に 0.5〜1.6 ms かかっていた（[research/benchmarks.md](research/benchmarks.md) の「CUDA のメッセージを device に置いた後」）。CPU のノードとホストしか読まないポートを device に置かないのは、置くと D2H が送り手から読み手へ移り、小さいメッセージでは送り手の stream の待ちと読み手の D2H の同期の 2 回になって遅くなるからである（[research/benchmarks.md](research/benchmarks.md) の同じ節）。`initial` と復元はホストのメモリに置く。

torch の `__dlpack__` は、呼ぶスレッドの current device がテンソルの device であることを要求する。ホストのスレッドの current device は保証できず、1 回の write に複数の device のテンソルが混ざりうるので、ホストの write の clone は codec の `_OnItsDevice` に包み、`__dlpack__` の間だけ自分の device を current にさせる。読み手の codec は包みを外してから読む。ノードの send では包まない。hook の中で別の device を current にしている間（`torch.cuda.device(...)`）は、device に置かずホストのメモリへ移す。

torch とコアの間は DLPack で受け渡す。DLPack はデータの場所・dtype・shape・strides を表す標準の形式で、torch は `__dlpack__` と `torch.from_dlpack` で対応している。コアは独自の dtype を持たず、DLPack の dtype の表現をそのまま使う。NumPy の配列を経由すると bfloat16 を受け渡せない。

受け渡しは、読み書きのどちらでも必ずコピーする。ゼロコピーで共有すると、利用者が受け取ったテンソルをその場で書き換えたときに、ほかの読み手が見ているバッファまで変わってしまうからである。

- 書くとき: 値を `detach()` する。勾配はノードをまたがない。ホストのメモリに置く値は、GPU 上なら呼んだスレッドの現在の stream でホストへコピーしてから、コアがプールから取ったバッファへコピーする。device に置く値は、呼んだスレッドのその device の current stream（ノードの hook ならノードの stream）の上で連続な row-major に `clone` し、その stream に event を記録してホストで待ってから渡す（下の「stream の同期」）
- 読むとき: コアのバッファの中身を、新しい `torch.Tensor` へコピーする。GPU のノードでは、ホストからデバイスへの転送がこのコピーを兼ねる。device に置いた値は、読み手の current stream の上で読み手の device へコピーし、元の値に `record_stream` を付ける。ホストの読み出しは CPU のテンソルを返す（device に置いた値ならここで 1 回だけ D2H）
- スカラーは DLPack を経由しない。bindings が Python の値をフィールドの dtype のバイト列に直して書き（`__dlpack__` を持たない値をスカラーとして読む）、`BufferView.item()` が Python の値に戻す。どのフィールドがスカラーかは codec が spec から決め、コアの `Schema` は種別を持たない（shape `()` のフィールドになる）
- bindings が受け取るのは、ホストから読めるメモリの値（DLPack のデバイス種別が CPU か pinned memory）と CUDA の値。CUDA の値は shape と dtype を照合（`dlpack::import::check`）してから、オブジェクトごと持つ。ほかのデバイスは `TypeError`
- 非連続のテンソルの stride は、Rust の `copy_from` が扱う。Python で `.contiguous()` してから渡すとコピーが 2 回になる。device に置く値は codec の `clone` が連続にする
- 密でないテンソル（sparse など）は torch の `__dlpack__` が `BufferError` で断る（検査は書かない）
- DLPack は 1.x（versioned）だけを使う。値は `torch.Tensor` だけで、依存する torch の `__dlpack__` は `max_version` を受け取り、`torch.from_dlpack` も渡すので、旧版（unversioned）の経路に入る場面が無い
- bindings は `__dlpack__` に `stream` を渡さない。依存する torch（2.14.1 以上）の既定は `-1`（同期しない）で、CUDA の値の同期は書き手が済ませている

### stream の同期

ノードはそれぞれ自分の CUDA stream で計算するので、device に置いた値を別のノードが読むには stream をまたぐ順序が要る。書き手が send の中で、`clone` を積んだ自分の stream に event を記録してホストで待ち、それから公開する（2026-10-24 の S2）。ホストの write も同じで、clone を積んだ各 device の current stream を待ってから公開する（複数のポートへの一斉の write では、全ポートの値を待ってから版番号を進める）。読み手は待たない。公開された値は device の上で出来上がっているので、読み手の stream は何にも依存せずにコピーを積める。

| 案 | 内容 | 判断 |
| --- | --- | --- |
| S2（採用） | 書き手が自分の stream を待ってから公開する | 読み手は書き手の進行中の GPU の仕事を待たない。スロットの時刻は「値が device の上で出来上がった後」で、ホストのメモリに置く値（D2H の後の時刻）と同じ意味になる |
| S1 | 書き手は event を記録するだけで公開し、読み手の stream が `wait_event` する | 1 send あたり約 12 µs 速いが、読み手の GPU の仕事が書き手の進行中の計算を待ち、`on_update` の後の同期を通じて読み手のホストまで書き手を待つ（「ノードは他のノードを待たない」が GPU の側で崩れる）。スロットの時刻も値が出来上がる前になる |

待つのは event で、stream 全体ではない。PyTorch の stream のプール（device ごとに 32 本）を超えると複数のノードが 1 本を共有するが、event は記録より後に別のノードが積む仕事を待たない。

読み手の側の順序は次のとおり。

- 同じ device の読み手: 自分の stream でコピーする
- 別の CUDA device の読み手: ATen は、コピーを「呼んだスレッドの、元の device の current stream」に積み、読み手の device の stream と event で前後を揃える。ワーカーには直前に走った別のノードの stream が残っているので、Driver がノードごとに元の device の副 stream を持ち、hook のたびに current にする（[ノード](nodes.md#cuda)）
- CPU のノードとホスト: 副 stream（ホストはそのスレッドの current stream）の上で `.to("cpu")` する。同期のコピーなので、終わってから返る
- ホストが書いた値の読み手: どの device に書かれるかは write まで分からないので、副 stream は作らない（`Wiring.read_devices` に入れない）。読み手の device と違う device に書かれた値は、読み手のスレッドの元の device の current stream でコピーする。値は出来上がっているので正しいが、その stream がワーカーで直前に走った別のノードのものなら、そのノードの GPU の仕事の後ろに並ぶ（性能だけの問題。2026-10-25 の判断。全 device に副 stream を作ると、使わない GPU にも CUDA の context を作る）

P2P の無い GPU の組（計測した機械の例は [research/benchmarks.md](research/benchmarks.md) の「CUDA のメッセージを device に置いた後」）では、device の間のコピーをドライバが内部でホストを中継する。コードはホストのメモリを経由しない。

## codec

torch とコアの変換は `codec.py` の 1 か所だけが行う。spec の公開の属性（`shape`、`dtype`、`specs`）だけを読む。

- メッセージは、フィールドの順に並べた値の列として扱う。1 つのフィールドのメッセージ（Tensor かスカラー）は名前 `""` のフィールド 1 つ、dict のメッセージはキーごとに 1 つ。正規化は `fields(spec)` の 1 つの `isinstance` だけで、残りの処理は 1 つのフィールドと dict を区別しない
- Python が検査するのは、Rust には見えない構造（フィールドの種別に合う値か、dict のキーに過不足が無いか）だけ（K9 の構造、`TypeError`）。shape と dtype の検査は、Rust の `copy_from`（device に置く値は、そこから切り出した `check`）にポートの label を渡して 1 回だけ行う（K9 の shape・dtype、`ValueError`）。Rust は memcpy の健全性のために照合をやめられないので、どのみち消せない方に正本を置き、send の hot path から Python の分岐を消した
- int の範囲の検査も Rust が書き込むときに 1 回だけ行う（K9 の範囲、`OverflowError`）。丸めも折り返しもしない
- Tensor の dtype は変換しない。spec と違う dtype の値は断る。スカラーはフィールドの注釈が Python の型付けで受け付ける値を受け付ける（`bool` は `int`、`int` は `float` として受け付け、読み出すと宣言した型になる）
- `initial`、inputs、復元する値も、段 2 で Outlet や Slot に書くときに同じ経路で検査され、ポートの label で名指しされる。`Port.__init__` の時点では label も Schema もまだ無い
- `to_core(spec, msg, label, *, device=None)` の `device` は、CUDA のノードが読むポートなら送るノードの CUDA device、ホストの write なら `THEIR_DEVICES`（各テンソルが自分の device に残る）、それ以外（CPU のノード、CUDA のノードが読まないポート、`initial`、復元）は `None`。ノードの値は `Wire.cuda` が段 2 で決めるので、send のたびに `device.type` や読み手を見ない（`device.type` は 1 回 約 0.1 µs、CPU の hot path の予算の大半になる）
- `from_core(spec, views, device)` は、ホストのバッファを `torch.from_dlpack` で、device に置いた値を `.to(device, copy=True)` で読む。区別は `type(view) is BufferView`（`isinstance(view, torch.Tensor)` は hot path で 0.1 µs 余計にかかる）
- `peek(spec, views)` は、device に置いた値をコピーせずに返す（ホストのバッファは CPU のテンソルにコピーする）。状態の取得の停止中だけに使い、呼び手が再開の後に CPU へ移す（`state.cpu`）

## spec

`TensorSpec(shape, dtype=torch.float32)` は、1 つの `torch.Tensor` の shape と dtype の組。

- dtype の既定は `torch.float32`。`torch.get_default_dtype()` に従う形にすると、module のレベルで定義した spec が import の順と既定の dtype の設定によって変わる
- `zeros()`（全 dtype）、`empty()`、`rand(*, generator=None)`（一様分布 $[0, 1)$）は、その形のテンソルを作る。どれも既定では CPU に作り、`device` で置き場所を変えられる。乱数の入口は `generator` 1 つで、ノードの中では `self.generator` を渡す。ほかの値で始めたいときは生成した結果をその場の操作で埋める（`spec.empty().fill_(0.5)`）
- spec は device を持たない。一致は shape と dtype だけで決まり、同じ spec のメッセージがノードごとに別のデバイスにあってよい

スカラーの spec は値を 1 つだけ表し、shape も `zeros` / `empty` も持たない。`0` や `False` はリテラルで書けば足りる。

| spec | 値 | コアの dtype | 初期値 |
| --- | --- | --- | --- |
| `IntSpec(bits=64, signed=True)` | `int` | `bits` は 8 / 16 / 32 / 64、`signed` で int か uint | `0` |
| `FloatSpec()` | `float` | float64 だけ | `0.0` |
| `BoolSpec()` | `bool` | bool | `False` |
| `TimestampSpec()` | `mujou.time.Timestamp` | uint64 の ns | `Timestamp(ns=0)` |

- メッセージのサイズは固定なので、`IntSpec` は bits を必ず持つ（既定は 64 bit 符号付き）。範囲外の値は送るときにエラーになる
- `FloatSpec` は 64 bit に固定する。Python の float は f64 で、32 bit を許すと書いた値と読み出した値がずれる
- `TimestampSpec` は時刻をメッセージで運ぶ（`pc.Activity.time`）。初期値の `Timestamp(ns=0)` は「まだ書かれていない」と時刻 0 の書き込みを区別しない（ユーザーの決定）
- 将来の候補は `ComplexSpec`（DLPack の complex、コード 5）。Python の array は持たない。固定長の列は `TensorSpec` で表せる。種別を足すときの変更は spec のクラス、codec の対応表、bindings の変換の 3 か所に限り、登録の口は作らない

メッセージの型は次のとおり。`FieldSpec` は `TensorSpec | IntSpec | FloatSpec | BoolSpec | TimestampSpec` の別名。

| 型 | spec |
| --- | --- |
| `torch.Tensor` | `TensorSpec` |
| Python のスカラー | `IntSpec`、`FloatSpec`、`BoolSpec`、`TimestampSpec` |
| 値が `torch.Tensor` かスカラーの `TypedDict`（入れ子は許さない） | `DictSpec(TypedDict, **specs)` |

- `DictSpec` は TypedDict と、キーごとの `FieldSpec` を受け取る。キーワード引数のキーが TypedDict のキーと一致しなければ、定義の行で断る（K8）。Python の型は実行時に TypedDict を生成できないので、キーを 2 回書くのは避けにくい（`Annotated` を使う案は [未決定事項](open-decisions.md)）
- 2 つの `DictSpec` が等しいのは、TypedDict のクラスが同一で、各キーの spec が等しいときだけ
- `DictSpec.specs` の値は `FieldSpec` の Union なので、1 つのキーの spec をその種別として使うときは `field(key, kind)` で取り出す（`act_spec.field("h", TensorSpec).shape`）。種別が違えば `TypeError`。種別ごとのメソッド（`tensor("h")` など）にしないのは、種別を足すたびにメソッドが増えるからである
- `DictSpec.zeros()` / `empty()` は、スカラーのフィールドをその初期値で埋める
- `MessageSpec[T]` は spec の総称型で、注釈と総称的な関数に使う。抽象メソッドは持たない（`zeros` / `empty` は `TensorSpec` と `DictSpec` だけにある）
- 値が無いこと（`torch.Tensor | None`）は表さない。「まだ届いていない」はメッセージの時刻 `None` で表す

## スキーマと初期値

スロットのスキーマ（コアの `Schema`）は、フィールドごとの名前・shape・DLPack の dtype の列で、ポートの spec から段 2 で作る。以後は変えない。shape が固定なので、メッセージ用のバッファは同じ大きさのものを使い回せる。

スロットは常に宣言どおりの型の値を持つ。既定の初期値はゼロ（スカラーは上の表の初期値）で、送るポートは `initial=` で初期値を宣言できる。その Port が書くスロット（outbox と各 Channel）は、その値で始まる。ホストが書くスロットの初期値は `mujou.run(..., inputs=)` で渡す。状態を復元するときは木の値が上書きする。`on_start` で `send` しても最初の値にはなるが、復元のときに復元した値を上書きしてしまう。

## 時刻

スロットは書き込みのたびに、書いた時点のその実行の時計（`RunClock`）の時刻を ns で添える。読み手は値と時刻を組で受け取り、値がどれだけ古いかを判断できる。

コアの `Message` は時刻を `Option<u64>` で持ち、`None` は書き手がまだ書いていない（スロットを作ったときのゼロか `initial`）ことを表す。「送られたか」と時刻を 1 回の読み出しから得るので、両者が別のメッセージの情報になる競合が起きない。時刻 0 を番兵にしないのは、実行の開始の時刻（0 かもしれない）に送ることがあるからである。

## メモリ管理: バッファプール

実行中にメッセージ用のメモリをヒープへ毎回確保し、解放する処理は避ける。スロットはバッファのプールを持ち、バッファを再利用する。

- 書き手は、プールから空いているバッファを取り出して新しいメッセージを書き、書き終えたらスロットの参照を差し替える。読み手が見ているバッファに書き手が書き込むことはない
- N 個の読み手は、最新のバッファを読み取り専用で共有する。バッファは参照カウントで管理し、どの読み手からも参照されなくなるとプールへ戻る。Python の読み手は読むときに中身をコピーするので、バッファを参照するのはコピーしている間だけ
- プールに空きが無いときは、書き手が追加で確保し、読み手を待たない。プールは空で始め、縮小しない
- ホストが外から書くスロットは、複数のポートへの書き込みを一斉に切り替えるために、現在の値と版番号付きの次の値の組を持つ（[ネットワーク](network.md#書き込み)）
- device に置く値のプールは torch のキャッシュアロケータで、定常状態で `cudaMalloc` しない。独自の device のプールは持たない（アロケータが二重になる）。device に置くフィールドは、ホストのプールからバッファを取らない
- 書き手の `clone` の塊は、値がスロットから外れると書き手の stream のアロケータへ戻り、次の `clone` がすぐ使いうる。読み手はコピーを積んだあと元の値に `record_stream(コピーした stream)` を付け、コピーが終わるまで再利用させない。読み手の Driver が読んだ値を `on_update` の後まで握る案は、recv のたびの記録と hook ごとの後始末が要るので採らない
- 各スロットは最新の値を device に 1 つ持つ（outbox とチャネルは同じメッセージを共有する）
- 保持する値（bindings の `Py<PyAny>`）は、それを持つ最後の `Message` が落ちたときに落ちる。書き手の send・ホストの write・復元が値を差し替えたときは、スロットの mutex の外、GIL を離した区間の中（`publish` や `InputGroup::write` の `drop(old)`）で落ち、PyO3 はその decref を reference pool に回して、`detach` から戻るときに同じスレッドで処理する。読み手の `Message`（Python のオブジェクト）が最後の参照なら、GIL を持ったまま落ちる

seqlock で 1 つのバッファを上書きし、読み手がコピーする方式は持たない。メモリの使用量は増えるが、安全性と速度を優先する。
