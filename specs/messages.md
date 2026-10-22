# メッセージ

ノード間でやり取りするメッセージの型と、torch とコアの間の変換、スロットのメモリ管理を定める。実行モデル全体は [実行モデル](execution-model.md)。

## 値の型

利用者がやり取りするメッセージの中の値は、`torch.Tensor` か Python のスカラー（`int`、`float`、`bool`、`mujou.time.Timestamp`）である。ノードがポートで送受信する値も、ホストが読み書きする値も同じ。mujou は独自のテンソル型も dtype も公開しない。

スカラーを Tensor に包まないのは、時刻や回数のような 1 つの値を受け取るたびに `int()` で取り出さずに済むようにするためである。GPU のノードでは、0 次元の Tensor から `int()` で取り出すたびに device と host の同期が起きる。スカラーは device へ送らず、Python の値のまま受け渡す（2026-10-22）。

サポートする dtype は `TensorSpec.dtypes`（16 / 32 / 64 bit の浮動小数点と bfloat16、8〜64 bit の符号付き整数、`uint8`、`bool`）。表の正本は `mujou.spec.DLPACK`（torch の dtype から DLPack の dtype への対応）で、`TensorSpec.dtypes` はそのキー。torch には `torch.dtype` を DLPack の dtype に直す公開 API が無いので、torch に依存する側が表を持つ。spec.py（層 1）に置けば、`TensorSpec` の検査（K7）と codec（層 2）が下から上へ一方向に読める。

バッチを使う場合は、バッチ次元も shape の一部として扱う。フレームワークはバッチ次元を特別扱いしない。

### コアの中の値

コアは、スロットの値を不変なバッファとしてホストのメモリに持つ。連続した row-major の配置で、GPU などのデバイスには置かない。このバッファは `_native` の内部の型で、利用者には見せない。不変なので、複数の読み手が同じバッファを同時に読んでも、書き手との間でデータ競合が起きない。

torch とコアの間は DLPack で受け渡す。DLPack はデータの場所・dtype・shape・strides を表す標準の形式で、torch は `__dlpack__` と `torch.from_dlpack` で対応している。コアは独自の dtype を持たず、DLPack の dtype の表現をそのまま使う。NumPy の配列を経由すると bfloat16 を受け渡せない。

受け渡しは、読み書きのどちらでも必ずコピーする。ゼロコピーで共有すると、利用者が受け取ったテンソルをその場で書き換えたときに、ほかの読み手が見ているバッファまで変わってしまうからである。

- 書くとき: 値を `detach()` し、GPU 上の値は呼んだスレッドの現在の stream（ノードの hook の中ならノードの stream）でホストへコピーしてから、コアがプールから取ったバッファへコピーする。勾配はノードをまたがない
- 読むとき: コアのバッファの中身を、新しい `torch.Tensor` へコピーする。GPU のノードでは、ホストからデバイスへの転送がこのコピーを兼ねる。ホストの読み出しは CPU のテンソルを返す
- スカラーは DLPack を経由しない。bindings が Python の値をフィールドの dtype のバイト列に直して書き（`__dlpack__` を持たない値をスカラーとして読む）、`BufferView.item()` が Python の値に戻す。どのフィールドがスカラーかは codec が spec から決め、コアの `Schema` は種別を持たない（shape `()` のフィールドになる）
- コアが受け取るのは、ホストから読めるメモリの値（DLPack のデバイス種別が CPU か pinned memory）だけ
- 非連続のテンソルの stride は、Rust の `copy_from` が扱う。Python で `.contiguous()` してから渡すとコピーが 2 回になる
- 密でないテンソル（sparse など）は torch の `__dlpack__` が `BufferError` で断る（検査は書かない）
- DLPack は 1.x（versioned）だけを使う。値は `torch.Tensor` だけで、依存する torch の `__dlpack__` は `max_version` を受け取り、`torch.from_dlpack` も渡すので、旧版（unversioned）の経路に入る場面が無い

## codec

torch とコアの変換は `codec.py` の 1 か所だけが行う。spec の公開の属性（`shape`、`dtype`、`specs`）だけを読む。

- メッセージは、フィールドの順に並べた値の列として扱う。1 つのフィールドのメッセージ（Tensor かスカラー）は名前 `""` のフィールド 1 つ、dict のメッセージはキーごとに 1 つ。正規化は `fields(spec)` の 1 つの `isinstance` だけで、残りの処理は 1 つのフィールドと dict を区別しない
- Python が検査するのは、Rust には見えない構造（フィールドの種別に合う値か、dict のキーに過不足が無いか）だけ（K9 の構造、`TypeError`）。shape と dtype の検査は、Rust の `copy_from` にポートの label を渡して 1 回だけ行う（K9 の shape・dtype、`ValueError`）。Rust は memcpy の健全性のために照合をやめられないので、どのみち消せない方に正本を置き、send の hot path から Python の分岐を消した
- int の範囲の検査も Rust が書き込むときに 1 回だけ行う（K9 の範囲、`OverflowError`）。丸めも折り返しもしない
- Tensor の dtype は変換しない。spec と違う dtype の値は断る。スカラーはフィールドの注釈が Python の型付けで受け付ける値を受け付ける（`bool` は `int`、`int` は `float` として受け付け、読み出すと宣言した型になる）
- `initial`、inputs、復元する値も、段 2 で Outlet や Slot に書くときに同じ経路で検査され、ポートの label で名指しされる。`Port.__init__` の時点では label も Schema もまだ無い

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

seqlock で 1 つのバッファを上書きし、読み手がコピーする方式は持たない。メモリの使用量は増えるが、安全性と速度を優先する。
