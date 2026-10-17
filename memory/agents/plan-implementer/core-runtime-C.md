# core-runtime WP-C（Python の構造）の判断ログ

計画書: `memory/agents/implementation-planner/core-runtime.md` の WP-C。

## IF 変更通知（WP-D 向け）

計画書の名前・引数・戻り値・例外の型は変えていない。次は計画書に書かれていなかった点、または細部の差分。

- **Port の型付けは計画どおり self 型の overload で解けた。`recv_all` 案には切り替えていない。** `recv()` は相手 1 つなら `R`、複数なら `dict[Channel, R]` と pyright strict で決まる（`tests/mujou/test_port.py` の `assert_type` を `just type` が検査する）
    - ただし `__init__` の overload は、メソッドスコープの型変数（`def __init__[S2, R2](self: Port[S2, R2, Single], ...)`）で書いた。pyright は `__init__` の self 注釈にクラスの型変数を使えず（reportInvalidTypeVarUse）、self 注釈つきの `__init__` で解けなかった型変数にはクラスの既定値も入れない
    - 省略した向きを `Never` に解くため、overload の既定値は `_NO_SPEC = cast(MessageSpec[Never], None)`（実行時は `None`）。実装のシグネチャの既定値は計画どおり `None`
    - 制約: `Port(None, spec)` のように `None` を明示して渡すと、その向きの型変数が解けずに残る。省略（キーワード引数）で書く
- **クラスの型変数は PEP 695 の `class Port[S = Never, ...]` ではなく、旧来の `TypeVar("S", infer_variance=True, default=Never)` で書いた。** pre-commit の `debug-statements` / `check-docstring-first` が Python 3.12 で ast を読み、型パラメータの既定値（3.13 以降の構文）を parse できないため。意味（既定値・variance の推論）は計画書と同じ。`NextPort` の相手の数は別の型変数 `MN`（既定 `Multi`）。pre-commit の `default_language_version` を 3.14 にすれば PEP 695 に戻せる
- **`PrevPort` / `NextPort` の `S`・`R` にも既定値 `Never` を置いた**（計画書は `PrevPort[S, R, M = Single]`）。片方向の Port で型が Unknown にならないようにするため
- **`_port.freeze(port)` と `_port.is_frozen(port)` を足した。** build が Port を凍結するためのモジュール関数（Port の `_frozen` を `_network` から直接触ると pyright strict の reportPrivateUsage になるため）
- `PrevPort` / `NextPort` の `__init__` は `super().__init__` ではなく private な `Port._init(send, recv, multi)` を呼ぶ（overload の self 型が基底の overload と噛み合わないため）。D が Port の初期化に状態を足すなら `_init` に足す
- `Graph.port_path(port)` は木の外の Port に `ValueError`
- `Network._graph` はクラス属性の既定値 `None`（サブクラスが `super().__init__()` を呼ばなくてよい）。`Component` と `Network` は `__init__` を持たない
- `_network.py` は `Node` を `TYPE_CHECKING` と `build` の中で import する（`_node.py` が `_network.Component` を import する循環のため）

### レビュー（`memory/agents/code-reviewer/core-runtime-C.md`）を受けた IF の追加・変更

- **追加: `_network.graph(network: Network) -> Graph`。** build 済みのネットワークの `Graph` を返す。未 build なら `RuntimeError`。テストと D（`mujou.run` の「未 build なら build」の判定、Session の構築）は `network._graph` ではなくこれを使う（M1）
- **変更: `Port`（3 種とも）の `send` / `recv` は `TensorSpec` / `DictSpec` / `None` 以外を `TypeError` で拒否する（S1）。** D は Port の spec を `Spec` へ `cast` してよい
- **変更: `DictSpec` は各キーの spec が `TensorSpec` でないと `TypeError`、キーの無い TypedDict を `ValueError` で拒否する（S1・N2）**
- **変更: `split` は strided でない tensor（sparse など）を `TypeError` で拒否する（N3）。** `join(TensorSpec, ...)` は tensor がちょうど 1 つであることを確かめる（N4）
- **変更: build の検査を 2 つ足した（`ValueError`）。** クラス属性として宣言した Port（S2）、入れ子のネットワークが自分の部分木の外のノードの Port を公開すること（S3）
- 同じ PrevPort / NextPort を 2 つの属性名で持つ場合は 1 つと数える（N8。`Component.connect` も同じ）
- `TensorSpec` は shape を最初に tuple にしてから検査する（S4。iterator も正しく受け付ける）

## 計画外の判断

- `connect` の検査順: build 済み（`RuntimeError`）→ 自分自身との接続（`ValueError`）→ spec の不一致（`TypeError`）→ 両向きとも spec なし（`TypeError`）→ 相手 1 つの Port の 2 本目（`ValueError`）。失敗したときはどちらの `channels` も変えない
- spec を 1 つも持たない `Port()` は作れる。接続しようとした時点で `TypeError`（計画書の「両向きとも None は拒否」を connect でだけ行った）
- 同じ Port を 1 つのノードが 2 つの属性名で持つのも、2 つのノードが持つのと同じく `ValueError`（パスが 1 つに決まらないため）
- build 済みのネットワークを含む木の build は `RuntimeError`（Port が凍結済み）
- 「ルートが Node」の検査: `build` は `Network` にしかないので、別の検査は置いていない（`Node` には `build` が無い）
- `TensorSpec.rand` に `seed` と `generator` の両方を渡すと `ValueError`
- `split` の例外: メッセージの型の違い・dtype の違いは `TypeError`、dict のキーの過不足・shape の違いは `ValueError`。どれもメッセージに `where`（dict ならキー名つき `where['key']`）を含む
- `DictSpec` は `NotRequired` も `total=False` も `__optional_keys__` で拒否（`TypeError`）
- `Channel(a, b)` のコンストラクタは制限していない（docstring で「`Port.connect` だけが作る」と書いた）
- テストは `_network.graph()` を通して `Graph` を読む（レビュー M1 の後）
