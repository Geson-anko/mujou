# core-runtime WP-C（Python の構造）レビュー

対象: worktree `.claude/worktrees/network-structure`、ブランチ `feature/2026-10-16/network-structure`（`chore/2026-10-15/core-scaffold..HEAD` = 7d132c4）。計画書 `memory/agents/implementation-planner/core-runtime.md` の WP-C・WP-D と、`docs/network.md`・`docs/messages.md`・`docs/nodes.md` に照らした。

## verdict: request-changes

must-fix は 1 件で、修正は小さい。should-fix のうち S1（spec の実行時検査）と S2（クラス属性の Port）は、D の結合前に入れておくと D が楽になる。

## must-fix

### M1. テストが private 属性 `Network._graph` を直接読んでいる

- 対象: `tests/mujou/test_network.py:50-58`（`built_graph`）。`TestBuild` のパス・`port_path`・`ports` の検証はすべてここを通る
- 問題: skill `refactor-conventions` の「`_` prefix の関数・メソッド・属性はテストしない。公開の名前を通して確かめる」に反する。実装者は判断ログで意図的な例外と書いているが、`_port.freeze` / `_port.is_frozen` と同じ方法で避けられる
- 根拠: `refactor-conventions` の「private モジュール規約」の 4 項目目、`testing-strategy` のモック・private の方針
- 修正の方向（参考）: `_network.py` に `_` の付かないモジュール関数（例: `graph(network) -> Graph`。未 build なら `RuntimeError`）を置き、テストと D の `mujou.run`（「未 build なら build」の判定）がそれを使う。`Network` に公開属性を足すと利用者の属性名とぶつかるので、モジュール関数にするのがよい
- 補足: 原因は計画書で、`Graph` を `self._graph` に置き公開の入口を設けなかったこと。D は `Network.run` の中から `self._graph` を `Session` に渡せるので、D の結合自体は止まらない
- 確信度: 中（規約違反なのは確実。計画書が原因なので、orchestrator の判断で should-fix に下げてもよい）／深刻度: 低

## should-fix

### S1. Port と DictSpec が spec の型を実行時に検査せず、誤りが D の結合で遅れて AttributeError になる

- 対象: `src/mujou/_port.py:110-118`（`Port._init`）、`src/mujou/_spec.py:169-195`（`DictSpec.__init__` の `**specs`）
- 問題: `Port(send=(2, 3))` も `Port(send=torch.zeros(3))` も通り、`connect` もこの値同士の `!=` で比べるだけになる。`DictSpec(F, a=(2,))` も通り、`fields()` で初めて `AttributeError: 'tuple' object has no attribute 'shape'` が出る（再現済み）。`TensorSpec` が次元と dtype を検査しているのとも揃っていない
- D への影響: Port は `MessageSpec[S] | None` を持つが、`fields` / `split` / `join` は `Spec`（`TensorSpec | DictSpec[Any]`）しか受け取らない。D の `_session.py` と `_port.py` では呼ぶたびに `cast` か `isinstance` で絞る必要がある。Port の構築時に `isinstance(spec, (TensorSpec, DictSpec))` を確かめて `TypeError` にしておけば、D は根拠のある `cast` を 1 か所に置くだけで済む
- 根拠: `docs/messages.md`「Port には、メッセージの型に応じた spec を渡す」。構築時に失敗させる方針（計画書の確認事項 12 と同じ考え方）
- 確信度: 高（挙動）／中（直すべきかどうか）／深刻度: 中

### S2. クラス属性として宣言した Port を build が黙って無視し、インスタンス間で共有される

- 対象: `src/mujou/_network.py:42-49`（`_ports` は `vars(component)` だけを見る）
- 再現: `class N(Node): port = Port(send=TensorSpec((2,)))` を木に置いて build すると成功するが、`graph.ports` は `{}` になる。D の結合後は、この Port の `recv` / `send` が実行中も「not bound」の `RuntimeError` になる。2 つのインスタンスは同じ Port を共有するので、「Port の所有ノードは 1 つ」という build の検査（単一書き手の原則）もすり抜ける
- 根拠: `docs/nodes.md:116`「`__init__` で Port を宣言したインスタンスが…」。`Node` の docstring は「Declare the ports in `__init__`」と書くだけで、誤った宣言を検出しない
- 修正の方向（参考）: build でクラスの MRO 上の Port 属性を見つけたら `ValueError` にするか、少なくとも `Node` の docstring に「クラス属性の Port は無視される」と書く
- 確信度: 高（挙動）／中（検出までするかどうか）／深刻度: 中

### S3. 入れ子のネットワークが自分の部分木の外の Port を公開できる

- 対象: `src/mujou/_network.py:191-202`
- 再現: `class Inner(Network): def __init__(self, n): self.prev = n.prev` を、`n` を兄弟に持つ `Outer` に置いて build すると成功する。公開した Port が木全体の `ports` にあるかだけを確かめ、公開したネットワークの部分木にあるかは確かめていない
- 根拠: `docs/network.md`「ネットワークは内側の Port を自分の属性として持つことで外向きに公開し」。計画書の「Network が公開する Port は木の中のノードが所有する」は字面どおりに満たしている
- 確信度: 低（docs の解釈による）／深刻度: 低

### S4. `TensorSpec` に iterator を渡すと、shape が黙って `()` になる

- 対象: `src/mujou/_spec.py:69-86`
- 問題: 検査のループで `shape` を 1 度消費し、`self._shape = tuple(shape)` を最後に作っている。そのため `TensorSpec(d for d in (2, 3))` は scalar の spec になる。型注釈は `Sequence[int]` なので pyright は弾くが、実行時の検査があるのに素通りする
- 根拠: コードを読んだ結果（ループの後で `tuple(shape)` を作る順序）
- 修正の方向（参考）: 先に `tuple(shape)` を作り、それを検査する
- 確信度: 高／深刻度: 低

## nit

- N1. `Port(None, spec)` と、`TensorSpec | None` 型の変数を `send` に渡した場合（`Port(send=maybe_spec, recv=s)`）は、型が `Port[S2@__init__, Tensor, Single]` になり、メソッドの型変数が漏れる（pyright で確認）。前者は判断ログと docstring に書いてある。後者（条件付きで spec を作るケース）は書かれていない。`_port.py:56` の docstring に「`None` を値として渡すと型が決まらない」と一言足すとよい（確信度: 高）
- N2. `DictSpec` は空の TypedDict を受け付け、`fields()` が `[]` を返す。フィールドが 0 個の `Schema` を A のコアが受け付けるかは確かめていない。拒否するか、A の挙動を確かめておく（`_spec.py:169`、確信度: 中）
- N3. `split` は sparse tensor と `requires_grad=True` の tensor をそのまま通す。`requires_grad` は D が `detach` するので問題ない。sparse は D の DLPack で分かりにくいエラーになる。`layout` の検査を足すかは D と合わせて決める（`_spec.py:246-255`、確信度: 中）
- N4. `join(TensorSpec, tensors)` は `len(tensors) == 1` を確かめず、`tensors[0]` だけを返す。dict 側は `zip(strict=True)` で検査しているので揃っていない（`_spec.py:293-294`、確信度: 高）
- N5. `Graph` は `@dataclass(frozen=True)` なので `__hash__` が生成されるが、`MappingProxyType` を含むので `hash(graph)` は `TypeError` になる。`eq=False` にするか放置かを選ぶ（`_network.py:65`、確信度: 高、実害はほぼない）
- N6. `Any` に理由のコメントが無い箇所がある: `_port.py:187-206`（`recv` / `recv_with_time` の実装の `-> Any`）、`_port.py:325-334`（`Channel`）、`_port.py:360-367`、`_spec.py:216`（`cast(DictSpec[Any], ...)`）。`refactor-conventions`「`Any` は理由をコメントで添えたときだけ」（確信度: 中）
- N7. `recv` の docstring（`_port.py:191-197`）は「while unconnected」のときだけ初期値を返すと読める。実際は、接続済みでも相手がまだ送っていなければ初期値を返す。そのことと、毎回コピーした新しい Tensor を返すこと（D の挙動）は、D で本体を入れるときに書き足すとよい（確信度: 中）
- N8. 同じ `NextPort` を 2 つの属性名で公開したネットワークは「2 NextPort attributes」で拒否される。同じ Port を 2 回数えているだけなので、メッセージが紛らわしい（`_network.py:164-170`、確信度: 中）
- N9. build が失敗しても Port は凍結されず、やり直せる（再現済み）。この性質を確かめるテストが無い。仕様として残すなら 1 本あるとよい（確信度: 低）
- N10. `Channel(a, b)` は誰でも作れるが、両端の `channels` には登録されない。判断ログと docstring で「`connect` だけが作る」と明示しているので許容範囲（確信度: 低）
- N11. list / dict に入れたノードは無視される（future work で、docstring にも書いてある）。その結果「the network has no node」や「connected to a port outside the network」になるが、原因が分かりにくい。メッセージに「直接の属性だけを辿る」と添えるとよい（確信度: 低）

## WP-D が結合するときに困る IF の確認

- 計画書の WP-C の名前・引数・戻り値・例外の型はすべてそろっている（`DLPACK_DTYPES`、`MessageSpec`、`TensorSpec`、`DictSpec`、`Spec`、`fields` / `split` / `join`、`Single` / `Multi`、`Port` 3 種、`Channel`、`Component.connect`、`Network.build`、`Graph`、`Node`）
- 型付け: `recv` / `recv_with_time` の self 型 overload は pyright で期待どおりに解決する。`Node` サブクラスの `self.prev.recv()` は `Tensor`、`self.next.recv()` は `dict[Channel, Tensor]` になる。受けるだけの Port への `send` と、TypedDict のキー違いの `send` は型エラーになる。`multi: bool`（非リテラル）は 2 つの overload の union になるが、エラーにはならない
- D が対応すべき点: `Graph` を取り出す private でない入口（M1）、`MessageSpec` から `Spec` への絞り込み（S1）、Port の初期化に状態を足すなら `Port._init` に足すこと（判断ログのとおり）。`Graph.ports` は「ノードの順 → 属性の定義順」の挿入順で決定的なので、スロットの生成順もこれに従えばよい

## 検証結果

- just run: pass（pre-commit の変更なし、pytest 123 passed、coverage は `_spec` / `_port` / `_network` / `_node` とも 100%、pyright 0 errors）
- pyright は `tests/` も検査対象（15 files）で、`test_port.py` の `assert_type` も検査されている
- 成果物汚染（`</content>` 等）: なし
