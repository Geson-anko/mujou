# python-structure（`src/mujou/_core` のモジュール構成の整理）の判断ログ

依頼: 学習則の実装に入る前に `_core` の Python のモジュール構成を整える（2026-10-17、構成の監査をもとに）。公開 API（`mujou.*` / `mujou.time` / `mujou.testing`）と振る舞いは変えない。ブランチ `refactor/2026-10-17/python-structure`。

## 新しい構成

モジュールの表と import の向きは `specs/torch.md` の「モジュールの構成」にある。要点は次のとおり。

- `component.py`: `Component`（`Node` と `Network` の基底、`connect`）と、属性の Port の列挙（`attribute_ports` / `sides`）。`network.py` から出した
- `hooks.py`: hook の中かどうかの判定（`in_hook` / `enter_node_thread` / `running_hook`）。`node.py` から出した。どこも import しない葉
- `graph.py`: 構築の仕組み（`Graph`、`build_tree`、構築済みの印 `_BUILT` と `BUILD_LOCK`、`build_node`、`unbuild_tree`、`start_once`、`ports`）。`network.py` から出した。`mujou.Graph` はここから再公開する
- `network.py`: 公開の `Network` / `Compose` と `RunCompose`、`unbuild`
- `binding.py`: `Binding` と `as_spec` / `message_value`。`session.py` から出した
- `state.py`: 状態の木（`make_tree` / `read_tree`）。`Run.state_dict` の形を定めるのはここだけ
- `lifetime.py`: 動いている実行の登録（`add` / `discard` / `running`）と終了時の停止（`_shutdown_all` / `_stop_at_exit` / atexit）。実行を `Stoppable` Protocol で持ち、`session` を import しない。`running(Session)` で型の付いた一覧を返す
- `spec.host_tensors(spec, msg, where=, copy=)`: ホストのメモリへのコピーを 1 つにした。送信・ホストの書き込み・スロットの復元は `copy=False`（コアがコピーする）、Port の `initial` は `copy=True`（保持するので）
- 改名: `run.start` → `run.start_run`（`Session.start` / `Runtime.start` と紛れるため）、`spec.M` → `spec._M`（API リファレンスの出力は変わらないことを `site/` の diff で確かめた）

## 判断

- 循環の解消: `node` は `component` を、`network` は `node` を先頭で import する。`network.py` の末尾の import と、import の順序に頼るコメントを消した。`timer.host_only` の遅延 import も消えた
- `Component.connect` の注釈 `Node | Network` は `TYPE_CHECKING` の import でだけ解決する。Python 3.14 の注釈の遅延評価で、griffe（API リファレンス）・pyright・pydoc（`help`）はそのまま。ただし `inspect.signature(mujou.Node.connect)` を既定の `annotation_format=VALUE` で呼ぶと `NameError` になる（以前は解決した）。`component` から `node` / `network` を実行時に import すると循環が戻るので、この差は受け入れた。`Compose.__init__` の注釈は `network.py` が `Node` を先頭で import するので今までどおり解決する
- `_side` の「expose one」のヒントは、以前は `isinstance(component, Network)` で出していた。`component.py` は `Network` を import できないので、「`Component` で、コアの `Node`（`_native.Node`）でないもの」で判定する（`Component` の子は `Node` と `Network` だけ）。`connect` の `other` に部品でないものが来たときにヒントを出さないのも以前と同じ
- `unbuild` は `network.py` に残した。根の `Network._graph` を忘れる処理があり、`graph.py` から `Network` を参照すると循環するため。ロックの下で `graph.unbuild_tree`（Port の解凍と印の削除、根を返す）を呼び、続けて `_graph` を消す。以前と同じく全体を 1 つのロックの下で行う
- `TestComponentConnect` は `tests/mujou/_core/test_component.py` に移さなかった。`test_network.py` の共通の部品（`Hidden` / `Stack` など）を使うので、移すとテストを書き換えることになる。`TestBuild` は公開の `Network.build` のテストなので `test_network.py` に残した
- `port.py` の `TYPE_CHECKING` の `Binding` は残した（`binding` が `port` を import するので、実行時に import すると循環する）。`session.py` の `TYPE_CHECKING` の `Graph` は、`graph` を実行時に import しているので普通の import にした
- `_CPU`（`torch.device("cpu")`）は `session.py` と `state.py` の両方にある。1 行の定数なので共有のために import を足さなかった
